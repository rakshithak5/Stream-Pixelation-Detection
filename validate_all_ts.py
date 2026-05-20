"""
Full frame-by-frame analysis + visual validation for all 5 .ts files.
Uses ffmpeg-extracted frames (avoids OpenCV frame-dropping on .ts).
Compares algorithm output against visual ground truth.
"""
import cv2, sys, numpy as np, uuid, os
sys.path.insert(0, '/Users/chandus/Desktop/Videograph/Dev/Stream-Pixelation-Detection')

from src.core.video_detector import video_detector
from src.core.detection import (
    compute_frozen_block_score,
    compute_macroblock_ratio_score,
    compute_block_variance_score,
)
from src.models.mvad_wrapper import model_manager

FPS = 25.0

FILES = {
    'glitch_65177.ts':                       '/tmp/frames_glitch_65177',
    'glitch_65178.ts':                       '/tmp/frames_glitch_65178',
    'glitch_67890.ts':                       '/tmp/frames_glitch_67890',
    'media-u2i5e5nyz_b2628000_64969.ts':     '/tmp/frames_media-u2i5e5nyz_b2628000_64969',
    'media-u2i5e5nyz_b2628000_64970.ts':     '/tmp/frames_media-u2i5e5nyz_b2628000_64970',
}

def ts_to_tc(s):
    h=int(s//3600); m=int((s%3600)//60); sec=s%60
    return f'{h:02d}:{m:02d}:{sec:06.3f}'

def visual_ground_truth(frozen, sad, mvad_b, mbr, prev_was_glitch):
    """
    Independent visual assessment based on raw signals.
    This is the 'human eye' check — used to validate algorithm output.
    """
    if frozen >= 0.50:
        return 'GLITCH', 'frozen_blocks (definitive — >50% blocks identical to prev)'
    if frozen >= 0.20 and sad < 8.0:
        return 'GLITCH', f'frozen_blocks (frozen={frozen:.2f}, low motion SAD={sad:.1f})'
    if frozen >= 0.15 and mvad_b >= 0.40:
        return 'GLITCH', f'frozen_blocks (frozen={frozen:.2f}, MVAD={mvad_b:.2f})'
    if mbr >= 0.10 and mvad_b >= 0.40:
        return 'GLITCH', f'macroblocking (mbr={mbr:.2f}, MVAD={mvad_b:.2f})'
    if mbr >= 0.15 and frozen >= 0.05:
        return 'GLITCH', f'macroblocking+frozen (mbr={mbr:.2f}, frozen={frozen:.2f})'
    if sad > 30 and frozen < 0.10:
        return 'SCENE_CUT', f'scene transition (SAD={sad:.1f})'
    if frozen >= 0.10 and sad < 5.0 and prev_was_glitch:
        return 'SUSPECT', f'possible glitch tail (frozen={frozen:.2f})'
    return 'CLEAN', f'no artifact signals (frozen={frozen:.2f}, SAD={sad:.1f}, MVAD={mvad_b:.2f})'

def analyze_file(ts_name, frame_dir):
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    if not frame_files:
        print(f'  ERROR: No frames in {frame_dir}')
        return None

    sid = f'val_{uuid.uuid4().hex[:8]}'
    video_detector.reset_stream(sid)

    rows = []
    prev = None
    prev_was_glitch = False

    for fname in frame_files:
        fn = int(fname.replace('frame_','').replace('.jpg','')) - 1
        frame = cv2.imread(os.path.join(frame_dir, fname))
        if frame is None: continue

        # Raw signals for visual validation
        raw_frozen = compute_frozen_block_score(frame, prev) if prev is not None else 0.0
        raw_mbr    = compute_macroblock_ratio_score(frame)
        raw_bv     = compute_block_variance_score(frame)
        raw_mb, _  = model_manager.predict(frame)

        if prev is not None:
            gray_c = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)
            gray_p = cv2.cvtColor(prev,  cv2.COLOR_BGR2GRAY).astype(float)
            sad = float(np.mean(np.abs(gray_c - gray_p)))
        else:
            sad = 0.0

        # Algorithm detection
        r = video_detector.analyze_frame(frame=frame, stream_id=sid, frame_number=fn)

        # Visual ground truth
        vgt, vgt_reason = visual_ground_truth(raw_frozen, sad, raw_mb, raw_mbr, prev_was_glitch)

        algo_det = r['artifact_detected']
        algo_type = r.get('artifact_type') or r.get('signals',{}).get('artifact_type','none')
        if r.get('decision_maker','') == '':
            algo_type = 'scene_cut'

        # Agreement check
        algo_glitch = algo_det
        visual_glitch = vgt in ('GLITCH', 'SUSPECT')
        if algo_glitch and vgt == 'GLITCH':
            agreement = '✅ AGREE'
        elif not algo_glitch and vgt == 'CLEAN':
            agreement = '✅ AGREE'
        elif not algo_glitch and vgt == 'SCENE_CUT':
            agreement = '✅ AGREE'
        elif algo_glitch and vgt == 'SCENE_CUT':
            agreement = '⚠️  FP?  '
        elif not algo_glitch and vgt == 'GLITCH':
            agreement = '❌ MISS  '
        elif algo_glitch and vgt == 'CLEAN':
            agreement = '❌ FP    '
        elif algo_glitch and vgt == 'SUSPECT':
            agreement = '🟡 MAYBE '
        elif not algo_glitch and vgt == 'SUSPECT':
            agreement = '🟡 MAYBE '
        else:
            agreement = '?'

        rows.append({
            'fn': fn, 'tc': ts_to_tc(fn/FPS),
            'algo_det': algo_det,
            'algo_conf': round(r['confidence'], 3),
            'algo_dm': r.get('decision_maker', ''),
            'algo_type': r.get('artifact_type') or 'none',
            'vgt': vgt,
            'vgt_reason': vgt_reason,
            'agreement': agreement,
            'frozen': round(raw_frozen, 3),
            'sad': round(sad, 2),
            'mvad_b': round(raw_mb, 3),
            'mbr': round(raw_mbr, 3),
            'bv': round(raw_bv, 4),
        })

        prev_was_glitch = algo_det
        prev = frame.copy()

    video_detector.reset_stream(sid)

    # ── Print per-frame table ─────────────────────────────────────────────────
    total = len(rows)
    flagged = [r for r in rows if r['algo_det']]
    clean   = [r for r in rows if not r['algo_det']]
    agrees  = [r for r in rows if '✅' in r['agreement']]
    misses  = [r for r in rows if '❌ MISS' in r['agreement']]
    fps_    = [r for r in rows if '❌ FP' in r['agreement']]
    maybes  = [r for r in rows if '🟡' in r['agreement']]

    print(f'\n{"="*100}')
    print(f'  FILE: {ts_name}  |  {total} frames  |  {total/FPS:.2f}s')
    print(f'{"="*100}')
    print(f'  {"Fr":>4}  {"Timecode":>12}  {"Algo":>5}  {"Conf":>6}  {"DM":<24}  {"Visual GT":<10}  {"Agree":>9}  {"Frozen":>7}  {"SAD":>6}  {"MVAD_B":>6}  {"MBR":>6}')
    print(f'  {"-"*4}  {"-"*12}  {"-"*5}  {"-"*6}  {"-"*24}  {"-"*10}  {"-"*9}  {"-"*7}  {"-"*6}  {"-"*6}  {"-"*6}')

    for r in rows:
        aflag = '🔴YES' if r['algo_det'] else '🟢 no'
        print(f'  {r["fn"]:>4}  {r["tc"]:>12}  {aflag}  {r["algo_conf"]:>6.3f}  {r["algo_dm"]:<24}  {r["vgt"]:<10}  {r["agreement"]:>9}  {r["frozen"]:>7.3f}  {r["sad"]:>6.2f}  {r["mvad_b"]:>6.3f}  {r["mbr"]:>6.3f}')

    # ── Ranges ────────────────────────────────────────────────────────────────
    ranges = []
    if flagged:
        s = flagged[0]; p = flagged[0]
        for r in flagged[1:]:
            if r['fn'] - p['fn'] <= 3: p = r
            else: ranges.append((s, p)); s = r; p = r
        ranges.append((s, p))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f'\n  ── SUMMARY ──────────────────────────────────────────────────────────────────────')
    print(f'  Total frames     : {total}')
    print(f'  Algo flagged     : {len(flagged)} ({len(flagged)/total*100:.1f}%)')
    print(f'  Algo clean       : {len(clean)} ({len(clean)/total*100:.1f}%)')
    print(f'  Agreement        : {len(agrees)}/{total} ({len(agrees)/total*100:.1f}%)')
    print(f'  Misses (FN)      : {len(misses)}  {[r["fn"] for r in misses]}')
    print(f'  False Pos (FP)   : {len(fps_)}  {[r["fn"] for r in fps_]}')
    print(f'  Uncertain        : {len(maybes)}  {[r["fn"] for r in maybes]}')

    print(f'\n  ── GLITCH RANGES ────────────────────────────────────────────────────────────────')
    if ranges:
        print(f'  {"#":<3}  {"Start":>6}  {"End":>5}  {"Start TC":>12}  {"End TC":>12}  {"Duration":>9}  {"Frames":>7}  {"Type"}')
        print(f'  {"-"*3}  {"-"*6}  {"-"*5}  {"-"*12}  {"-"*12}  {"-"*9}  {"-"*7}  {"-"*16}')
        for i, (s, e) in enumerate(ranges, 1):
            rf = [r for r in rows if s['fn'] <= r['fn'] <= e['fn'] and r['algo_det']]
            types = list(set(r['algo_type'] for r in rf if r['algo_type'] not in ('none', None)))
            dur = (e['fn'] - s['fn']) / FPS
            print(f'  {i:<3}  {s["fn"]:>6}  {e["fn"]:>5}  {s["tc"]:>12}  {e["tc"]:>12}  {dur:>8.3f}s  {e["fn"]-s["fn"]+1:>7}  {"/".join(types) if types else "frozen_blocks"}')
    else:
        print(f'  ✅ No glitches detected.')

    # Confidence stats
    if flagged:
        confs = [r['algo_conf'] for r in flagged]
        print(f'\n  Confidence (flagged): min={min(confs):.3f}  max={max(confs):.3f}  avg={np.mean(confs):.3f}')

    return {
        'file': ts_name, 'total': total,
        'flagged': len(flagged), 'clean': len(clean),
        'ranges': ranges, 'agrees': len(agrees),
        'misses': misses, 'fps': fps_, 'maybes': maybes,
        'rows': rows,
    }


# ── Run all files ─────────────────────────────────────────────────────────────
print('\n' + '='*100)
print('  FULL VALIDATION — 5 Transport Stream Files (ffmpeg frame extraction)')
print('='*100)

all_results = []
for ts_name, frame_dir in FILES.items():
    result = analyze_file(ts_name, frame_dir)
    if result:
        all_results.append(result)

# ── Cross-file summary ────────────────────────────────────────────────────────
print(f'\n\n{"="*100}')
print('  CROSS-FILE SUMMARY')
print(f'{"="*100}')
print(f'  {"File":<52}  {"Total":>6}  {"Flagged":>8}  {"%":>6}  {"Ranges":>7}  {"Agree%":>7}  {"Misses":>7}  {"FPs":>5}')
print(f'  {"-"*52}  {"-"*6}  {"-"*8}  {"-"*6}  {"-"*7}  {"-"*7}  {"-"*7}  {"-"*5}')
for r in all_results:
    pct = r['flagged']/r['total']*100 if r['total'] > 0 else 0
    agree_pct = r['agrees']/r['total']*100 if r['total'] > 0 else 0
    print(f'  {r["file"]:<52}  {r["total"]:>6}  {r["flagged"]:>8}  {pct:>5.1f}%  {len(r["ranges"]):>7}  {agree_pct:>6.1f}%  {len(r["misses"]):>7}  {len(r["fps"]):>5}')

tf = sum(r['total'] for r in all_results)
tfl = sum(r['flagged'] for r in all_results)
ta = sum(r['agrees'] for r in all_results)
tm = sum(len(r['misses']) for r in all_results)
tfp = sum(len(r['fps']) for r in all_results)
print(f'  {"TOTAL":<52}  {tf:>6}  {tfl:>8}  {tfl/tf*100:>5.1f}%  {"":>7}  {ta/tf*100:>6.1f}%  {tm:>7}  {tfp:>5}')
print(f'\n{"="*100}\n')
