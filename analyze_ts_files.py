"""
Full frame-by-frame pixelation analysis for all .ts files.
Analyzes every single frame, reports exact frame ranges with pixelation.
"""
import cv2
import sys
import uuid
import numpy as np
from pathlib import Path

sys.path.insert(0, '/Users/chandus/Desktop/Videograph/Dev/Stream-Pixelation-Detection')

from src.core.video_detector import video_detector
from src.core.image_detector import image_detector

BASE = Path('/Users/chandus/Desktop/Videograph/Dev/Stream-Pixelation-Detection/data/uploads')

FILES = [
    'glitch_65177.ts',
    'glitch_65178.ts',
    'glitch_67890.ts',
    'media-u2i5e5nyz_b2628000_64969.ts',
    'media-u2i5e5nyz_b2628000_64970.ts',
]

def ts_to_timecode(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def analyze_file(filename: str):
    path = BASE / filename
    cap  = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        print(f"  ❌ Could not open {filename}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration     = total_frames / fps

    stream_id = f"ts_{uuid.uuid4().hex[:8]}"
    video_detector.reset_stream(stream_id)

    frame_results = []
    frame_number  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = video_detector.analyze_frame(
            frame=frame,
            stream_id=stream_id,
            frame_number=frame_number,
            qp_value=None
        )

        ts = frame_number / fps
        frame_results.append({
            'frame':     frame_number,
            'ts':        ts,
            'tc':        ts_to_timecode(ts),
            'detected':  result['artifact_detected'],
            'conf':      round(result['confidence'], 3),
            'type':      result.get('artifact_type') or 'none',
            'severity':  result.get('severity', 'none'),
            'dm':        result.get('decision_maker', ''),
            'bv':        round(result['signals'].get('block_variance', 0), 4),
            'brisque':   round(result['signals'].get('brisque', 0), 1),
            'mvad_b':    round(result['signals'].get('mvad_blockiness', 0), 3),
            'mvad_p':    round(result['signals'].get('mvad_pixelation', 0), 3),
            'edge':      round(result['signals'].get('boundary_edge', 0), 4),
            'bd':        round(result['signals'].get('boundary_density', 0), 3),
            'col_cov':   round(result['signals'].get('artifact_col_cov', 0), 3),
        })
        frame_number += 1

    cap.release()
    video_detector.reset_stream(stream_id)

    # ── Compute pixelation ranges ─────────────────────────────────────────────
    flagged_frames = [r for r in frame_results if r['detected']]
    clean_frames   = [r for r in frame_results if not r['detected']]

    # Group consecutive flagged frames into ranges
    ranges = []
    if flagged_frames:
        start = flagged_frames[0]
        prev  = flagged_frames[0]
        for r in flagged_frames[1:]:
            # Allow gap of up to 2 frames (temporal smoothing)
            if r['frame'] - prev['frame'] <= 3:
                prev = r
            else:
                ranges.append((start, prev))
                start = r
                prev  = r
        ranges.append((start, prev))

    # ── Print report ──────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  FILE: {filename}")
    print(f"  {total_frames} frames  |  {fps:.0f} fps  |  {width}x{height}  |  {duration:.2f}s")
    print(f"{'='*80}")

    # Per-frame table
    print(f"\n  {'Frame':>6}  {'Time':>10}  {'Detected':>9}  {'Conf':>6}  {'Type':<16}  {'Sev':<6}  {'DM':<22}  {'MVAD_B':>6}  {'MVAD_P':>6}  {'BRISQUE':>7}  {'BlockVar':>8}  {'BDens':>6}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*9}  {'-'*6}  {'-'*16}  {'-'*6}  {'-'*22}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*8}  {'-'*6}")

    for r in frame_results:
        flag = '🔴 YES' if r['detected'] else '🟢 no '
        print(f"  {r['frame']:>6}  {r['tc']:>10}  {flag:>9}  {r['conf']:>6.3f}  {r['type']:<16}  {r['severity']:<6}  {r['dm']:<22}  {r['mvad_b']:>6.3f}  {r['mvad_p']:>6.3f}  {r['brisque']:>7.1f}  {r['bv']:>8.4f}  {r['bd']:>6.3f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    n_flagged = len(flagged_frames)
    n_clean   = len(clean_frames)
    pct       = n_flagged / total_frames * 100 if total_frames > 0 else 0

    print(f"\n  ── SUMMARY ──────────────────────────────────────────────────────────────")
    print(f"  Total frames analyzed : {total_frames}")
    print(f"  Pixelated frames      : {n_flagged}  ({pct:.1f}%)")
    print(f"  Clean frames          : {n_clean}  ({100-pct:.1f}%)")

    if ranges:
        print(f"\n  ── PIXELATION RANGES ────────────────────────────────────────────────────")
        print(f"  {'#':<4}  {'Start Frame':>12}  {'End Frame':>10}  {'Start Time':>12}  {'End Time':>10}  {'Duration':>10}  {'Frames':>7}  {'Avg Conf':>9}  {'Type'}")
        print(f"  {'-'*4}  {'-'*12}  {'-'*10}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*7}  {'-'*9}  {'-'*16}")
        for i, (s, e) in enumerate(ranges, 1):
            dur_range = e['ts'] - s['ts']
            # Get all frames in this range
            range_frames = [r for r in frame_results if s['frame'] <= r['frame'] <= e['frame'] and r['detected']]
            avg_conf = np.mean([r['conf'] for r in range_frames]) if range_frames else 0
            types = list(set(r['type'] for r in range_frames if r['type'] != 'none'))
            type_str = '/'.join(types) if types else 'mixed'
            print(f"  {i:<4}  {s['frame']:>12}  {e['frame']:>10}  {s['tc']:>12}  {e['tc']:>10}  {dur_range:>9.3f}s  {e['frame']-s['frame']+1:>7}  {avg_conf:>9.3f}  {type_str}")
    else:
        print(f"\n  ✅ No pixelation detected in this file.")

    # Confidence stats
    if flagged_frames:
        confs = [r['conf'] for r in flagged_frames]
        print(f"\n  ── CONFIDENCE STATS (flagged frames) ────────────────────────────────────")
        print(f"  Min: {min(confs):.3f}  Max: {max(confs):.3f}  Avg: {np.mean(confs):.3f}")
        # Severity breakdown
        sev_counts = {}
        for r in flagged_frames:
            sev_counts[r['severity']] = sev_counts.get(r['severity'], 0) + 1
        print(f"  Severity: " + "  ".join(f"{k}={v}" for k,v in sorted(sev_counts.items())))

    return {
        'file': filename,
        'total_frames': total_frames,
        'flagged_frames': n_flagged,
        'ranges': ranges,
        'frame_results': frame_results,
    }


# ── Run all files ─────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("  FULL FRAME-BY-FRAME PIXELATION ANALYSIS — 5 Transport Stream Files")
print("="*80)

all_results = []
for fname in FILES:
    r = analyze_file(fname)
    if r:
        all_results.append(r)

# ── Cross-file summary ────────────────────────────────────────────────────────
print(f"\n\n{'='*80}")
print(f"  CROSS-FILE SUMMARY")
print(f"{'='*80}")
print(f"\n  {'File':<50}  {'Total':>7}  {'Flagged':>8}  {'%':>6}  {'Ranges':>7}")
print(f"  {'-'*50}  {'-'*7}  {'-'*8}  {'-'*6}  {'-'*7}")
for r in all_results:
    pct = r['flagged_frames'] / r['total_frames'] * 100 if r['total_frames'] > 0 else 0
    print(f"  {r['file']:<50}  {r['total_frames']:>7}  {r['flagged_frames']:>8}  {pct:>5.1f}%  {len(r['ranges']):>7}")

total_f = sum(r['total_frames'] for r in all_results)
total_flag = sum(r['flagged_frames'] for r in all_results)
print(f"\n  {'TOTAL':<50}  {total_f:>7}  {total_flag:>8}  {total_flag/total_f*100:>5.1f}%")
print(f"\n{'='*80}\n")
