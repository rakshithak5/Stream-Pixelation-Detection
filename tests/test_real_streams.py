"""
Fix #4 — Real-world labeled test dataset from broadcast .ts files.

Ground truth was established by:
1. Extracting all frames via ffmpeg (avoids OpenCV frame-dropping)
2. Computing raw frozen_block_score and mb_ratio_score per frame
3. Visual validation: frozen >= 0.20 + SAD < 8 = GLITCH,
   mb_ratio >= 0.10 + MVAD >= 0.40 = GLITCH, SAD > 30 + no artifacts = SCENE_CUT

This test suite validates the full detection pipeline against real broadcast
stream artifacts — not just synthetic test images.

Run:
    python3 tests/test_real_streams.py
"""
import sys, os, cv2, numpy as np, uuid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.video_detector import video_detector
from src.core.detection import (
    compute_frozen_block_score,
    compute_macroblock_ratio_score,
)
from src.models.mvad_wrapper import model_manager

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, 'data', 'uploads')
FRAME_DIRS = {
    'glitch_65177.ts':                   '/tmp/frames_glitch_65177',
    'glitch_65178.ts':                   '/tmp/frames_glitch_65178',
    'glitch_67890.ts':                   '/tmp/frames_glitch_67890',
    'media-u2i5e5nyz_b2628000_64969.ts': '/tmp/frames_media-u2i5e5nyz_b2628000_64969',
    'media-u2i5e5nyz_b2628000_64970.ts': '/tmp/frames_media-u2i5e5nyz_b2628000_64970',
}

# ── Ground truth labels ───────────────────────────────────────────────────────
# Derived from visual validation + raw signal analysis.
# Format: {filename: {frame_number: 'GLITCH' | 'CLEAN' | 'SCENE_CUT'}}
# Only frames with high confidence labels are included.
# Frames not listed are excluded from evaluation (ambiguous).

GROUND_TRUTH = {
    'glitch_65177.ts': {
        **{i: 'GLITCH'    for i in range(0, 11)},    # 0-10: frozen blocks
        **{i: 'SCENE_CUT' for i in range(11, 16)},   # 11-15: scene transitions
        **{i: 'GLITCH'    for i in range(16, 56)},   # 16-55: frozen blocks
        56: 'SCENE_CUT',
        **{i: 'GLITCH'    for i in range(57, 62)},   # 57-61: frozen blocks
        **{i: 'CLEAN'     for i in range(62, 68)},   # 62-67: clean
        **{i: 'GLITCH'    for i in range(68, 73)},   # 68-72: frozen blocks
        73: 'CLEAN',
        75: 'SCENE_CUT',
        **{i: 'GLITCH'    for i in range(77, 79)},   # 77-78: frozen blocks
    },
    'glitch_65178.ts': {
        **{i: 'GLITCH'    for i in range(1, 17)},    # 1-16: frozen blocks
        17: 'SCENE_CUT',
        **{i: 'GLITCH'    for i in range(18, 25)},   # 18-24: frozen blocks
        **{i: 'CLEAN'     for i in range(27, 33)},   # 27-32: clean
        **{i: 'GLITCH'    for i in range(33, 46)},   # 33-45: frozen blocks
        50: 'SCENE_CUT',
        **{i: 'GLITCH'    for i in range(51, 70)},   # 51-69: frozen blocks
        **{i: 'GLITCH'    for i in range(89, 106)},  # 89-105: frozen blocks
    },
    'glitch_67890.ts': {
        **{i: 'GLITCH'    for i in range(1, 22)},    # 1-21: frozen blocks
        22: 'SCENE_CUT',
        **{i: 'GLITCH'    for i in range(23, 49)},   # 23-48: frozen blocks
        49: 'SCENE_CUT',
        **{i: 'GLITCH'    for i in range(50, 85)},   # 50-84: frozen blocks
        85: 'SCENE_CUT',
        **{i: 'GLITCH'    for i in range(86, 100)},  # 86-99: frozen blocks
    },
    'media-u2i5e5nyz_b2628000_64969.ts': {
        **{i: 'GLITCH'    for i in range(1, 3)},     # 1-2: frozen blocks
        3:  'SCENE_CUT',
        **{i: 'GLITCH'    for i in range(4, 23)},    # 4-22: frozen blocks
        23: 'SCENE_CUT',
        **{i: 'GLITCH'    for i in range(24, 52)},   # 24-51: frozen blocks
        52: 'SCENE_CUT',
        **{i: 'GLITCH'    for i in range(53, 72)},   # 53-71: frozen blocks
        **{i: 'CLEAN'     for i in range(72, 73)},   # 72: clean
        **{i: 'GLITCH'    for i in range(73, 100)},  # 73-99: frozen blocks
    },
    'media-u2i5e5nyz_b2628000_64970.ts': {
        **{i: 'GLITCH'    for i in range(1, 17)},    # 1-16: frozen blocks
        17: 'SCENE_CUT',
        **{i: 'GLITCH'    for i in range(18, 35)},   # 18-34: frozen blocks
        35: 'SCENE_CUT',
        **{i: 'GLITCH'    for i in range(36, 42)},   # 36-41: frozen blocks
        42: 'SCENE_CUT',
        **{i: 'CLEAN'     for i in range(43, 54)},   # 43-53: clean
        **{i: 'SCENE_CUT' for i in range(54, 58)},   # 54-57: scene cuts
        **{i: 'GLITCH'    for i in range(77, 80)},   # 77-79: frozen blocks
        **{i: 'GLITCH'    for i in range(81, 89)},   # 81-88: frozen blocks
        **{i: 'GLITCH'    for i in range(91, 100)},  # 91-99: frozen blocks
    },
}


def run_test(ts_name: str, frame_dir: str) -> dict:
    """Run detection on all labeled frames and compare against ground truth."""
    gt = GROUND_TRUTH.get(ts_name, {})
    if not gt:
        print(f'  No ground truth for {ts_name}, skipping.')
        return {}

    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    if not frame_files:
        print(f'  No frames in {frame_dir} — run ffmpeg extraction first.')
        return {}

    sid = f'test_{uuid.uuid4().hex[:8]}'
    video_detector.reset_stream(sid)

    tp = tn = fp = fn = 0
    errors = []
    prev = None

    for fname in frame_files:
        fn_num = int(fname.replace('frame_', '').replace('.jpg', '')) - 1
        if fn_num not in gt:
            # Skip unlabeled frames
            frame = cv2.imread(os.path.join(frame_dir, fname))
            if frame is not None:
                video_detector.analyze_frame(frame=frame, stream_id=sid, frame_number=fn_num)
                prev = frame.copy()
            continue

        frame = cv2.imread(os.path.join(frame_dir, fname))
        if frame is None:
            continue

        result = video_detector.analyze_frame(frame=frame, stream_id=sid, frame_number=fn_num)
        label  = gt[fn_num]
        detected = result['artifact_detected']

        if label == 'GLITCH':
            if detected:
                tp += 1
            else:
                fn += 1
                errors.append(f'  FN frame {fn_num}: {result.get("decision_maker","")} conf={result["confidence"]:.3f}')
        elif label in ('CLEAN', 'SCENE_CUT'):
            if not detected:
                tn += 1
            else:
                fp += 1
                errors.append(f'  FP frame {fn_num} (gt={label}): {result.get("decision_maker","")} conf={result["confidence"]:.3f}')

        prev = frame.copy()

    video_detector.reset_stream(sid)

    total   = tp + tn + fp + fn
    acc     = (tp + tn) / total * 100 if total > 0 else 0
    prec    = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall  = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1      = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0

    return {
        'file': ts_name, 'total': total,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'accuracy': acc, 'precision': prec, 'recall': recall, 'f1': f1,
        'errors': errors,
    }


def main():
    print('\n' + '='*80)
    print('  REAL-WORLD STREAM TEST SUITE — Fix #4')
    print('  Ground truth from visual validation of 5 broadcast .ts files')
    print('='*80)

    all_results = []
    for ts_name, frame_dir in FRAME_DIRS.items():
        print(f'\n  Testing {ts_name}...')
        r = run_test(ts_name, frame_dir)
        if not r:
            continue
        all_results.append(r)
        status = '✅ PASS' if r['accuracy'] >= 85 and r['recall'] >= 90 else '❌ FAIL'
        print(f'  {status}  Acc={r["accuracy"]:.1f}%  Prec={r["precision"]:.1f}%  '
              f'Recall={r["recall"]:.1f}%  F1={r["f1"]:.1f}  '
              f'TP={r["tp"]} TN={r["tn"]} FP={r["fp"]} FN={r["fn"]}')
        for e in r['errors'][:5]:
            print(e)

    if not all_results:
        print('\n  No results — extract frames first with ffmpeg.')
        return

    # Aggregate
    total_tp = sum(r['tp'] for r in all_results)
    total_tn = sum(r['tn'] for r in all_results)
    total_fp = sum(r['fp'] for r in all_results)
    total_fn = sum(r['fn'] for r in all_results)
    total    = total_tp + total_tn + total_fp + total_fn
    agg_acc  = (total_tp + total_tn) / total * 100 if total > 0 else 0
    agg_prec = total_tp / (total_tp + total_fp) * 100 if (total_tp + total_fp) > 0 else 0
    agg_rec  = total_tp / (total_tp + total_fn) * 100 if (total_tp + total_fn) > 0 else 0
    agg_f1   = 2 * agg_prec * agg_rec / (agg_prec + agg_rec) if (agg_prec + agg_rec) > 0 else 0

    print(f'\n{"="*80}')
    print(f'  AGGREGATE RESULTS — {total} labeled frames across {len(all_results)} files')
    print(f'{"="*80}')
    print(f'  Accuracy  : {agg_acc:.1f}%')
    print(f'  Precision : {agg_prec:.1f}%')
    print(f'  Recall    : {agg_rec:.1f}%')
    print(f'  F1 Score  : {agg_f1:.1f}%')
    print(f'  TP={total_tp}  TN={total_tn}  FP={total_fp}  FN={total_fn}')

    overall_pass = agg_acc >= 85 and agg_rec >= 90
    print(f'\n  {"✅ OVERALL PASS" if overall_pass else "❌ OVERALL FAIL"}')
    print(f'  (Thresholds: Accuracy >= 85%, Recall >= 90%)')
    print(f'{"="*80}\n')

    return overall_pass


if __name__ == '__main__':
    result = main()
    sys.exit(0 if result else 1)
