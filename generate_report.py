"""
Generate shareable analysis report for all 5 .ts files.
"""
import pickle, numpy as np
from datetime import datetime

with open('/tmp/ts_analysis_results.pkl', 'rb') as f:
    all_results = pickle.load(f)

NOW = datetime.now().strftime('%Y-%m-%d %H:%M')

def ts_bar(flagged_frames, total, width=60):
    """ASCII timeline bar showing glitch positions."""
    bar = ['.'] * width
    for r in flagged_frames:
        pos = int(r['fn'] / total * width)
        if pos < width:
            bar[pos] = '█'
    return ''.join(bar)

lines = []
lines.append('=' * 90)
lines.append('  STREAM PIXELATION DETECTION — ANALYSIS REPORT')
lines.append(f'  Generated : {NOW}')
lines.append(f'  Algorithm : MVAD (RMViT) + FrozenBlock + MBRatio + BRISQUE hybrid')
lines.append(f'  Decoder   : PyAV (libav/ffmpeg) — full frame accuracy')
lines.append('=' * 90)

for res in all_results:
    fname   = res['file']
    total   = res['total']
    fps     = res['fps']
    flagged = res['flagged']
    ranges  = res['ranges']
    rows    = res['rows']
    pct     = flagged / total * 100 if total > 0 else 0
    dur     = total / fps

    flagged_rows = [r for r in rows if r['det']]
    clean_rows   = [r for r in rows if not r['det']]

    lines.append('')
    lines.append('─' * 90)
    lines.append(f'  FILE    : {fname}')
    lines.append(f'  Duration: {dur:.2f}s  |  {total} frames  |  {fps:.0f} fps  |  {int(total/fps*1000)}ms')
    lines.append('─' * 90)

    # Summary box
    lines.append('')
    lines.append(f'  ┌─ SUMMARY ──────────────────────────────────────────────────────────────────┐')
    lines.append(f'  │  Total frames analyzed : {total:<10}                                        │')
    lines.append(f'  │  Glitched frames       : {flagged:<10} ({pct:.1f}%)                              │')
    lines.append(f'  │  Clean frames          : {len(clean_rows):<10} ({100-pct:.1f}%)                              │')
    lines.append(f'  │  Glitch ranges         : {len(ranges):<10}                                        │')
    if flagged_rows:
        confs = [r['conf'] for r in flagged_rows]
        lines.append(f'  │  Confidence (avg/max)  : {np.mean(confs):.3f} / {max(confs):.3f}                              │')
    lines.append(f'  └────────────────────────────────────────────────────────────────────────────┘')

    # Timeline
    lines.append('')
    lines.append(f'  TIMELINE  (█ = glitch, . = clean, {total} frames → {dur:.2f}s)')
    lines.append(f'  0s{"":>28}{dur/2:.1f}s{"":>27}{dur:.1f}s')
    lines.append(f'  |{ts_bar(flagged_rows, total)}|')

    # Glitch ranges table
    lines.append('')
    if ranges:
        lines.append(f'  GLITCH RANGES:')
        lines.append(f'  {"#":<4}  {"Start Frame":>12}  {"End Frame":>10}  {"Start TC":>12}  {"End TC":>12}  {"Duration":>10}  {"Frames":>7}  {"Avg Conf":>9}  {"Type"}')
        lines.append(f'  {"─"*4}  {"─"*12}  {"─"*10}  {"─"*12}  {"─"*12}  {"─"*10}  {"─"*7}  {"─"*9}  {"─"*16}')
        for i, (s, e) in enumerate(ranges, 1):
            rf = [r for r in rows if s['fn'] <= r['fn'] <= e['fn'] and r['det']]
            avg_conf = np.mean([r['conf'] for r in rf]) if rf else 0
            types = list(set(r['type'] for r in rf if r['type'] not in ('none', None)))
            type_str = '/'.join(types) if types else 'frozen_blocks'
            dur_range = (e['fn'] - s['fn']) / fps
            n_frames = e['fn'] - s['fn'] + 1
            lines.append(
                f'  {i:<4}  {s["fn"]:>12}  {e["fn"]:>10}  {s["tc"]:>12}  {e["tc"]:>12}  '
                f'{dur_range:>9.3f}s  {n_frames:>7}  {avg_conf:>9.3f}  {type_str}'
            )
    else:
        lines.append(f'  ✅ No glitches detected.')

    # Per-frame detail
    lines.append('')
    lines.append(f'  PER-FRAME DETAIL:')
    lines.append(f'  {"Frame":>6}  {"Timecode":>12}  {"Status":>8}  {"Conf":>6}  {"Type":<16}  {"Severity":<8}  {"Decision Maker"}')
    lines.append(f'  {"─"*6}  {"─"*12}  {"─"*8}  {"─"*6}  {"─"*16}  {"─"*8}  {"─"*24}')
    for r in rows:
        status = '🔴 GLITCH' if r['det'] else '🟢 clean '
        lines.append(
            f'  {r["fn"]:>6}  {r["tc"]:>12}  {status}  {r["conf"]:>6.3f}  '
            f'{r["type"]:<16}  {r["sev"]:<8}  {r["dm"]}'
        )

lines.append('')
lines.append('=' * 90)
lines.append('  CROSS-FILE SUMMARY')
lines.append('=' * 90)
lines.append('')
lines.append(f'  {"File":<52}  {"Frames":>7}  {"Glitched":>9}  {"%":>6}  {"Ranges":>7}  {"Duration"}')
lines.append(f'  {"─"*52}  {"─"*7}  {"─"*9}  {"─"*6}  {"─"*7}  {"─"*10}')

total_frames = 0
total_flagged = 0
for res in all_results:
    pct = res['flagged'] / res['total'] * 100 if res['total'] > 0 else 0
    dur = res['total'] / res['fps']
    lines.append(
        f'  {res["file"]:<52}  {res["total"]:>7}  {res["flagged"]:>9}  '
        f'{pct:>5.1f}%  {len(res["ranges"]):>7}  {dur:.2f}s'
    )
    total_frames  += res['total']
    total_flagged += res['flagged']

lines.append(f'  {"─"*52}  {"─"*7}  {"─"*9}  {"─"*6}  {"─"*7}  {"─"*10}')
lines.append(
    f'  {"TOTAL":<52}  {total_frames:>7}  {total_flagged:>9}  '
    f'{total_flagged/total_frames*100:>5.1f}%'
)

lines.append('')
lines.append('  ARTIFACT TYPE LEGEND:')
lines.append('  frozen_blocks  — Transmission/decoder error: blocks copied from previous frame')
lines.append('                   (packet loss, bitstream corruption, decoder failure)')
lines.append('  macroblocking  — H.264/HEVC codec artifact: sharp block boundaries')
lines.append('                   (heavy compression, low bitrate, encoder error)')
lines.append('  pixelation     — Spatial resolution artifact: visible block structure')
lines.append('')
lines.append('  DETECTION METHOD:')
lines.append('  FrozenBlock_definitive — frozen_score >= 0.50 (>50% blocks identical to prev frame)')
lines.append('  FrozenBlock            — frozen_score >= 0.15 AND MVAD >= 0.40')
lines.append('  MVAD+*                 — ML model (RMViT) + spatial corroboration')
lines.append('  MBRatio+MVAD           — Macroblock boundary ratio + ML model')
lines.append('=' * 90)

report = '\n'.join(lines)
print(report)

# Save to file
out_path = '/Users/chandus/Desktop/Videograph/Dev/Stream-Pixelation-Detection/ANALYSIS_REPORT.txt'
with open(out_path, 'w') as f:
    f.write(report)
print(f'\n\nReport saved to: {out_path}')
