"""
Extended accuracy test — 40 images (20 pixelated + 20 clean).
Prints per-image results + full summary with confusion matrix.
"""
import requests
import os

API_URL = "http://localhost:8000/analyze/image"
IMAGE_DIR = "/tmp/pixelation_test_images"

EXPECTED = {
    # ── PIXELATED (expected: True) ──────────────────────────────
    "pix_01_extreme_4x4.jpg":           True,
    "pix_02_heavy_8x8.jpg":             True,
    "pix_03_macroblock_16x16.jpg":      True,
    "pix_04_large_32x32.jpg":           True,
    "pix_05_quality1_jpeg.jpg":         True,
    "pix_06_quality3_jpeg.jpg":         True,
    "pix_07_checkerboard.jpg":          True,
    "pix_08_skin_tone_pixelated.jpg":   True,
    "pix_09_dark_scene.jpg":            True,
    "pix_10_bright_overexposed.jpg":    True,
    "pix_11_text_like.jpg":             True,
    "pix_12_motion_blur_pixelated.jpg": True,
    "pix_13_color_banding.jpg":         True,
    "pix_14_half_pixelated.jpg":        True,
    "pix_15_sky_pixelated.jpg":         True,
    "pix_16_recompressed_5x.jpg":       True,
    "pix_17_pixelated_with_noise.jpg":  True,
    "pix_18_portrait_pixelated.jpg":    True,
    "pix_19_sports_field.jpg":          True,
    "pix_20_night_scene.jpg":           True,
    # ── CLEAN (expected: False) ─────────────────────────────────
    "clean_01_smooth_sine.jpg":         False,
    "clean_02_linear_gradient.png":     False,
    "clean_03_radial_gradient.png":     False,
    "clean_04_high_quality_photo.jpg":  False,
    "clean_05_solid_blue.png":          False,
    "clean_06_solid_green.png":         False,
    "clean_07_sky_gradient.jpg":        False,
    "clean_08_gaussian_blurred.jpg":    False,
    "clean_09_portrait_clean.jpg":      False,
    "clean_10_sports_field.jpg":        False,
    "clean_11_night_scene.jpg":         False,
    "clean_12_heavy_blur_noise.jpg":    False,
    "clean_13_watercolor_like.jpg":     False,
    "clean_14_color_bars.png":          False,
    "clean_15_vignette.jpg":            False,
    "clean_16_overexposed.jpg":         False,
    "clean_17_underexposed.jpg":        False,
    "clean_18_smooth_stripes.jpg":      False,
    "clean_19_bokeh_like.jpg":          False,
    "clean_20_high_quality_resaved.jpg":False,
}

results = []

print("\n" + "="*90)
print("  EXTENDED PIXELATION DETECTION ACCURACY TEST  —  40 images (20 pixelated + 20 clean)")
print("="*90)

# ── PIXELATED section ────────────────────────────────────────────────────────
print("\n┌─ PIXELATED IMAGES ─────────────────────────────────────────────────────────────────┐")
print(f"  {'File':<45} {'Result':<12} {'Detected':<10} {'Conf':>6}  {'Type':<16} {'Severity'}")
print(f"  {'-'*43} {'-'*10} {'-'*8} {'-'*6}  {'-'*14} {'-'*8}")

for filename, expected in EXPECTED.items():
    if not expected:
        continue
    filepath = os.path.join(IMAGE_DIR, filename)
    try:
        with open(filepath, "rb") as f:
            resp = requests.post(API_URL, files={"file": (filename, f)}, timeout=30)
        data = resp.json()
        detected   = data.get("artifact_detected", False)
        confidence = data.get("confidence", 0.0)
        atype      = data.get("artifact_type") or "none"
        severity   = data.get("severity", "-")
        correct    = (detected == expected)
        status     = "✅ CORRECT" if correct else "❌ WRONG  "
        print(f"  {filename:<45} {status}  {str(detected):<8}  {confidence:>5.3f}  {atype:<16} {severity}")
        results.append({"file": filename, "expected": expected, "detected": detected,
                        "correct": correct, "confidence": confidence,
                        "artifact_type": atype, "severity": severity,
                        "signals": data.get("signals", {}),
                        "decision_maker": data.get("decision_maker",""),
                        "corroborating": data.get("corroboration",{}).get("corroborating","")})
    except Exception as e:
        print(f"  {filename:<45} ERROR: {e}")
        results.append({"file": filename, "expected": expected, "correct": None})

# ── CLEAN section ────────────────────────────────────────────────────────────
print("\n├─ CLEAN IMAGES ─────────────────────────────────────────────────────────────────────┤")
print(f"  {'File':<45} {'Result':<12} {'Detected':<10} {'Conf':>6}  {'Type':<16} {'Severity'}")
print(f"  {'-'*43} {'-'*10} {'-'*8} {'-'*6}  {'-'*14} {'-'*8}")

for filename, expected in EXPECTED.items():
    if expected:
        continue
    filepath = os.path.join(IMAGE_DIR, filename)
    try:
        with open(filepath, "rb") as f:
            resp = requests.post(API_URL, files={"file": (filename, f)}, timeout=30)
        data = resp.json()
        detected   = data.get("artifact_detected", False)
        confidence = data.get("confidence", 0.0)
        atype      = data.get("artifact_type") or "none"
        severity   = data.get("severity", "-")
        correct    = (detected == expected)
        status     = "✅ CORRECT" if correct else "❌ WRONG  "
        print(f"  {filename:<45} {status}  {str(detected):<8}  {confidence:>5.3f}  {atype:<16} {severity}")
        results.append({"file": filename, "expected": expected, "detected": detected,
                        "correct": correct, "confidence": confidence,
                        "artifact_type": atype, "severity": severity,
                        "signals": data.get("signals", {}),
                        "decision_maker": data.get("decision_maker",""),
                        "corroborating": data.get("corroboration",{}).get("corroborating","")})
    except Exception as e:
        print(f"  {filename:<45} ERROR: {e}")
        results.append({"file": filename, "expected": expected, "correct": None})

# ── Summary ──────────────────────────────────────────────────────────────────
valid   = [r for r in results if r.get("correct") is not None]
pix_res = [r for r in valid if r["expected"] == True]
cln_res = [r for r in valid if r["expected"] == False]

tp = sum(1 for r in pix_res if r["correct"])
tn = sum(1 for r in cln_res if r["correct"])
fp = sum(1 for r in cln_res if not r["correct"])
fn = sum(1 for r in pix_res if not r["correct"])

total   = len(valid)
correct = tp + tn
acc     = correct / total * 100
prec    = tp / (tp + fp) if (tp + fp) > 0 else 0
rec     = tp / (tp + fn) if (tp + fn) > 0 else 0
f1      = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
spec    = tn / (tn + fp) if (tn + fp) > 0 else 0

print("\n" + "="*90)
print("  SUMMARY")
print("="*90)
print(f"\n  Confusion Matrix:")
print(f"                        Predicted PIXELATED   Predicted CLEAN")
print(f"  Actual PIXELATED      TP = {tp:<3}               FN = {fn}")
print(f"  Actual CLEAN          FP = {fp:<3}               TN = {tn}")
print(f"\n  Overall Accuracy  : {correct}/{total} = {acc:.1f}%")
print(f"  Precision         : {prec:.3f}  (of flagged images, how many were truly pixelated)")
print(f"  Recall            : {rec:.3f}  (of pixelated images, how many were caught)")
print(f"  Specificity       : {spec:.3f}  (of clean images, how many passed correctly)")
print(f"  F1 Score          : {f1:.3f}")

if fp > 0:
    print(f"\n  False Positives (clean flagged as pixelated):")
    for r in cln_res:
        if not r["correct"]:
            print(f"    ❌ {r['file']:<45} conf={r['confidence']:.3f}  type={r['artifact_type']}  dm={r['decision_maker']}")
            sigs = r.get("signals", {})
            if sigs:
                print(f"       signals: " + "  ".join(f"{k}={v:.3f}" for k,v in sigs.items()))

if fn > 0:
    print(f"\n  False Negatives (pixelated missed):")
    for r in pix_res:
        if not r["correct"]:
            print(f"    ❌ {r['file']:<45} conf={r['confidence']:.3f}  type={r['artifact_type']}  dm={r['decision_maker']}")
            sigs = r.get("signals", {})
            if sigs:
                print(f"       signals: " + "  ".join(f"{k}={v:.3f}" for k,v in sigs.items()))

# Confidence distribution
pix_confs = [r["confidence"] for r in pix_res if r.get("confidence") is not None]
cln_confs = [r["confidence"] for r in cln_res if r.get("confidence") is not None]
if pix_confs:
    print(f"\n  Confidence stats (pixelated images):  min={min(pix_confs):.3f}  max={max(pix_confs):.3f}  avg={sum(pix_confs)/len(pix_confs):.3f}")
if cln_confs:
    print(f"  Confidence stats (clean images):      min={min(cln_confs):.3f}  max={max(cln_confs):.3f}  avg={sum(cln_confs)/len(cln_confs):.3f}")

print("\n" + "="*90 + "\n")
