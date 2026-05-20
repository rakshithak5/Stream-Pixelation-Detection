"""
Accuracy test: run all test images through the detection API and report results.
"""
import requests
import json
import os

API_URL = "http://localhost:8000/analyze/image"
IMAGE_DIR = "/tmp/pixelation_test_images"

EXPECTED = {
    "pixelated_heavy.jpg":            True,
    "pixelated_moderate.jpg":         True,
    "pixelated_blocky_compression.jpg": True,
    "pixelated_grid_8x8.jpg":         True,
    "pixelated_macroblock_16x16.jpg": True,
    "clean_gradient.png":             False,
    "clean_photo_like.jpg":           False,
    "clean_solid_color.png":          False,
    "clean_smooth_noise.jpg":         False,
    "clean_high_quality.jpg":         False,
}

results = []
print(f"\n{'='*80}")
print(f"  PIXELATION DETECTION ACCURACY TEST")
print(f"{'='*80}\n")

for filename, expected_pixelated in EXPECTED.items():
    filepath = os.path.join(IMAGE_DIR, filename)
    label = "PIXELATED" if expected_pixelated else "CLEAN    "

    try:
        with open(filepath, "rb") as f:
            resp = requests.post(API_URL, files={"file": (filename, f)}, timeout=30)

        if resp.status_code != 200:
            print(f"  [{label}] {filename:<45} → ERROR {resp.status_code}: {resp.text[:80]}")
            results.append({"file": filename, "expected": expected_pixelated, "correct": None, "error": True})
            continue

        data = resp.json()
        detected = data.get("artifact_detected", False)
        confidence = data.get("confidence", 0.0)
        artifact_type = data.get("artifact_type", "none")
        severity = data.get("severity", "-")
        decision_maker = data.get("decision_maker", "-")
        corroboration = data.get("corroboration", {})
        corroborating = corroboration.get("corroborating", "-")

        correct = (detected == expected_pixelated)
        status = "✅ CORRECT" if correct else "❌ WRONG  "

        print(f"  {status} [{label}] {filename}")
        print(f"           detected={detected}, confidence={confidence:.3f}, type={artifact_type}, severity={severity}")
        print(f"           decision_maker={decision_maker}, corroborating={corroborating}")

        signals = data.get("signals", {})
        if signals:
            sig_str = "  ".join([f"{k}={v:.3f}" for k, v in signals.items()])
            print(f"           signals: {sig_str}")
        print()

        results.append({
            "file": filename,
            "expected": expected_pixelated,
            "detected": detected,
            "correct": correct,
            "confidence": confidence,
            "artifact_type": artifact_type,
            "severity": severity,
        })

    except Exception as e:
        print(f"  [{label}] {filename:<45} → EXCEPTION: {e}\n")
        results.append({"file": filename, "expected": expected_pixelated, "correct": None, "error": True})

# ─── Summary ─────────────────────────────────────────────────────────────────
valid = [r for r in results if r.get("correct") is not None]
correct_count = sum(1 for r in valid if r["correct"])
total = len(valid)

pixelated_results = [r for r in valid if r["expected"] == True]
clean_results     = [r for r in valid if r["expected"] == False]

tp = sum(1 for r in pixelated_results if r["correct"])   # correctly detected pixelated
tn = sum(1 for r in clean_results     if r["correct"])   # correctly detected clean
fp = sum(1 for r in clean_results     if not r["correct"])  # clean flagged as pixelated
fn = sum(1 for r in pixelated_results if not r["correct"])  # pixelated missed

print(f"{'='*80}")
print(f"  SUMMARY")
print(f"{'='*80}")
print(f"  Overall Accuracy : {correct_count}/{total} = {correct_count/total*100:.1f}%")
print(f"  True Positives   : {tp}/{len(pixelated_results)}  (pixelated correctly detected)")
print(f"  True Negatives   : {tn}/{len(clean_results)}  (clean correctly passed)")
print(f"  False Positives  : {fp}/{len(clean_results)}  (clean wrongly flagged)")
print(f"  False Negatives  : {fn}/{len(pixelated_results)}  (pixelated missed)")
if tp + fp > 0:
    precision = tp / (tp + fp)
    print(f"  Precision        : {precision:.2f}")
if tp + fn > 0:
    recall = tp / (tp + fn)
    print(f"  Recall           : {recall:.2f}")
print(f"{'='*80}\n")
