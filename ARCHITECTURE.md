# Macroblocking & Pixelation Detection System

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                     FastAPI REST API                     │
│                    (src/api/main.py)                     │
└─────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
        ▼                                       ▼
┌──────────────────┐                   ┌──────────────────┐
│ Image Detector   │                   │ Video Detector   │
│ (Single Frame)   │                   │ (Frame Stream)   │
└──────────────────┘                   └──────────────────┘
        │                                       │
        └───────────────────┬───────────────────┘
                            ↓
        ┌───────────────────┴───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   Tier 1:    │   │   Tier 2:    │   │   Tier 3:    │
│   Spatial    │   │   MVAD ML    │   │  Temporal    │
│  Detection   │   │    Model     │   │ Refinement   │
└──────────────┘   └──────────────┘   └──────────────┘
```

## Detection Pipeline

### Image Analysis

```
1. Convert to Y (luma) channel
   ↓
2. Tier 1: Spatial Detection
   - Boundary edge pairing (8/16/32/64px)
   - Grid periodicity (8/16/32/64px)
   - Color quantization (HSV)
   ↓
3. Tier 2: MVAD ML Model
   - Blockiness + pixelation scores
   ↓
4. Quality Metrics
   - BRISQUE: gradient-ratio at 8px boundaries
   - Block Variance: flat interior + strong boundary
   - DCT Score: AC/DC energy ratio (informational)
   ↓
5. Corroboration Check
   - MVAD alone is NOT trusted
   - At least one spatial signal must agree
   ↓
6. Coverage Guard
   - Passes if: corroboration present OR block_var ≥ 10% OR MVAD ≥ 0.60
   ↓
7. Hybrid Decision
   - MVAD > 0.35 AND corroboration AND coverage → ARTIFACT
   - MVAD > 0.35, no corroboration/coverage     → CLEAN (MVAD_unconfirmed)
   - BRISQUE > 55                               → ARTIFACT
   - Tier1 > 0.50                               → ARTIFACT
   - Weighted 60/40 fallback                    → borderline
```

### Video Analysis

Same as image analysis, plus:

```
Before detection:
  - Scene cut guard (skip transitions)

After detection:
  - Block persistence (3+ consecutive frames)
  - Sliding window (8 of 15 frames → alert)
  - Color shift detection

Output:
  artifact_detected = per-frame signal
  alert_fired       = sustained artifact (actionable)
```

## Core Components

### 1. Luma Channel (`_to_luma`)

All spatial detectors use the Y (luma) channel from YUV conversion. H.264/HEVC compression artifacts are primarily in luma, not chroma. More accurate than grayscale which mixes all channels equally.

### 2. Spatial Detector — Tier 1 (`SpatialDetector`)

**`boundary_edge_pairing()`**
Checks gradient strength on both sides of macroblock boundaries. Natural edge: one side strong. Macroblock: both sides strong and aligned. Covers 8/16px (H.264), 32/64px (H.265/HEVC, AV1).

**`grid_periodicity_check()`**
Detects variance dips at 8/16/32/64px intervals. Macroblocking creates flat blocks → variance drops at boundaries.

**`color_quantization_check()`**
High saturation + low hue variance (< 20) = color banding. Operates in HSV. Used in Tier1 composite score only, not as a standalone corroborator.

### 3. MVAD ML Model — Tier 2 (`MVADWrapper`)

RMViT-based model trained on 10 video artifact types. Input: 4 frames at 224×224. Output: blockiness + pixelation scores [0, 1].

**Limitation:** Trained on video content. Can produce false positives on text overlays, UI frames, and clean video with specific compression characteristics. Corroboration guard mitigates this.

### 4. BRISQUE (`compute_brisque_score`)

Compares gradient energy at 8px boundary rows/cols vs interior rows/cols (3px inside each block). Clean: ratio ≈ 1.0. Blocked: ratio > 2.0. Returns [0, 100].

### 5. Block Variance (`compute_block_variance_score`)

Flags 8×8 blocks with all three: flat interior (std < 3), strong boundary gradient (mean > 15), no interior gradients (mean < 5). Distinguishes macroblocked blocks from natural smooth regions (bokeh, sky, skin). Returns [0, 1].

### 6. DCT Score (`compute_dct_score`)

Compares AC energy at grid-aligned block rows vs interior rows. Informational only — not used in corroboration due to unreliability on compressed video. Returns [0, 1].

### 7. Corroboration Check

MVAD alone is not trusted. At least one of these must agree:

| Signal | Threshold | What it detects |
|---|---|---|
| edge | > 0.05 | Paired strong edges at block boundaries |
| brisque | > 30.0 | Quality degradation at 8px boundaries |
| grid | > 0.05 | Periodic variance pattern |
| tier1_spatial | > 0.035 | edge×0.70 + grid×0.30 combined |
| block_var | > 0.30 | Flat blocks with strong boundaries |

Color quantization and DCT are excluded from corroboration — both fire on natural content.

### 8. Coverage Guard

Prevents small UI elements or text overlays from triggering frame-level detection.

**Passes if any of:**
- `has_corroboration = True` — spatial signals already confirm
- `block_var_score ≥ 0.10` — pixel-level coverage
- `mvad_score ≥ 0.60` — very high ML confidence

**Image detector:** uses the above logic.
**Video detector:** `block_var ≥ 0.10 OR mvad ≥ 0.40` (slightly stricter, temporal refinement provides additional safety).

### 9. Temporal Refinement — Tier 3 (Video Only)

| Component | Purpose | Threshold |
|---|---|---|
| Scene Cut Guard | Skip scene transitions | SAD > 30; disabled for frame gaps > 5 |
| Block Persistence | Require 3+ consecutive frames | 3 frames |
| Sliding Window | Require sustained artifact | 8 of 15 frames |
| Color Shift | Detect sudden color changes | Saturation delta > 25 |

**`artifact_detected`** = per-frame signal (raw detection)
**`alert_fired`** = sustained artifact confirmed over time (actionable)

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | API info |
| GET | `/health` | Health check |
| GET | `/config` | Current thresholds |
| POST | `/analyze/image` | Upload image file |
| POST | `/analyze/image/url` | Analyze image from URL |
| POST | `/analyze/image/debug` | Full 10-step internal trace |
| POST | `/analyze/video` | Upload video file |
| POST | `/analyze/video/url` | Analyze video from URL |
| GET | `/results` | List saved results |
| GET | `/results/{id}` | Get specific result |
| DELETE | `/results/{id}` | Delete result |
| POST | `/stream/reset` | Reset video stream state |

## Configuration

All settings in `.env`. No hardcoded defaults.

```bash
# Detection thresholds
MVAD_VOTE_THRESHOLD=0.35
BRISQUE_VOTE_THRESHOLD=55.0
EDGE_VOTE_THRESHOLD=0.08
GRID_VOTE_THRESHOLD=0.08

# Temporal refinement (video)
SCENE_CUT_SAD_THRESHOLD=30.0
BLOCK_PERSISTENCE_FRAMES=3
VIDEO_WINDOW_SIZE=15
VIDEO_MIN_FLAGGED_FRAMES=8

# Model
MVAD_MODEL_PATH=MVAD_repo/logs/checkpoints/mvad.ckpt
DEVICE=cpu
```

Corroboration and coverage thresholds are in `image_detector.py` and `video_detector.py`.

## File Structure

```
src/
├── api/main.py              # FastAPI application
├── core/
│   ├── config.py            # Settings from .env
│   ├── detection.py         # Luma conversion, Tier 1, BRISQUE, BlockVar, DCT
│   ├── image_detector.py    # Image pipeline
│   ├── video_detector.py    # Video pipeline
│   ├── temporal.py          # Temporal refinement (video only)
│   └── debug_analyzer.py    # /analyze/image/debug endpoint
└── models/
    └── mvad_wrapper.py      # MVAD model loader and inference
```

## Signal Summary

| Signal | Domain | Threshold | Role |
|---|---|---|---|
| MVAD blockiness | ML | > 0.35 | Primary detector |
| MVAD pixelation | ML | > 0.35 | Primary detector |
| Boundary Edge | Spatial/luma | > 0.05 corr | Corroborator |
| Grid Periodicity | Spatial/luma | > 0.05 corr | Corroborator |
| Color Quantization | Color/HSV | — | Tier1 composite only |
| BRISQUE | Gradient/luma | > 30 corr / > 55 alone | Corroborator + standalone |
| Block Variance | Pixel/luma | > 0.30 corr | Corroborator + coverage |
| DCT Score | Frequency/luma | — | Informational only |

## Known Limitations

1. **MVAD false positives** on text overlays, UI frames, clean video with specific compression. Mitigated by corroboration + coverage guard.
2. **DCT unreliable** on compressed video — excluded from corroboration.
3. **Subtle artifacts** where all spatial signals are below thresholds may be missed. Intentional tradeoff to reduce false positives.
4. **Thresholds** are set by analysis and testing, not trained on a labeled dataset. Calibrating with labeled data would improve accuracy.

## References

- MVAD: [ChenFeng-Bristol/MVAD](https://github.com/ChenFeng-Bristol/MVAD)
- H.264/AVC: 8×8, 16×16 macroblock structure
- H.265/HEVC: 32×32, 64×64 CTU structure
- AV1: 64×64 superblock structure

---

**Version**: 4.0.0
**Last Updated**: 2026-05-20
