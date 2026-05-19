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
1. Input: Image (JPEG, PNG, etc.)
   ↓
2. Tier 1: Spatial Detection
   - Edge Detection (boundary pairing)
   - Grid Detection (periodicity)
   - Color Quantization
   ↓
3. Tier 2: MVAD ML Model
   - Blockiness score
   - Pixelation score
   ↓
4. BRISQUE Quality Metric
   - Gradient-ratio at 8px boundaries
   - Compares boundary rows/cols vs interior rows/cols
   ↓
5. Corroboration Check
   - MVAD alone is NOT trusted
   - At least one spatial signal must agree with MVAD
   ↓
6. Hybrid MVAD-Primary Decision
   - If MVAD > 35% AND corroboration → ARTIFACT
   - If MVAD > 35% AND no corroboration → CLEAN (MVAD_unconfirmed)
   - Else if BRISQUE > 55 → ARTIFACT
   - Else if Tier1 > 50% → ARTIFACT
   - Else → Weighted: 60% MVAD + 40% Tier1
   ↓
7. Output: Detection result
```

### Video Analysis

```
1. Input: Video file (MP4, TS, etc.)
   ↓
2. For each frame:
   ↓
   a. Tier 3: Scene Cut Guard
      - Skip scene transitions
   ↓
   b. Tier 1: Spatial Detection
      - Same as image analysis
   ↓
   c. Tier 2: MVAD ML Model
      - With temporal context (4 frames)
   ↓
   d. BRISQUE Quality Metric
   ↓
   e. Corroboration Check (same as image)
   ↓
   f. Hybrid MVAD-Primary Decision (same as image)
   ↓
   g. Tier 3: Temporal Refinement
      - Block persistence (3+ frames)
      - Sliding window (8 of 15 frames)
      - Color shift detection
   ↓
3. Output: Frame-by-frame results + summary
```

## Core Components

### 1. Spatial Detector (Tier 1)

**File**: `src/core/detection.py`

**Methods**:
- `boundary_edge_pairing()`: Detects macroblock boundaries by checking gradient strength on both sides. Natural edge: one side strong. Macroblock: both sides strong and aligned.
- `grid_periodicity_check()`: Detects 8x8/16x16 grid patterns using variance analysis
- `color_quantization_check()`: Detects color banding in HSV space (high saturation + low hue variance). Threshold tightened to hue_variance < 20 to reduce false positives on natural content.

**Output**: Edge score, grid score, color score (0-1 range)

### 2. MVAD ML Model (Tier 2)

**File**: `src/models/mvad_wrapper.py`

**Model**: RMViT-based deep learning model trained on 10 artifact types:
- motion_blur, dark_scenes, graininess, aliasing, banding
- **blockiness** (primary), spatial_blur, frame_drop, transmission_error, black_screen

**Input**: 4 frames (224x224, normalized)
**Output**: Blockiness score, pixelation score (0-1 range)

**Important**: MVAD was trained on video content. It can produce false positives on:
- Text on white background (title cards, lower thirds)
- UI overlays and graphics
- Clean video with specific compression characteristics

This is why corroboration is required.

### 3. BRISQUE Quality Metric

**File**: `src/core/detection.py` - `compute_brisque_score()`

**Algorithm**: Gradient-ratio approach (corrected implementation)
1. Compute gradient magnitude (Sobel)
2. Compare energy at boundary rows/cols vs interior rows/cols (3px inside)
   - Uses full boundary rows/cols, NOT a 2D mask (which dilutes the signal)
3. Compute blockiness ratio
4. Add sharpness penalty (Laplacian variance)
5. Weighted combination: 75% blockiness + 25% sharpness

**Output**: Quality score (0-100 range, higher = more artifacts)

**Note**: Previous implementation used a 2D OR mask which made boundary and interior pixel counts equal (~25% each), diluting the signal. Fixed to compare full boundary rows vs interior rows separately.

### 4. Corroboration Check

**File**: `src/core/image_detector.py`, `src/core/video_detector.py`

**Purpose**: Prevent MVAD false positives by requiring at least one spatial signal to agree.

**Corroborating signals (any one sufficient)**:
```
edge_score    > 0.05   — boundary pairing detected
brisque_score > 30.0   — quality degradation detected
grid_score    > 0.05   — periodicity detected
tier1_spatial > 0.035  — edge*0.70 + grid*0.30 combined
```

**Note**: Color quantization is intentionally excluded from corroboration. It fires on natural saturated content (sky, skin, UI colors) and is unreliable as a corroborator.

**Why this matters**:
- Clean video with MVAD=0.63, edge=0.031, grid=0.022 → tier1_spatial=0.028 < 0.035 → CLEAN ✅
- Macroblocked frame with MVAD=0.63, edge=0.066 → tier1_spatial=0.050 > 0.035 → ARTIFACT ✅

### 5. Hybrid MVAD-Primary Decision

**File**: `src/core/image_detector.py`, `src/core/video_detector.py`

**Decision tree**:
```
1. MVAD > 0.35 AND corroboration  → ARTIFACT (decision_maker: "MVAD")
2. MVAD > 0.35 AND no corroboration → CLEAN (decision_maker: "MVAD_unconfirmed")
3. BRISQUE > 55                   → ARTIFACT (decision_maker: "BRISQUE")
4. Tier1 > 0.50                   → ARTIFACT (decision_maker: "Tier1")
5. Weighted 60/40 > 0.30          → ARTIFACT (decision_maker: "Hybrid")
6. else                           → CLEAN
```

### 6. Temporal Refinement (Tier 3, Video Only)

**File**: `src/core/temporal.py`

**Methods**:
- `scene_cut_guard()`: Skip scene transitions (high SAD). Disabled for sampled analysis (frame gaps > 5).
- `block_persistence_check()`: Require 3+ consecutive frames. Adaptive for sampled analysis.
- `should_alert()`: Check sliding window (8 of 15 frames rule). Adaptive: 50% for partial windows.
- `color_shift_detection()`: Detect sudden saturation changes (delta > 25 units).
- `qp_bitstream_hint()`: Use QP values from encoder if available.

**Purpose**: Reduce false positives in video streams by requiring temporal consistency.

## API Endpoints

### POST /analyze/image
Analyze single image for artifacts (file upload).

**Input**: `file` — Image file (multipart/form-data)

**Output**:
```json
{
  "artifact_detected": true,
  "confidence": 0.633,
  "artifact_type": "macroblocking",
  "severity": "medium",
  "signals": {
    "boundary_edge": 0.066,
    "grid_periodicity": 0.013,
    "mvad_blockiness": 0.633,
    "mvad_pixelation": 0.017,
    "brisque": 14.9
  },
  "voting": { "artifact_votes": 1, "clean_votes": 3, "details": {...} },
  "corroboration": {
    "has_corroboration": true,
    "corroborating": "edge",
    "signals_checked": {...}
  },
  "decision_maker": "MVAD"
}
```

### POST /analyze/image/url
Analyze single image from URL.

**Input**: `image_url` — Image URL (form data)

**Output**: Same as `/analyze/image`

### POST /analyze/image/debug
Full internal breakdown of detection workflow (10 steps).

**Input**: `file` — Image file

**Output**: Step-by-step trace including:
- Image properties
- Gradient computation
- Per-boundary edge scores
- Per-grid-size periodicity scores
- Per-block color quantization
- BRISQUE breakdown (boundary vs interior energy)
- MVAD scores
- Tier1 composite
- Final decision tree
- Per-block artifact map (8x8 grid)

### POST /analyze/video
Analyze video file for artifacts (file upload).

**Input**:
- `file` — Video file (multipart/form-data)
- `sample_rate` — Analyze every Nth frame (default: 1)
- `max_frames` — Maximum frames to analyze (optional)

**Output**:
```json
{
  "video_properties": { "width": 1920, "height": 1080, "fps": 24, "total_frames": 1864 },
  "summary": {
    "frames_flagged": 45,
    "alerts_fired": 38,
    "flagged_percentage": 5.0
  },
  "average_signals": { "boundary_edge": 0.066, "mvad_blockiness": 0.633, ... },
  "frame_results": [...]
}
```

### POST /analyze/video/url
Analyze video from URL.

**Input**:
- `video_url` — Video URL (form data)
- `sample_rate`, `max_frames` — same as above

**Output**: Same as `/analyze/video`

### GET /health
```json
{ "status": "healthy", "ml_model": "mvad", "tier1_enabled": true, "tier2_enabled": true, "tier3_enabled": true }
```

### GET /results
List saved analysis results. Filter by `result_type` (image/video), `limit`.

### GET /results/{result_id}
Get specific result by ID.

### DELETE /results/{result_id}
Delete specific result.

### POST /stream/reset
Reset temporal state for a video stream.

### GET /config
Get current detection configuration.

## Configuration

**File**: `src/core/config.py` + `.env`

All configuration is loaded from `.env`. No hardcoded defaults.

**Key Settings**:
```bash
# Voting thresholds
EDGE_VOTE_THRESHOLD=0.08
GRID_VOTE_THRESHOLD=0.08
MVAD_VOTE_THRESHOLD=0.35
BRISQUE_VOTE_THRESHOLD=55.0

# Corroboration thresholds (hardcoded in detector logic)
# edge > 0.05, brisque > 30, grid > 0.05, tier1_spatial > 0.035

# Temporal refinement (video)
SCENE_CUT_SAD_THRESHOLD=30.0
BLOCK_PERSISTENCE_FRAMES=3
VIDEO_WINDOW_SIZE=15
VIDEO_MIN_FLAGGED_FRAMES=8

# Model settings
MVAD_MODEL_PATH=MVAD_repo/logs/checkpoints/mvad.ckpt
DEVICE=cpu
```

## File Structure

```
blockiness_detector/
├── src/
│   ├── api/
│   │   └── main.py              # FastAPI application
│   ├── core/
│   │   ├── config.py            # Configuration (loads from .env)
│   │   ├── detection.py         # Tier 1 spatial + BRISQUE
│   │   ├── image_detector.py    # Image pipeline + corroboration
│   │   ├── video_detector.py    # Video pipeline + corroboration
│   │   ├── temporal.py          # Temporal refinement (video only)
│   │   └── debug_analyzer.py    # Debug endpoint logic
│   └── models/
│       └── mvad_wrapper.py      # MVAD model wrapper
├── data/
│   ├── uploads/                 # Temporary uploads
│   └── results/                 # JSON results
├── MVAD_repo/                   # Pre-trained model
│   └── logs/checkpoints/
│       └── mvad.ckpt            # 462.6 MB checkpoint
├── requirements.txt
├── .env                         # All configuration
└── ARCHITECTURE.md              # This file
```

## Key Algorithm Details

### Boundary Edge Pairing
Checks BOTH sides of macroblock boundaries:
- Natural edge: one side strong
- Macroblock: both sides strong and aligned
- Checks 8x8 and 16x16 boundaries (H.264/HEVC)

### Grid Periodicity
Detects repeating patterns at 8, 16, 32 pixel intervals:
- Computes variance across rows/columns
- Checks if variance dips align with grid boundaries
- Score = 1 - (boundary_variance / midpoint_variance)

### BRISQUE Gradient-Ratio (Corrected)
Compares gradient energy at 8px boundary rows/cols vs interior:
- **Boundary**: rows at multiples of 8, cols at multiples of 8
- **Interior**: rows 3px inside each block, cols 3px inside each block
- Clean image: ratio ≈ 1.0
- Blocked image: ratio > 2.0
- Score = 0.75 × blockiness + 0.25 × sharpness_penalty

### Corroboration Logic
Prevents MVAD false positives on non-video content:
```python
tier1_spatial = edge_score * 0.70 + grid_score * 0.30
corroborating = (
    edge_score    > 0.05 or
    brisque_score > 30.0 or
    grid_score    > 0.05 or
    tier1_spatial > 0.035
)
# Color quantization excluded — fires on natural saturated content
```

### Temporal Refinement
Reduces false positives in video streams:
- Scene cut guard: skip transitions
- Block persistence: require 3+ frames
- Sliding window: 8 of 15 frames rule

## Known Limitations

1. **MVAD false positives**: MVAD can score high on clean video with specific compression characteristics. Mitigated by corroboration check.

2. **BRISQUE insensitivity**: BRISQUE gradient-ratio is effective for obvious blocking but may miss subtle artifacts where boundary energy is only slightly elevated.

3. **Color quantization noise**: Color quantization check fires on natural saturated content. Used only in Tier1 composite score, not as a standalone corroborator.

4. **Subtle artifacts**: Very subtle macroblocking (edge < 0.035, MVAD 0.35-0.50) may be missed by the corroboration requirement. This is an intentional tradeoff to reduce false positives.

## Performance

**Speed**:
- Image: ~200ms per image (with MVAD)
- Video: ~30 FPS processing speed
- Tier 1 only: ~50ms per frame

**Memory**:
- ~2GB with MVAD loaded
- GPU optional (CUDA support)

## Tuning

### More Sensitive (detect more artifacts, more false positives)
```bash
MVAD_VOTE_THRESHOLD=0.30
BRISQUE_VOTE_THRESHOLD=45.0
# Lower corroboration thresholds in image_detector.py and video_detector.py
```

### Less Sensitive (fewer false positives, may miss subtle artifacts)
```bash
MVAD_VOTE_THRESHOLD=0.45
BRISQUE_VOTE_THRESHOLD=60.0
# Raise corroboration thresholds in image_detector.py and video_detector.py
```

## References

- MVAD Model: [ChenFeng-Bristol/MVAD](https://github.com/ChenFeng-Bristol/MVAD)
- RMViT Architecture: Vision Transformer for video quality assessment
- BRISQUE: Blind/Referenceless Image Spatial Quality Evaluator
- ITU-T P.910: Subjective video quality assessment methods

---

**Version**: 2.0.0
**Last Updated**: 2026-05-19
