# Macroblocking & Pixelation Detection System

Production-ready REST API for detecting compression artifacts (macroblocking, pixelation, banding) in images and video streams using a hybrid ML + traditional CV approach.

---

## How It Works

The system combines four independent signals and requires corroboration before flagging an artifact:

1. **MVAD ML Model** — RMViT deep learning model trained on 10 video artifact types
2. **BRISQUE** — Gradient-ratio quality metric targeting 8px block boundaries
3. **Edge Detection** — Boundary edge pairing at 8x8/16x16 macroblock boundaries
4. **Grid Periodicity** — Detects repeating patterns at codec macroblock intervals

MVAD alone is not trusted — at least one spatial signal must corroborate before detection fires. This prevents false positives on clean video with specific compression characteristics.

See [ARCHITECTURE.md](ARCHITECTURE.md) for full system design and algorithm details.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/blockiness-detector.git
cd blockiness-detector
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up MVAD model

The MVAD checkpoint is not included in the repository (462 MB). You need to set it up manually.

**Option A: Clone the MVAD repo and download the checkpoint**

```bash
# Clone MVAD repo into project root
git clone https://github.com/ChenFeng-Bristol/MVAD.git MVAD_repo

# Download the pretrained checkpoint
# Place it at: MVAD_repo/logs/checkpoints/mvad.ckpt
```

The checkpoint can be downloaded from the [MVAD releases page](https://github.com/ChenFeng-Bristol/MVAD) or the authors' provided link.

**Option B: If you already have the checkpoint**

Place it at the following path relative to the project root:
```
MVAD_repo/logs/checkpoints/mvad.ckpt
```

**Verify the checkpoint is in place:**
```bash
ls -lh MVAD_repo/logs/checkpoints/mvad.ckpt
# Should show: ~462 MB
```

### 5. Configure environment

```bash
cp .env.example .env
```

Edit `.env` to set your configuration. Key settings:

```bash
# Path to MVAD checkpoint (relative to project root)
MVAD_MODEL_PATH=MVAD_repo/logs/checkpoints/mvad.ckpt

# Device: "cuda" for GPU, "cpu" for CPU
DEVICE=cpu

# Detection thresholds
MVAD_VOTE_THRESHOLD=0.35
BRISQUE_VOTE_THRESHOLD=55.0
```

### 6. Start the server

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

**Verify it's running:**
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "ml_model": "mvad",
  "tier1_enabled": true,
  "tier2_enabled": true,
  "tier3_enabled": true
}
```

If `ml_model` shows `"unavailable"`, the checkpoint was not found — check the path in `.env`.

---

## API Usage

### Analyze Image (file upload)
```bash
curl -X POST http://localhost:8000/analyze/image \
  -F "file=@frame.jpg"
```

### Analyze Image (URL)
```bash
curl -X POST http://localhost:8000/analyze/image/url \
  -F "image_url=https://example.com/frame.jpg"
```

### Analyze Video (file upload)
```bash
curl -X POST http://localhost:8000/analyze/video \
  -F "file=@video.mp4" \
  -F "sample_rate=10"
```

### Analyze Video (URL)
```bash
curl -X POST http://localhost:8000/analyze/video/url \
  -F "video_url=https://example.com/video.mp4" \
  -F "sample_rate=10"
```

### Debug an image (step-by-step trace)
```bash
curl -X POST http://localhost:8000/analyze/image/debug \
  -F "file=@frame.jpg"
```

Returns a 10-step internal breakdown showing exactly how each signal was computed and why the decision was made.

### Interactive API docs
```
http://localhost:8000/docs
```

---

## Example Response

```json
{
  "artifact_detected": true,
  "confidence": 0.633,
  "artifact_type": "macroblocking",
  "severity": "medium",
  "signals": {
    "boundary_edge": 0.066,
    "grid_periodicity": 0.013,
    "color_quantization": 0.390,
    "mvad_blockiness": 0.633,
    "mvad_pixelation": 0.017,
    "brisque": 14.9
  },
  "corroboration": {
    "has_corroboration": true,
    "corroborating": "edge"
  },
  "decision_maker": "MVAD"
}
```

---

## Configuration

All thresholds are in `.env`. No code changes needed to tune detection sensitivity.

| Setting | Default | Description |
|---|---|---|
| `MVAD_VOTE_THRESHOLD` | 0.35 | MVAD confidence required to flag |
| `BRISQUE_VOTE_THRESHOLD` | 55.0 | BRISQUE score required to flag |
| `EDGE_VOTE_THRESHOLD` | 0.08 | Edge score required to vote |
| `GRID_VOTE_THRESHOLD` | 0.08 | Grid score required to vote |
| `DEVICE` | cpu | `cuda` for GPU acceleration |
| `MVAD_MODEL_PATH` | MVAD_repo/logs/checkpoints/mvad.ckpt | Path to checkpoint |

**More sensitive** (detect more, more false positives):
```bash
MVAD_VOTE_THRESHOLD=0.30
BRISQUE_VOTE_THRESHOLD=45.0
```

**Less sensitive** (fewer false positives, may miss subtle artifacts):
```bash
MVAD_VOTE_THRESHOLD=0.45
BRISQUE_VOTE_THRESHOLD=60.0
```

---

## Requirements

- Python 3.10+
- 2GB RAM minimum (MVAD model)
- CUDA GPU optional (CPU works, slower)
- MVAD checkpoint: 462 MB (not included, see Setup)

---

## Project Structure

```
blockiness_detector/
├── src/
│   ├── api/main.py              # FastAPI application + all endpoints
│   ├── core/
│   │   ├── config.py            # Settings loaded from .env
│   │   ├── detection.py         # Tier 1 spatial detection + BRISQUE
│   │   ├── image_detector.py    # Image pipeline + corroboration logic
│   │   ├── video_detector.py    # Video pipeline + corroboration logic
│   │   ├── temporal.py          # Temporal refinement (video only)
│   │   └── debug_analyzer.py    # Step-by-step debug breakdown
│   └── models/
│       └── mvad_wrapper.py      # MVAD model loader and inference
├── MVAD_repo/                   # MVAD source code (git submodule)
│   └── logs/checkpoints/
│       └── mvad.ckpt            # ← Place checkpoint here (not in repo)
├── data/
│   ├── uploads/                 # Temporary upload storage
│   └── results/                 # JSON detection results
├── .env                         # Your configuration (not in repo)
├── .env.example                 # Configuration template
├── requirements.txt
├── ARCHITECTURE.md              # Full system design and algorithms
└── README.md
```

---

## License

Detection system code: MIT

MVAD model: See [ChenFeng-Bristol/MVAD](https://github.com/ChenFeng-Bristol/MVAD) for license terms.
