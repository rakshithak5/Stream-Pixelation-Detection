
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from typing import Optional, Generator, Tuple
import io
from PIL import Image
import requests
from pathlib import Path
import json
from datetime import datetime
import uuid
import tempfile
import aiofiles

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import settings
from src.core.image_detector import image_detector
from src.core.video_detector import video_detector
from src.core.debug_analyzer import analyze_image_debug

# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="Production-grade macroblocking and pixelation detection for live streams"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create results directory
RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Fix #1: PyAV-based frame reader ──────────────────────────────────────────
def iter_video_frames_pyav(
    video_path: str,
    sample_rate: int = 1,
    max_frames: Optional[int] = None,
) -> Generator[Tuple[int, float, np.ndarray], None, None]:
    """
    Reliable frame iterator using PyAV (libav/ffmpeg backend).

    Fixes the OpenCV frame-dropping bug on .ts transport streams.
    OpenCV's VideoCapture silently drops frames in .ts files (decoded 50/79
    in testing). PyAV decodes every frame correctly.

    Yields: (frame_number, timestamp_seconds, bgr_frame_ndarray)
    """
    try:
        import av
    except ImportError:
        raise RuntimeError(
            "PyAV not installed. Run: pip install av"
        )

    container = av.open(video_path)
    video_stream = next(
        (s for s in container.streams if s.type == 'video'), None
    )
    if video_stream is None:
        container.close()
        raise ValueError("No video stream found in file")

    fps = float(video_stream.average_rate or video_stream.base_rate or 25)
    frame_number = 0
    analyzed_count = 0

    try:
        for packet in container.demux(video_stream):
            for av_frame in packet.decode():
                if max_frames and analyzed_count >= max_frames:
                    return

                if frame_number % sample_rate == 0:
                    # Convert to BGR numpy array (OpenCV format)
                    bgr_frame = av_frame.to_ndarray(format='bgr24')
                    ts = float(av_frame.pts * video_stream.time_base) if av_frame.pts else frame_number / fps
                    yield frame_number, ts, bgr_frame
                    analyzed_count += 1

                frame_number += 1
    finally:
        container.close()


def get_video_properties_pyav(video_path: str) -> dict:
    """Get video metadata using PyAV."""
    try:
        import av
        container = av.open(video_path)
        video_stream = next(
            (s for s in container.streams if s.type == 'video'), None
        )
        if video_stream is None:
            container.close()
            return {'width': 0, 'height': 0, 'fps': 25.0, 'total_frames': 0, 'duration': 0.0}

        fps = float(video_stream.average_rate or video_stream.base_rate or 25)
        total_frames = video_stream.frames or 0
        duration = float(video_stream.duration * video_stream.time_base) if video_stream.duration else 0.0
        width  = video_stream.codec_context.width
        height = video_stream.codec_context.height
        container.close()
        return {
            'width': width, 'height': height,
            'fps': fps, 'total_frames': total_frames, 'duration': duration,
        }
    except Exception:
        # Fallback to OpenCV for non-.ts files
        cap = cv2.VideoCapture(video_path)
        props = {
            'width':        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height':       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps':          cap.get(cv2.CAP_PROP_FPS) or 25.0,
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration':     0.0,
        }
        props['duration'] = props['total_frames'] / props['fps'] if props['fps'] > 0 else 0.0
        cap.release()
        return props


def normalize_json_value(value):
    """Recursively convert NumPy scalar values to native Python types."""
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, dict):
        return {k: normalize_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_json_value(v) for v in value]
    return value


def save_result_to_json(result: dict, result_type: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_id = str(uuid.uuid4())[:8]
    filename = f"{result_type}_{timestamp}_{result_id}.json"
    filepath = RESULTS_DIR / filename
    
    # Add metadata
    result['metadata'] = {
        'result_id': result_id,
        'timestamp': datetime.now().isoformat(),
        'result_type': result_type
    }
    
    # Normalize any NumPy scalars before saving
    normalized_result = normalize_json_value(result)
    with open(filepath, 'w') as f:
        json.dump(normalized_result, f, indent=2)
    
    return str(filepath)


def download_from_url(url: str) -> bytes:
    """Download file from URL"""
    try:
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        return response.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download from URL: {str(e)}")


def compute_average_signals(results: list) -> dict:
    """
    Compute average signals across all analyzed frames.
    Skips scene cut frames (different signal structure) and boolean fields.
    """
    # Fixed signal keys from normal frames (not scene cuts)
    signal_keys = ['boundary_edge', 'grid_periodicity', 'color_quantization',
                   'mvad_blockiness', 'mvad_pixelation', 'brisque']

    avg_signals = {}
    for key in signal_keys:
        values = []
        for r in results:
            val = r.get('signals', {}).get(key)
            if val is not None and isinstance(val, (int, float)):
                values.append(float(val))
        avg_signals[key] = float(np.mean(values)) if values else 0.0

    return avg_signals
    try:
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        return response.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download from URL: {str(e)}")


@app.get("/")
async def root():
    return {
        "service": settings.API_TITLE,
        "version": settings.API_VERSION,
        "status": "operational",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "image_analysis_upload": "/analyze/image",
            "image_analysis_url": "/analyze/image/url",
            "video_analysis_upload": "/analyze/video",
            "video_analysis_url": "/analyze/video/url",
            "list_results": "/results",
            "get_result": "/results/{result_id}",
            "reset_stream": "/stream/reset"
        }
    }


@app.get("/health")
async def health():
    from src.models.mvad_wrapper import model_manager
    
    ml_status = "unavailable"
    if model_manager.mvad and model_manager.mvad.model:
        ml_status = "mvad"
    elif model_manager.inception:
        ml_status = "inception_v3"
    
    return {
        "status": "healthy",
        "ml_model": ml_status,
        "tier1_enabled": True,
        "tier2_enabled": ml_status != "unavailable",
        "tier3_enabled": True
    }


@app.post("/analyze/image")
async def analyze_image(
    file: UploadFile = File(...)
):
    """
    Analyze single image for macroblocking and pixelation artifacts.
    Upload image file directly.
    Results are automatically saved to JSON.
    
    Args:
        file: Image file upload (JPEG, PNG, etc.)
    
    Returns:
        Detection result with confidence, type, and signal breakdown
    """
    try:
        # Read image
        contents = await file.read()
        source = file.filename
        
        # Read image
        image = Image.open(io.BytesIO(contents))
        image_np = np.array(image)
        
        # Ensure image is BGR (3 channels) for OpenCV
        if len(image_np.shape) == 2:
            # Grayscale -> BGR
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        elif len(image_np.shape) == 3:
            if image_np.shape[2] == 4:
                # RGBA -> BGR (remove alpha channel)
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
            elif image_np.shape[2] == 3:
                # RGB -> BGR
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported image format: {image_np.shape[2]} channels")
        else:
            raise HTTPException(status_code=400, detail=f"Invalid image shape: {image_np.shape}")
        
        # Run detection
        result = image_detector.analyze(image_np)
        
        # Add metadata
        result['source'] = source
        result['source_type'] = 'upload'
        result['image_size'] = {
            'width': image_np.shape[1],
            'height': image_np.shape[0]
        }
        
        # Save result to JSON
        json_path = save_result_to_json(result, 'image')
        result['saved_to'] = json_path
        result = normalize_json_value(result)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze/image/url")
async def analyze_image_url(
    image_url: str = Form(...)
):
    """
    Analyze single image from URL for macroblocking and pixelation artifacts.
    Provide image URL.
    Results are automatically saved to JSON.
    
    Args:
        image_url: URL to image file
    
    Returns:
        Detection result with confidence, type, and signal breakdown
    """
    try:
        # Download image
        contents = download_from_url(image_url)
        source = image_url
        
        # Read image
        image = Image.open(io.BytesIO(contents))
        image_np = np.array(image)
        
        # Ensure image is BGR (3 channels) for OpenCV
        if len(image_np.shape) == 2:
            # Grayscale -> BGR
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        elif len(image_np.shape) == 3:
            if image_np.shape[2] == 4:
                # RGBA -> BGR (remove alpha channel)
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
            elif image_np.shape[2] == 3:
                # RGB -> BGR
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported image format: {image_np.shape[2]} channels")
        else:
            raise HTTPException(status_code=400, detail=f"Invalid image shape: {image_np.shape}")
        
        # Run detection
        result = image_detector.analyze(image_np)
        
        # Add metadata
        result['source'] = source
        result['source_type'] = 'url'
        result['image_size'] = {
            'width': image_np.shape[1],
            'height': image_np.shape[0]
        }
        
        # Save result to JSON
        json_path = save_result_to_json(result, 'image')
        result['saved_to'] = json_path
        result = normalize_json_value(result)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/image/debug")
async def analyze_image_debug_endpoint(
    file: UploadFile = File(...)
):
    """
    Full internal debug breakdown of image detection.

    Returns step-by-step workflow trace:
    - Step 1: Image properties (size, brightness, contrast)
    - Step 2: Gradient computation (Sobel)
    - Step 3: Boundary edge pairing (per 8/16px boundary)
    - Step 4: Grid periodicity (per 8/16/32px grid)
    - Step 5: Color quantization (per 16x16 block)
    - Step 6: BRISQUE gradient-ratio (boundary vs interior energy)
    - Step 7: MVAD ML model (blockiness + pixelation scores)
    - Step 8: Tier 1 composite score
    - Step 9: Final hybrid decision tree
    - Step 10: Per-block artifact map (8x8 grid)
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_np = np.array(image)

        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        elif len(image_np.shape) == 3:
            if image_np.shape[2] == 4:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
            elif image_np.shape[2] == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        result = analyze_image_debug(image_np)
        result = normalize_json_value(result)
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug analysis failed: {str(e)}")


@app.post("/analyze/video")
async def analyze_video(
    file: UploadFile = File(...),
    max_frames: Optional[int] = Form(None),
    sample_rate: int = Form(1)
):
    """
    Analyze video file for macroblocking and pixelation artifacts.

    Uses PyAV (libav/ffmpeg) for frame decoding — correctly handles .ts
    transport streams without the frame-dropping bug in cv2.VideoCapture.

    Args:
        file:        Video file upload (.ts, .mp4, .mov, etc.)
        max_frames:  Maximum frames to analyze (None = all)
        sample_rate: Analyze every Nth frame (default: 1 = every frame)
    """
    temp_file = None
    try:
        suffix = Path(file.filename).suffix if file.filename else '.ts'
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        contents = await file.read()
        temp_file.write(contents)
        temp_file.close()
        video_path = temp_file.name
        source = file.filename

        props = get_video_properties_pyav(video_path)
        stream_id = f"api_{uuid.uuid4().hex[:8]}"
        video_detector.reset_stream(stream_id)

        results = []
        flagged_count = 0
        alert_count   = 0

        for frame_number, timestamp, frame in iter_video_frames_pyav(
            video_path, sample_rate=sample_rate, max_frames=max_frames
        ):
            frame_result = video_detector.analyze_frame(
                frame=frame,
                stream_id=stream_id,
                frame_number=frame_number,
                qp_value=None,
            )
            frame_result['frame_number'] = frame_number
            frame_result['timestamp']    = round(timestamp, 4)
            results.append(frame_result)

            if frame_result['artifact_detected']:
                flagged_count += 1
            if frame_result['alert_fired']:
                alert_count += 1

        video_detector.reset_stream(stream_id)
        analyzed_count = len(results)
        avg_signals    = compute_average_signals(results)

        response = {
            'source':      source,
            'source_type': 'upload',
            'video_properties': {
                **props,
                'frames_decoded': analyzed_count,
            },
            'analysis_settings': {
                'sample_rate': sample_rate,
                'max_frames':  max_frames,
                'frames_analyzed': analyzed_count,
                'decoder': 'pyav',
            },
            'summary': {
                'frames_flagged':    flagged_count,
                'alerts_fired':      alert_count,
                'flagged_percentage': flagged_count / analyzed_count * 100 if analyzed_count > 0 else 0,
                'alert_percentage':   alert_count  / analyzed_count * 100 if analyzed_count > 0 else 0,
            },
            'average_signals': avg_signals,
            'frame_results':   results,
        }

        json_path = save_result_to_json(response, 'video')
        response['saved_to'] = json_path
        return JSONResponse(content=normalize_json_value(response))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {str(e)}")
    finally:
        if temp_file and Path(temp_file.name).exists():
            Path(temp_file.name).unlink(missing_ok=True)


@app.post("/analyze/video/url")
async def analyze_video_url(
    video_url: str = Form(...),
    max_frames: Optional[int] = Form(None),
    sample_rate: int = Form(1)
):
    """
    Analyze video from URL for macroblocking and pixelation artifacts.
    Uses PyAV for reliable .ts frame decoding.
    """
    temp_file = None
    try:
        contents = download_from_url(video_url)
        suffix = Path(video_url).suffix or '.ts'
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(contents)
        temp_file.close()
        video_path = temp_file.name

        props = get_video_properties_pyav(video_path)
        stream_id = f"api_{uuid.uuid4().hex[:8]}"
        video_detector.reset_stream(stream_id)

        results = []
        flagged_count = 0
        alert_count   = 0

        for frame_number, timestamp, frame in iter_video_frames_pyav(
            video_path, sample_rate=sample_rate, max_frames=max_frames
        ):
            frame_result = video_detector.analyze_frame(
                frame=frame,
                stream_id=stream_id,
                frame_number=frame_number,
                qp_value=None,
            )
            frame_result['frame_number'] = frame_number
            frame_result['timestamp']    = round(timestamp, 4)
            results.append(frame_result)

            if frame_result['artifact_detected']:
                flagged_count += 1
            if frame_result['alert_fired']:
                alert_count += 1

        video_detector.reset_stream(stream_id)
        analyzed_count = len(results)
        avg_signals    = compute_average_signals(results)

        response = {
            'source':      video_url,
            'source_type': 'url',
            'video_properties': {**props, 'frames_decoded': analyzed_count},
            'analysis_settings': {
                'sample_rate': sample_rate,
                'max_frames':  max_frames,
                'frames_analyzed': analyzed_count,
                'decoder': 'pyav',
            },
            'summary': {
                'frames_flagged':    flagged_count,
                'alerts_fired':      alert_count,
                'flagged_percentage': flagged_count / analyzed_count * 100 if analyzed_count > 0 else 0,
                'alert_percentage':   alert_count  / analyzed_count * 100 if analyzed_count > 0 else 0,
            },
            'average_signals': avg_signals,
            'frame_results':   results,
        }

        json_path = save_result_to_json(response, 'video')
        response['saved_to'] = json_path
        return JSONResponse(content=normalize_json_value(response))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {str(e)}")
    finally:
        if temp_file and Path(temp_file.name).exists():
            Path(temp_file.name).unlink(missing_ok=True)


@app.get("/results")
async def list_results(result_type: Optional[str] = None, limit: int = 50):
    """
    List all saved analysis results.
    
    Args:
        result_type: Filter by 'image' or 'video' (None = all)
        limit: Maximum number of results to return
    
    Returns:
        List of saved results with metadata
    """
    try:
        results = []
        
        # Get all JSON files
        json_files = sorted(RESULTS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        
        for json_file in json_files[:limit]:
            # Filter by type if specified
            if result_type and not json_file.name.startswith(result_type):
                continue
            
            try:
                with open(json_file) as f:
                    data = json.load(f)
                
                # Extract summary info
                summary = {
                    'filename': json_file.name,
                    'filepath': str(json_file),
                    'result_id': data.get('metadata', {}).get('result_id'),
                    'timestamp': data.get('metadata', {}).get('timestamp'),
                    'result_type': data.get('metadata', {}).get('result_type'),
                    'source': data.get('source'),
                    'artifact_detected': data.get('artifact_detected') or data.get('summary', {}).get('alerts_fired', 0) > 0
                }
                
                if data.get('metadata', {}).get('result_type') == 'video':
                    summary['frames_analyzed'] = data.get('analysis_settings', {}).get('frames_analyzed')
                    summary['alerts_fired'] = data.get('summary', {}).get('alerts_fired')
                
                results.append(summary)
            
            except Exception as e:
                print(f"Error reading {json_file}: {e}")
                continue
        
        return JSONResponse(content={
            'total_results': len(results),
            'results': results
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list results: {str(e)}")


@app.get("/results/{result_id}")
async def get_result(result_id: str):
    """
    Get specific analysis result by ID.
    
    Args:
        result_id: Result ID from saved JSON
    
    Returns:
        Complete analysis result
    """
    try:
        # Find file with this result_id
        for json_file in RESULTS_DIR.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                
                if data.get('metadata', {}).get('result_id') == result_id:
                    return JSONResponse(content=data)
            
            except Exception:
                continue
        
        raise HTTPException(status_code=404, detail=f"Result with ID '{result_id}' not found")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get result: {str(e)}")


@app.delete("/results/{result_id}")
async def delete_result(result_id: str):
    """
    Delete specific analysis result by ID.
    
    Args:
        result_id: Result ID from saved JSON
    
    Returns:
        Success message
    """
    try:
        # Find and delete file with this result_id
        for json_file in RESULTS_DIR.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                
                if data.get('metadata', {}).get('result_id') == result_id:
                    json_file.unlink()
                    return JSONResponse(content={
                        'status': 'success',
                        'message': f"Result '{result_id}' deleted",
                        'deleted_file': str(json_file)
                    })
            
            except Exception:
                continue
        
        raise HTTPException(status_code=404, detail=f"Result with ID '{result_id}' not found")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete result: {str(e)}")


@app.post("/stream/reset")
async def reset_stream(stream_id: str = Form(...)):
    """
    Reset temporal state for a stream.
    Call this when stream restarts or ends.
    
    Args:
        stream_id: Stream identifier to reset
    """
    try:
        video_detector.reset_stream(stream_id)
        return {"status": "success", "stream_id": stream_id, "message": "Stream state reset"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.get("/config")
async def get_config():
    """Get current detection configuration"""
    return {
        "thresholds": {
            "tier1_edge": settings.EDGE_THRESHOLD,
            "tier1_color_quant": settings.COLOR_QUANT_THRESHOLD,
            "image_alert": settings.IMAGE_ALERT_THRESHOLD,
            "video_window_size": settings.VIDEO_WINDOW_SIZE,
            "video_min_flagged": settings.VIDEO_MIN_FLAGGED_FRAMES
        },
        "weights": {
            "boundary_edge": settings.BOUNDARY_EDGE_WEIGHT,
            "mvad_blockiness": settings.MVAD_BLOCKINESS_WEIGHT,
            "color_quant": settings.COLOR_QUANT_WEIGHT,
            "mvad_pixelation": settings.MVAD_PIXELATION_WEIGHT,
            "brisque": settings.BRISQUE_WEIGHT
        },
        "temporal": {
            "scene_cut_threshold": settings.SCENE_CUT_SAD_THRESHOLD,
            "persistence_frames": settings.BLOCK_PERSISTENCE_FRAMES,
            "qp_threshold": settings.QP_THRESHOLD,
            "color_shift_delta": settings.COLOR_SHIFT_SATURATION_DELTA
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )
