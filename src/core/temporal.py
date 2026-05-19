"""
Tier 3 - Temporal refinement (video only)
False positive guards and temporal state management
"""
import cv2
import numpy as np
from collections import deque
from typing import Dict, Optional, List
from src.core.config import settings


class TemporalRefinement:
    """
    Video-only temporal checks to reduce false positives.
    Maintains sliding window state per stream.
    """
    
    def __init__(self, stream_id: str):
        self.stream_id = stream_id
        self.window_size = settings.VIDEO_WINDOW_SIZE
        self.min_flagged = settings.VIDEO_MIN_FLAGGED_FRAMES
        
        # Sliding window of frame scores
        self.score_window = deque(maxlen=self.window_size)
        
        # Block persistence tracking
        self.consecutive_flagged = 0
        self.frames_since_last_check = 0  # Track frame gaps for sampled analysis
        
        # Previous frame for scene cut detection
        self.prev_frame = None
        self.prev_frame_gray = None
        self.prev_frame_number = None
    
    def scene_cut_guard(self, frame: np.ndarray, signals: Dict[str, float], frame_number: Optional[int] = None) -> bool:
        """
        Scene cut guard: high inter-frame SAD + clean blocks = legitimate transition.
        
        IMPORTANT: Only applies to consecutive frames. For sampled analysis (frame gaps > 5),
        scene cut detection is disabled since large gaps naturally have high SAD.
        
        Args:
            frame: Current frame
            signals: Tier 1 signals
            frame_number: Optional frame number to detect sampling gaps
        
        Returns: True if this is a scene cut (skip flagging), False otherwise
        """
        if self.prev_frame is None:
            self.prev_frame = frame.copy()
            self.prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.prev_frame_number = frame_number
            return False
        
        # Detect frame gap (sampled analysis)
        if frame_number is not None and self.prev_frame_number is not None:
            frame_gap = frame_number - self.prev_frame_number
            if frame_gap > 5:
                # Sampled analysis - disable scene cut detection
                # Large gaps naturally have high SAD, not indicative of scene cuts
                self.prev_frame = frame.copy()
                self.prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.prev_frame_number = frame_number
                return False
        
        # Compute Sum of Absolute Differences (SAD)
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sad = np.mean(np.abs(current_gray.astype(np.float32) - self.prev_frame_gray.astype(np.float32)))
        
        # Update previous frame
        self.prev_frame = frame.copy()
        self.prev_frame_gray = current_gray
        self.prev_frame_number = frame_number
        
        # High SAD + low artifact scores = scene cut
        if sad > settings.SCENE_CUT_SAD_THRESHOLD:
            edge_score = signals.get('edge_score', 0.0)
            if edge_score < settings.EDGE_THRESHOLD:
                return True  # Scene cut detected - skip flagging
        
        return False
    
    def block_persistence_check(self, is_flagged: bool, frame_gap: int = 1) -> bool:
        """
        Artifact must appear in at least 3 consecutive frames (or sampled frames).
        Single noisy frame never triggers alert.
        
        For sampled analysis (frame_gap > 1), we track flagged frames within a time window
        rather than requiring strict consecutive frames.
        
        NOTE: This is called BEFORE update_window(), so current frame is not in window yet.
        
        Args:
            is_flagged: Whether current frame is flagged
            frame_gap: Number of frames since last check (1 = consecutive, >1 = sampled)
        
        Returns: True if persistence threshold met, False otherwise
        """
        # Detect if we're in sampled mode by checking if ANY previous frame had a large gap
        in_sampled_mode = frame_gap > 5
        if not in_sampled_mode and len(self.score_window) > 0:
            # Check if we detected sampling in a previous frame
            # (This handles the case where frame 0 has gap=1 but frame 30 has gap=30)
            in_sampled_mode = getattr(self, '_sampled_mode_detected', False)
        
        if frame_gap > 5:
            self._sampled_mode_detected = True
            in_sampled_mode = True
        
        if in_sampled_mode:
            # Sampled analysis - use window-based persistence instead of consecutive
            window_len = len(self.score_window)
            
            if window_len == 0:
                # First frame - allow alert if flagged (no history to check)
                return is_flagged
            elif window_len == 1:
                # Second frame - require both frames flagged
                prev_flagged = self.score_window[0]['flagged']
                return is_flagged and prev_flagged
            else:
                # 3+ frames - count flagged in last 2 frames + current
                last_two = list(self.score_window)[-2:]
                prev_flagged_count = sum(1 for f in last_two if f['flagged'])
                
                # Total flagged = previous 2 + current
                total_flagged = prev_flagged_count + (1 if is_flagged else 0)
                
                # Need at least 2 of 3 frames flagged
                return total_flagged >= 2
        else:
            # Consecutive frame analysis - original logic
            if is_flagged:
                self.consecutive_flagged += 1
            else:
                self.consecutive_flagged = 0
            
            return self.consecutive_flagged >= settings.BLOCK_PERSISTENCE_FRAMES
    
    def qp_bitstream_hint(self, qp_value: Optional[int]) -> float:
        """
        QP (Quantization Parameter) hint from encoder bitstream.
        QP > 40 = heavy compression, strong confirmation of real artifact.
        
        Returns: confidence boost [0, 0.2] to add to composite score
        """
        if qp_value is None:
            return 0.0
        
        if qp_value > settings.QP_THRESHOLD:
            # High QP = high compression = likely real artifact
            boost = min((qp_value - settings.QP_THRESHOLD) / 20.0, 0.2)
            return boost
        
        return 0.0
    
    def color_shift_detection(self, frame: np.ndarray) -> bool:
        """
        Detect sudden color shifts between consecutive frames.
        Large saturation delta (> 25 units) = color shift artifact.
        
        Returns: True if color shift detected
        """
        if self.prev_frame is None:
            return False
        
        # Convert both frames to HSV
        current_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        prev_hsv = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2HSV)
        
        # Compute saturation delta
        current_sat = np.mean(current_hsv[:, :, 1])
        prev_sat = np.mean(prev_hsv[:, :, 1])
        
        sat_delta = abs(current_sat - prev_sat)
        
        return sat_delta > settings.COLOR_SHIFT_SATURATION_DELTA
    
    def update_window(self, composite_score: float, is_flagged: bool):
        """Update sliding window with new frame score"""
        self.score_window.append({
            'score': composite_score,
            'flagged': is_flagged
        })
    
    def should_alert(self) -> bool:
        """
        Check if alert should fire based on sliding window.
        
        Adaptive logic:
        - If window not full (<15 frames): require 50% of frames flagged
        - If window full (>=15 frames): require 8 of 15 frames (53%)
        
        This allows alerts to fire even with sampled analysis.
        """
        window_len = len(self.score_window)
        
        if window_len == 0:
            return False
        
        flagged_count = sum(1 for frame in self.score_window if frame['flagged'])
        
        if window_len < self.window_size:
            # Adaptive: use percentage threshold for partial window
            flagged_percentage = flagged_count / window_len
            return flagged_percentage >= 0.50  # 50% of available frames
        else:
            # Original: use count threshold for full window
            return flagged_count >= self.min_flagged  # 8 of 15 frames
    
    def get_severity(self) -> str:
        """
        Compute severity based on recent scores.
        
        Returns: 'low', 'medium', or 'high'
        """
        if len(self.score_window) == 0:
            return 'low'
        
        recent_scores = [frame['score'] for frame in self.score_window]
        avg_score = np.mean(recent_scores)
        
        if avg_score > 0.7:
            return 'high'
        elif avg_score > 0.5:
            return 'medium'
        else:
            return 'low'
    
    def reset(self):
        """Reset temporal state (e.g., on stream restart)"""
        self.score_window.clear()
        self.consecutive_flagged = 0
        self.frames_since_last_check = 0
        self.prev_frame = None
        self.prev_frame_gray = None
        self.prev_frame_number = None
        self._sampled_mode_detected = False


class TemporalStateManager:
    """
    Manages temporal state for multiple streams.
    Can use Redis for distributed deployments.
    """
    
    def __init__(self, use_redis: bool = False):
        self.use_redis = use_redis
        self.local_state = {}  # stream_id -> TemporalRefinement
        
        if use_redis:
            import redis
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                decode_responses=False
            )
        else:
            self.redis_client = None
    
    def get_stream_state(self, stream_id: str) -> TemporalRefinement:
        """Get or create temporal state for a stream"""
        if stream_id not in self.local_state:
            self.local_state[stream_id] = TemporalRefinement(stream_id)
        
        return self.local_state[stream_id]
    
    def cleanup_stream(self, stream_id: str):
        """Clean up state when stream ends"""
        if stream_id in self.local_state:
            del self.local_state[stream_id]
        
        if self.redis_client:
            # Clean up Redis keys
            pattern = f"stream:{stream_id}:*"
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)


# Global state manager
temporal_state_manager = TemporalStateManager(use_redis=settings.USE_REDIS)
