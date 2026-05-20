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
        self.frames_since_last_check = 0
        
        # Previous frame for scene cut detection
        self.prev_frame = None
        self.prev_frame_gray = None
        self.prev_frame_number = None

    def scene_cut_guard(
        self,
        frame: np.ndarray,
        signals: Dict[str, float],
        frame_number: Optional[int] = None,
        mb_ratio_score: float = 0.0,
        frozen_score: float = 0.0,
    ) -> bool:
        """
        Scene cut guard: high inter-frame SAD + clean blocks = legitimate transition.

        Fix #3: Do NOT suppress frames where macroblocking or frozen blocks are
        present — a glitched frame can have high SAD (glitch→clean transition)
        which previously caused the guard to fire incorrectly.

        Suppression is skipped when:
          - frozen_score >= 0.15  (frozen blocks = transmission error, not scene cut)
          - mb_ratio_score >= 0.08 (macroblocking present = artifact, not scene cut)

        Args:
            frame:          Current frame
            signals:        Tier 1 signals
            frame_number:   Optional frame number to detect sampling gaps
            mb_ratio_score: Macroblock boundary/interior ratio score
            frozen_score:   Frozen block fraction vs previous frame

        Returns: True if this is a genuine scene cut (skip flagging), False otherwise
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
                self.prev_frame = frame.copy()
                self.prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.prev_frame_number = frame_number
                return False

        # Compute SAD
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sad = float(np.mean(np.abs(
            current_gray.astype(np.float32) - self.prev_frame_gray.astype(np.float32)
        )))

        # Update previous frame
        self.prev_frame = frame.copy()
        self.prev_frame_gray = current_gray
        self.prev_frame_number = frame_number

        # High SAD alone is not enough — must also have no artifact signals
        if sad > settings.SCENE_CUT_SAD_THRESHOLD:
            edge_score = signals.get('edge_score', 0.0)

            # Fix #3: If macroblocking or frozen blocks are present, this is NOT
            # a scene cut — it's a glitch causing high inter-frame difference
            if frozen_score >= 0.15:
                return False  # Frozen blocks → glitch, not scene cut
            # Only bypass scene cut for macroblocking if SAD is low (< 15)
            # High SAD + high mb_ratio = scene cut with residual compression artifacts
            if mb_ratio_score >= 0.15 and sad < 15.0:
                return False  # Macroblocking with low motion → artifact, not scene cut

            if edge_score < settings.EDGE_THRESHOLD:
                return True  # Genuine scene cut

        return False

    def block_persistence_check(self, is_flagged: bool, frame_gap: int = 1) -> bool:
        """
        Artifact must appear in at least 3 consecutive frames (or sampled frames).
        Single noisy frame never triggers alert.
        """
        in_sampled_mode = frame_gap > 5
        if not in_sampled_mode and len(self.score_window) > 0:
            in_sampled_mode = getattr(self, '_sampled_mode_detected', False)

        if frame_gap > 5:
            self._sampled_mode_detected = True
            in_sampled_mode = True

        if in_sampled_mode:
            window_len = len(self.score_window)
            if window_len == 0:
                return is_flagged
            elif window_len == 1:
                return is_flagged and self.score_window[0]['flagged']
            else:
                last_two = list(self.score_window)[-2:]
                prev_flagged_count = sum(1 for f in last_two if f['flagged'])
                return (prev_flagged_count + (1 if is_flagged else 0)) >= 2
        else:
            if is_flagged:
                self.consecutive_flagged += 1
            else:
                self.consecutive_flagged = 0
            return self.consecutive_flagged >= settings.BLOCK_PERSISTENCE_FRAMES

    def qp_bitstream_hint(self, qp_value: Optional[int]) -> float:
        if qp_value is None:
            return 0.0
        if qp_value > settings.QP_THRESHOLD:
            return min((qp_value - settings.QP_THRESHOLD) / 20.0, 0.2)
        return 0.0

    def color_shift_detection(self, frame: np.ndarray) -> bool:
        if self.prev_frame is None:
            return False
        current_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        prev_hsv    = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2HSV)
        sat_delta   = abs(float(np.mean(current_hsv[:, :, 1])) -
                          float(np.mean(prev_hsv[:, :, 1])))
        return sat_delta > settings.COLOR_SHIFT_SATURATION_DELTA

    def update_window(self, composite_score: float, is_flagged: bool):
        self.score_window.append({'score': composite_score, 'flagged': is_flagged})

    def should_alert(self) -> bool:
        """
        Fix #2 (partial): Tighter sliding window — require 70% of frames flagged
        (up from 50%) for partial windows to reduce false positives from window
        carry-over after a glitch burst ends.
        """
        window_len = len(self.score_window)
        if window_len == 0:
            return False

        flagged_count = sum(1 for f in self.score_window if f['flagged'])

        if window_len < self.window_size:
            # Tightened: require 70% for partial window (was 50%)
            return flagged_count / window_len >= 0.70
        else:
            return flagged_count >= self.min_flagged

    def get_severity(self) -> str:
        if len(self.score_window) == 0:
            return 'low'
        avg_score = float(np.mean([f['score'] for f in self.score_window]))
        if avg_score > 0.7:
            return 'high'
        elif avg_score > 0.5:
            return 'medium'
        return 'low'

    def reset(self):
        self.score_window.clear()
        self.consecutive_flagged = 0
        self.frames_since_last_check = 0
        self.prev_frame = None
        self.prev_frame_gray = None
        self.prev_frame_number = None
        self._sampled_mode_detected = False


class TemporalStateManager:
    """Manages temporal state for multiple streams."""

    def __init__(self, use_redis: bool = False):
        self.use_redis   = use_redis
        self.local_state = {}

        if use_redis:
            import redis
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                decode_responses=False,
            )
        else:
            self.redis_client = None

    def get_stream_state(self, stream_id: str) -> TemporalRefinement:
        if stream_id not in self.local_state:
            self.local_state[stream_id] = TemporalRefinement(stream_id)
        return self.local_state[stream_id]

    def cleanup_stream(self, stream_id: str):
        if stream_id in self.local_state:
            del self.local_state[stream_id]
        if self.redis_client:
            keys = self.redis_client.keys(f"stream:{stream_id}:*")
            if keys:
                self.redis_client.delete(*keys)


# Global state manager
temporal_state_manager = TemporalStateManager(use_redis=settings.USE_REDIS)
