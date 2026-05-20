"""
Video detection path — frame loop with temporal state.
"""
import cv2
import numpy as np
from typing import Dict, List, Optional
from src.core.detection import (
    SpatialDetector,
    compute_brisque_score,
    compute_block_variance_score,
    compute_dct_score,
)
from src.models.mvad_wrapper import model_manager
from src.core.temporal import temporal_state_manager
from src.core.config import settings


class VideoDetector:
    """Video stream artifact detection with temporal refinement."""

    def __init__(self):
        self.spatial_detector  = SpatialDetector()
        self.mvad_threshold    = settings.MVAD_VOTE_THRESHOLD
        self.brisque_threshold = settings.BRISQUE_VOTE_THRESHOLD

    def analyze_frame(
        self,
        frame: np.ndarray,
        stream_id: str,
        frame_number: Optional[int] = None,
        qp_value: Optional[int] = None,
        temporal_context: Optional[List[np.ndarray]] = None
    ) -> Dict:
        """
        Analyze single frame with Hybrid MVAD-Primary logic + temporal refinement.
        Same corroboration and coverage guards as image detector.
        Adds: scene cut guard, block persistence, sliding window alert.
        """
        temporal_state = temporal_state_manager.get_stream_state(stream_id)

        frame_gap = 1
        if frame_number is not None:
            if temporal_state.prev_frame_number is not None:
                frame_gap = frame_number - temporal_state.prev_frame_number
            elif frame_number > 5:
                frame_gap = frame_number

        tier1_signals = self.spatial_detector.compute_tier1_signals(frame)

        if temporal_state.scene_cut_guard(frame, tier1_signals, frame_number):
            temporal_state.update_window(0.0, False)
            return {
                'artifact_detected': False,
                'alert_fired':       False,
                'confidence':        0.0,
                'artifact_type':     'scene_cut',
                'severity':          'none',
                'signals':           tier1_signals,
                'tier':              3,
                'stream_id':         stream_id,
                'note':              'Scene cut detected - skipped'
            }

        mvad_blockiness, mvad_pixelation = model_manager.predict(frame, temporal_context)
        brisque_score   = compute_brisque_score(frame)
        block_var_score = compute_block_variance_score(frame)
        dct_score       = compute_dct_score(frame)
        color_shift     = temporal_state.color_shift_detection(frame)

        mvad_score  = max(mvad_blockiness, mvad_pixelation)
        tier1_score = (
            tier1_signals['edge_score']       * 0.50 +
            tier1_signals['color_quant_score'] * 0.40 +
            tier1_signals['grid_score']        * 0.10
        )
        tier1_spatial = tier1_signals['edge_score'] * 0.70 + tier1_signals['grid_score'] * 0.30

        corroborating = (
            tier1_signals['edge_score'] > 0.05 or
            brisque_score               > 30.0  or
            tier1_signals['grid_score'] > 0.05  or
            tier1_spatial               > 0.035 or
            block_var_score             > 0.30
        )

        sufficient_coverage = block_var_score >= 0.10 or mvad_score >= 0.40

        if mvad_score > self.mvad_threshold and corroborating and sufficient_coverage:
            confidence, is_flagged, decision_maker = mvad_score, True, 'MVAD'
        elif mvad_score > self.mvad_threshold and (not corroborating or not sufficient_coverage):
            confidence, is_flagged, decision_maker = mvad_score * 0.40, False, 'MVAD_unconfirmed'
        elif brisque_score > self.brisque_threshold:
            confidence, is_flagged, decision_maker = brisque_score / 100.0, True, 'BRISQUE'
        elif tier1_score > 0.50:
            confidence, is_flagged, decision_maker = tier1_score, True, 'Tier1'
        else:
            confidence     = mvad_score * 0.60 + tier1_score * 0.40
            is_flagged     = confidence > 0.30
            decision_maker = 'Hybrid'

        if qp_value is not None:
            confidence = min(confidence + temporal_state.qp_bitstream_hint(qp_value), 1.0)

        persistence_met = temporal_state.block_persistence_check(is_flagged, frame_gap)
        temporal_state.update_window(confidence, is_flagged)
        alert_fired = temporal_state.should_alert() and persistence_met

        artifact_type = None
        if alert_fired:
            artifact_type = 'macroblocking' if mvad_blockiness > mvad_pixelation else 'pixelation'

        return {
            'artifact_detected': is_flagged,
            'alert_fired':       alert_fired,
            'confidence':        confidence,
            'artifact_type':     artifact_type,
            'severity':          temporal_state.get_severity() if alert_fired else 'none',
            'signals': {
                'boundary_edge':      tier1_signals['edge_score'],
                'grid_periodicity':   tier1_signals['grid_score'],
                'color_quantization': tier1_signals['color_quant_score'],
                'mvad_blockiness':    mvad_blockiness,
                'mvad_pixelation':    mvad_pixelation,
                'brisque':            brisque_score,
                'block_variance':     block_var_score,
                'dct_score':          dct_score,
                'color_shift':        color_shift,
            },
            'temporal': {
                'consecutive_flagged': temporal_state.consecutive_flagged,
                'window_flagged_count': sum(1 for f in temporal_state.score_window if f['flagged']),
                'window_size':         len(temporal_state.score_window),
                'should_alert':        temporal_state.should_alert(),
                'persistence_met':     persistence_met,
            },
            'tier':           2,
            'stream_id':      stream_id,
            'decision_maker': decision_maker,
            'note':           f'Detection by {decision_maker}: MVAD={mvad_score:.3f}, Tier1={tier1_score:.3f}',
        }

    def reset_stream(self, stream_id: str):
        temporal_state_manager.cleanup_stream(stream_id)


video_detector = VideoDetector()
