"""
Video detection path - frame loop with temporal state
"""
import cv2
import numpy as np
from typing import Dict, List, Optional
from src.core.detection import SpatialDetector, CompositeScorer, compute_brisque_score
from src.models.mvad_wrapper import model_manager
from src.core.temporal import temporal_state_manager
from src.core.config import settings


class VideoDetector:
    """Video stream artifact detection with temporal refinement"""

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
        Analyze single frame in video stream context using Hybrid MVAD-Primary approach.
        
        Decision logic (same as image detection):
        1. If MVAD confident (>0.35) → Trust MVAD
        2. If Tier 1 confident (>0.50) → Trust Tier 1
        3. Otherwise → Weighted combination
        
        Args:
            frame: Current frame (H, W, 3)
            stream_id: Unique stream identifier
            frame_number: Optional frame number for gap detection
            qp_value: Optional QP value from encoder bitstream
            temporal_context: Optional list of surrounding frames for MVAD
        
        Returns:
            Detection result with temporal refinement
        """
        # Get temporal state for this stream
        temporal_state = temporal_state_manager.get_stream_state(stream_id)
        
        # Calculate frame gap BEFORE scene_cut_guard (which updates prev_frame_number)
        frame_gap = 1
        if frame_number is not None:
            if temporal_state.prev_frame_number is not None:
                frame_gap = frame_number - temporal_state.prev_frame_number
            elif frame_number == 0:
                # First frame at position 0 - check if we're sampling by looking at next expected frame
                # If this is truly frame 0, we'll know on the next frame
                frame_gap = 1  # Assume consecutive for now
            elif frame_number > 5:
                # First frame but frame_number > 5 means we're definitely sampling
                frame_gap = frame_number
        
        # Tier 1 - Compute spatial signals
        tier1_signals = self.spatial_detector.compute_tier1_signals(frame)
        
        # Tier 3 - Scene cut guard (before ML to save compute)
        if temporal_state.scene_cut_guard(frame, tier1_signals, frame_number):
            temporal_state.update_window(0.0, False)
            return {
                'artifact_detected': False,
                'alert_fired': False,
                'confidence': 0.0,
                'artifact_type': 'scene_cut',
                'severity': 'none',
                'signals': tier1_signals,
                'tier': 3,
                'stream_id': stream_id,
                'note': 'Scene cut detected - skipped'
            }
        
        # Tier 2 - ML verification with temporal context (ALWAYS run, no gate)
        mvad_blockiness, mvad_pixelation = model_manager.predict(frame, temporal_context)
        
        # Add BRISQUE score
        brisque_score = compute_brisque_score(frame)
        
        # Tier 3 - Color shift detection
        color_shift_detected = temporal_state.color_shift_detection(frame)
        
        # Hybrid MVAD-Primary decision logic (same as image detection)
        mvad_score = max(mvad_blockiness, mvad_pixelation)
        
        tier1_score = (
            tier1_signals['edge_score'] * 0.50 +
            tier1_signals['color_quant_score'] * 0.40 +
            tier1_signals['grid_score'] * 0.10
        )
        
        # Corroboration check — same as image detector
        # MVAD alone is not trusted; at least one spatial signal must agree
        tier1_spatial = tier1_signals['edge_score'] * 0.70 + tier1_signals['grid_score'] * 0.30
        corroborating = (
            tier1_signals['edge_score'] > 0.05 or
            brisque_score               > 30.0  or
            tier1_signals['grid_score'] > 0.05  or
            tier1_spatial               > 0.035
        )

        # Decision logic
        if mvad_score > self.mvad_threshold and corroborating:
            confidence     = mvad_score
            is_flagged     = True
            decision_maker = 'MVAD'
        elif mvad_score > self.mvad_threshold and not corroborating:
            # MVAD fires but no spatial signal agrees — likely false positive
            confidence     = mvad_score * 0.40
            is_flagged     = False
            decision_maker = 'MVAD_unconfirmed'
        elif brisque_score > self.brisque_threshold:
            confidence     = brisque_score / 100.0
            is_flagged     = True
            decision_maker = 'BRISQUE'
        elif tier1_score > 0.50:
            confidence     = tier1_score
            is_flagged     = True
            decision_maker = 'Tier1'
        else:
            confidence     = mvad_score * 0.60 + tier1_score * 0.40
            is_flagged     = confidence > 0.30
            decision_maker = 'Hybrid'
        
        # Apply QP bitstream hint if available
        if qp_value is not None:
            qp_boost = temporal_state.qp_bitstream_hint(qp_value)
            confidence = min(confidence + qp_boost, 1.0)
        
        # Tier 3 - Block persistence check (adaptive for sampled analysis)
        persistence_met = temporal_state.block_persistence_check(is_flagged, frame_gap)
        
        # Update sliding window
        temporal_state.update_window(confidence, is_flagged)
        
        # Check if alert should fire (8 of 15 frames rule)
        alert_fired = temporal_state.should_alert() and persistence_met
        
        # Debug info
        should_alert_result = temporal_state.should_alert()
        
        # Classify artifact type
        artifact_type = None
        if alert_fired:
            if mvad_blockiness > mvad_pixelation:
                artifact_type = 'macroblocking'
            else:
                artifact_type = 'pixelation'
        
        # Get severity from temporal window
        severity = temporal_state.get_severity() if alert_fired else 'none'
        
        return {
            'artifact_detected': is_flagged,
            'alert_fired': alert_fired,
            'confidence': confidence,
            'artifact_type': artifact_type,
            'severity': severity,
            'signals': {
                'boundary_edge': tier1_signals['edge_score'],
                'grid_periodicity': tier1_signals['grid_score'],
                'color_quantization': tier1_signals['color_quant_score'],
                'mvad_blockiness': mvad_blockiness,
                'mvad_pixelation': mvad_pixelation,
                'brisque': brisque_score,
                'color_shift': color_shift_detected
            },
            'temporal': {
                'consecutive_flagged': temporal_state.consecutive_flagged,
                'window_flagged_count': sum(1 for f in temporal_state.score_window if f['flagged']),
                'window_size': len(temporal_state.score_window),
                'should_alert': should_alert_result,
                'persistence_met': persistence_met
            },
            'tier': 2,
            'stream_id': stream_id,
            'decision_maker': decision_maker,
            'note': f'Detection by {decision_maker}: MVAD={mvad_score:.3f}, Tier1={tier1_score:.3f}'
        }
    
    def reset_stream(self, stream_id: str):
        """Reset temporal state for a stream (e.g., on stream restart)"""
        temporal_state_manager.cleanup_stream(stream_id)


# Global detector instance
video_detector = VideoDetector()
