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
    compute_block_boundary_density,
    compute_artifact_col_coverage,
    compute_frozen_block_score,
    compute_macroblock_ratio_score,
)
from src.models.mvad_wrapper import model_manager
from src.core.temporal import temporal_state_manager
from src.core.config import settings

# Mirror the same gates as image_detector for consistency
MIN_BLOCK_VAR_FOR_CORROBORATION    = 0.02
MIN_BOUNDARY_DENSITY_FOR_EDGE_CORR  = 0.30
MIN_BOUNDARY_DENSITY_FOR_BRISQUE_CORR = 0.08
MIN_BLOCK_VAR_FOR_BRISQUE          = 0.02
MIN_BLOCK_VAR_FOR_TIER1            = 0.02
MIN_BLOCK_VAR_FOR_HYBRID           = 0.02
HYBRID_THRESHOLD                   = 0.42
BRISQUE_HIGH_STANDALONE_THRESHOLD  = 60.0
BRISQUE_HIGH_DENSITY_THRESHOLD     = 0.50
MVAD_HIGH_CONFIDENCE_THRESHOLD     = 0.90
MIN_BLOCK_VAR_FOR_HIGH_MVAD        = 0.01


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

        # Compute frozen score BEFORE scene cut guard so we can override it
        # A frozen frame (>50% blocks identical to prev) is a glitch, not a scene cut
        pre_frozen_score = compute_frozen_block_score(frame, temporal_state.prev_frame)

        # Pre-compute mb_ratio for scene cut guard (Fix #3)
        pre_mb_ratio = compute_macroblock_ratio_score(frame)

        if (pre_frozen_score < 0.50 and
                temporal_state.scene_cut_guard(
                    frame, tier1_signals, frame_number,
                    mb_ratio_score=pre_mb_ratio,
                    frozen_score=pre_frozen_score,
                )):
            # Scene cut: reset consecutive counter so glitch burst doesn't carry over
            temporal_state.consecutive_flagged = 0
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
        brisque_score    = compute_brisque_score(frame)
        block_var_score  = compute_block_variance_score(frame)
        dct_score        = compute_dct_score(frame)
        color_shift      = temporal_state.color_shift_detection(frame)
        boundary_density = compute_block_boundary_density(frame)
        artifact_col_cov = compute_artifact_col_coverage(frame)
        mb_ratio_score   = compute_macroblock_ratio_score(frame)

        # Frozen block detection — requires previous frame from temporal state
        frozen_block_score = compute_frozen_block_score(frame, temporal_state.prev_frame)
        # Note: is_frozen_artifact/is_frozen_definitive computed after mvad_score below

        mvad_score    = max(mvad_blockiness, mvad_pixelation)

        # Three-tier frozen block classification (needs mvad_score):
        # Tier A: frozen >= 0.50 → definitive glitch regardless of MVAD
        # Tier B: frozen >= 0.15 AND MVAD >= 0.40 → confirmed glitch
        FROZEN_BLOCK_THRESHOLD      = 0.15
        FROZEN_DEFINITIVE_THRESHOLD = 0.50
        is_frozen_definitive = frozen_block_score >= FROZEN_DEFINITIVE_THRESHOLD
        is_frozen_artifact   = (
            frozen_block_score >= FROZEN_BLOCK_THRESHOLD and
            (mvad_score >= 0.40 or is_frozen_definitive)
        )
        tier1_score   = (
            tier1_signals['edge_score']        * 0.50 +
            tier1_signals['color_quant_score'] * 0.40 +
            tier1_signals['grid_score']        * 0.10
        )
        tier1_spatial = tier1_signals['edge_score'] * 0.70 + tier1_signals['grid_score'] * 0.30

        # Block structure gate — same logic as image_detector
        has_block_structure = (
            block_var_score >= MIN_BLOCK_VAR_FOR_CORROBORATION or
            (mb_ratio_score >= 0.06 and artifact_col_cov >= 0.25)
        )

        edge_corroborates = (
            tier1_signals['edge_score'] > 0.05 and
            has_block_structure and
            boundary_density >= MIN_BOUNDARY_DENSITY_FOR_EDGE_CORR
        )
        brisque_corroborates   = (brisque_score > 30.0 and has_block_structure and
                                   boundary_density >= MIN_BOUNDARY_DENSITY_FOR_BRISQUE_CORR)
        grid_corroborates      = tier1_signals['grid_score'] > 0.05 and has_block_structure
        tier1_spatial_corr     = tier1_spatial > 0.035 and has_block_structure
        block_var_corroborates = block_var_score > 0.30
        mb_ratio_corroborates  = mb_ratio_score >= 0.08 and artifact_col_cov >= 0.25

        corroborating = (
            edge_corroborates or
            brisque_corroborates or
            grid_corroborates or
            tier1_spatial_corr or
            block_var_corroborates or
            mb_ratio_corroborates
        )

        sufficient_coverage = (
            (corroborating and has_block_structure) or
            block_var_score >= 0.15 or
            (mb_ratio_score >= 0.10 and artifact_col_cov >= 0.25) or
            (mvad_score >= 0.60 and has_block_structure)
        )

        very_high_mvad = mvad_score >= MVAD_HIGH_CONFIDENCE_THRESHOLD
        high_mvad_block_structure = block_var_score >= MIN_BLOCK_VAR_FOR_HIGH_MVAD

        # Path -1: Frozen block artifact (transmission/decoder error)
        # Tier A: frozen >= 0.50 → definitive glitch, no MVAD needed
        # Tier B: frozen >= 0.15 AND MVAD >= 0.40 → confirmed glitch
        if is_frozen_definitive:
            confidence, is_flagged, decision_maker = (
                frozen_block_score,
                True,
                'FrozenBlock_definitive'
            )
        elif is_frozen_artifact:
            confidence, is_flagged, decision_maker = (
                max(frozen_block_score, mvad_score * 0.8),
                True,
                'FrozenBlock'
            )
        # Path -0.5: Macroblock ratio + MVAD (real broadcast macroblocking)
        elif mb_ratio_score >= 0.08 and artifact_col_cov >= 0.25 and mvad_score >= 0.40:
            confidence, is_flagged, decision_maker = mvad_score, True, 'MBRatio+MVAD'
        elif (very_high_mvad and high_mvad_block_structure and
                (corroborating or brisque_score > 30.0)):
            confidence, is_flagged, decision_maker = mvad_score, True, 'MVAD_highconf'
        elif (mvad_score > self.mvad_threshold and
                has_block_structure and corroborating and sufficient_coverage):
            confidence, is_flagged, decision_maker = mvad_score, True, 'MVAD'
        elif (brisque_score >= BRISQUE_HIGH_STANDALONE_THRESHOLD and
              boundary_density >= BRISQUE_HIGH_DENSITY_THRESHOLD):
            confidence, is_flagged, decision_maker = max(brisque_score/100.0, mvad_score), True, 'BRISQUE_highconf'
        elif (brisque_score >= BRISQUE_HIGH_STANDALONE_THRESHOLD and
              block_var_score >= 0.15 and
              artifact_col_cov >= 0.25):
            confidence, is_flagged, decision_maker = max(brisque_score/100.0, mvad_score), True, 'BRISQUE_blockvar'
        elif mvad_score > self.mvad_threshold and (
                not has_block_structure or not corroborating or not sufficient_coverage):
            confidence, is_flagged, decision_maker = mvad_score * 0.40, False, 'MVAD_unconfirmed'
        elif brisque_score > self.brisque_threshold and block_var_score >= MIN_BLOCK_VAR_FOR_BRISQUE:
            confidence, is_flagged, decision_maker = brisque_score / 100.0, True, 'BRISQUE'
        elif tier1_score > 0.50 and block_var_score >= MIN_BLOCK_VAR_FOR_TIER1:
            confidence, is_flagged, decision_maker = tier1_score, True, 'Tier1'
        else:
            confidence     = mvad_score * 0.60 + tier1_score * 0.40
            is_flagged     = (confidence > HYBRID_THRESHOLD and
                              block_var_score >= MIN_BLOCK_VAR_FOR_HYBRID)
            decision_maker = 'Hybrid'

        if qp_value is not None:
            confidence = min(confidence + temporal_state.qp_bitstream_hint(qp_value), 1.0)

        persistence_met = temporal_state.block_persistence_check(is_flagged, frame_gap)
        temporal_state.update_window(confidence, is_flagged)
        alert_fired = temporal_state.should_alert() and persistence_met

        # Fix #2: artifact_detected is the per-frame signal — it must be grounded
        # in the frame's own signals, not just the temporal window carry-over.
        # is_flagged already reflects the frame's own detection paths above.
        # We do NOT override is_flagged with the window here — that's alert_fired's job.

        artifact_type = None
        if alert_fired:
            if decision_maker in ('FrozenBlock', 'FrozenBlock_definitive'):
                artifact_type = 'frozen_blocks'
            elif decision_maker == 'MBRatio+MVAD':
                artifact_type = 'macroblocking'
            else:
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
                'boundary_density':   boundary_density,
                'artifact_col_cov':   artifact_col_cov,
                'frozen_block_score': frozen_block_score,
                'mb_ratio_score':     mb_ratio_score,
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
