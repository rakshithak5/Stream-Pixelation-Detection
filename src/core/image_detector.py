"""
Image detection path — single frame analysis.
"""
import cv2
import numpy as np
from typing import Dict
from src.core.detection import (
    SpatialDetector,
    compute_brisque_score,
    compute_block_variance_score,
    compute_dct_score,
)
from src.models.mvad_wrapper import model_manager
from src.core.config import settings

MIN_COVERAGE_FRACTION = 0.10


class ImageDetector:
    """Single image artifact detection."""

    def __init__(self):
        self.spatial_detector  = SpatialDetector()
        self.edge_threshold    = settings.EDGE_VOTE_THRESHOLD
        self.grid_threshold    = settings.GRID_VOTE_THRESHOLD
        self.mvad_threshold    = settings.MVAD_VOTE_THRESHOLD
        self.brisque_threshold = settings.BRISQUE_VOTE_THRESHOLD

    def analyze(self, image: np.ndarray) -> Dict:
        """
        Hybrid MVAD-Primary with corroboration and coverage guards.

        Decision:
          MVAD > threshold AND corroboration AND coverage → ARTIFACT
          MVAD > threshold, no corroboration/coverage    → CLEAN (MVAD_unconfirmed)
          BRISQUE > threshold                            → ARTIFACT
          Tier1 > 0.50                                   → ARTIFACT
          Weighted 60/40 fallback                        → borderline

        Corroborating signals (any one sufficient):
          edge > 0.05, brisque > 30, grid > 0.05,
          tier1_spatial > 0.035, block_var > 0.30

        Coverage: passes if corroboration present, block_var >= 10%, or MVAD >= 0.60
        """
        tier1_signals   = self.spatial_detector.compute_tier1_signals(image)
        mvad_blockiness, mvad_pixelation = model_manager.predict(image)
        brisque_score   = compute_brisque_score(image)
        block_var_score = compute_block_variance_score(image)
        dct_score       = compute_dct_score(image)

        edge_score  = tier1_signals['edge_score']
        grid_score  = tier1_signals['grid_score']
        color_score = tier1_signals['color_quant_score']
        mvad_score  = max(mvad_blockiness, mvad_pixelation)
        tier1_score = edge_score * 0.50 + color_score * 0.40 + grid_score * 0.10
        tier1_spatial = edge_score * 0.70 + grid_score * 0.30

        corroborating_signals = {
            'edge':          edge_score      > 0.05,
            'brisque':       brisque_score   > 30.0,
            'grid':          grid_score      > 0.05,
            'tier1_spatial': tier1_spatial   > 0.035,
            'block_var':     block_var_score > 0.30,
        }
        has_corroboration = any(corroborating_signals.values())
        corroborating_signal_name = next(
            (k for k, v in corroborating_signals.items() if v), None
        )

        sufficient_coverage = (
            has_corroboration or
            block_var_score >= MIN_COVERAGE_FRACTION or
            mvad_score >= 0.60
        )

        if mvad_score > self.mvad_threshold and has_corroboration and sufficient_coverage:
            artifact_detected = True
            confidence        = mvad_score
            decision_maker    = f'MVAD+{corroborating_signal_name}'
        elif mvad_score > self.mvad_threshold and (not has_corroboration or not sufficient_coverage):
            artifact_detected = False
            confidence        = mvad_score * 0.40
            decision_maker    = 'MVAD_unconfirmed'
        elif brisque_score > self.brisque_threshold:
            artifact_detected = True
            confidence        = brisque_score / 100.0
            decision_maker    = 'BRISQUE'
        elif tier1_score > 0.50:
            artifact_detected = True
            confidence        = tier1_score
            decision_maker    = 'Tier1'
        else:
            confidence        = mvad_score * 0.60 + tier1_score * 0.40
            artifact_detected = confidence > 0.30
            decision_maker    = 'Hybrid'

        if artifact_detected:
            artifact_type = 'pixelation' if mvad_pixelation > mvad_blockiness else 'macroblocking'
        else:
            artifact_type = None

        if artifact_detected:
            severity = 'high' if confidence > 0.7 else ('medium' if confidence > 0.5 else 'low')
        else:
            severity = 'none'

        votes = {
            'edge':    edge_score    > self.edge_threshold,
            'grid':    grid_score    > self.grid_threshold,
            'mvad':    mvad_score    > self.mvad_threshold,
            'brisque': brisque_score > self.brisque_threshold,
        }
        vote_details = {
            'edge':    f'{"YES" if votes["edge"]    else "NO"} (score={edge_score:.3f} {">" if votes["edge"]    else "<"}{self.edge_threshold})',
            'grid':    f'{"YES" if votes["grid"]    else "NO"} (score={grid_score:.3f} {">" if votes["grid"]    else "<"}{self.grid_threshold})',
            'mvad':    f'{"YES" if votes["mvad"]    else "NO"} (score={mvad_score:.3f} {">" if votes["mvad"]    else "<"}{self.mvad_threshold})',
            'brisque': f'{"YES" if votes["brisque"] else "NO"} (score={brisque_score:.1f} {">" if votes["brisque"] else "<"}{self.brisque_threshold})',
        }

        return {
            'artifact_detected': artifact_detected,
            'confidence':        round(confidence, 4),
            'artifact_type':     artifact_type,
            'severity':          severity,
            'signals': {
                'boundary_edge':      edge_score,
                'grid_periodicity':   grid_score,
                'color_quantization': color_score,
                'mvad_blockiness':    mvad_blockiness,
                'mvad_pixelation':    mvad_pixelation,
                'brisque':            brisque_score,
                'block_variance':     block_var_score,
                'dct_score':          dct_score,
            },
            'voting': {
                'artifact_votes': sum(votes.values()),
                'clean_votes':    4 - sum(votes.values()),
                'details':        vote_details,
            },
            'corroboration': {
                'has_corroboration':   has_corroboration,
                'sufficient_coverage': sufficient_coverage,
                'corroborating':       corroborating_signal_name,
                'signals_checked': {
                    k: f'{"✅" if v else "❌"} {k}'
                    for k, v in corroborating_signals.items()
                }
            },
            'tier':           2,
            'decision_maker': decision_maker,
            'note': (
                f'Detection by {decision_maker}: '
                f'MVAD={mvad_score:.3f}, Tier1={tier1_score:.3f}, '
                f'BRISQUE={brisque_score:.1f}, DCT={dct_score:.3f}, '
                f'BlockVar={block_var_score:.3f}, Coverage={sufficient_coverage}'
            ),
        }


image_detector = ImageDetector()
