"""
Image detection path — single frame analysis.

Production-grade hybrid detection with pixel-level block structure validation.

Core principle: every corroboration signal must be validated against pixel-level
block structure evidence (block_var_score). This eliminates false positives from:
  - Blurred/noisy images (MVAD confusion, no actual block structure)
  - Dark/underexposed images (MVAD confusion, no actual block structure)
  - Color bars / test cards (hard edges but not at every block boundary)
  - Smooth stripe patterns (edge signal fires but no flat block interiors)
  - Uniform color regions (color_quant fires but no block structure)
"""
import cv2
import numpy as np
from typing import Dict
from src.core.detection import (
    SpatialDetector,
    compute_brisque_score,
    compute_block_variance_score,
    compute_dct_score,
    compute_block_boundary_density,
    compute_artifact_col_coverage,
    compute_macroblock_ratio_score,
)
from src.models.mvad_wrapper import model_manager
from src.core.config import settings

# Minimum fraction of blocks that must show flat-interior + strong-boundary pattern
# for any corroboration to be considered valid.
# Real macroblocking: 0.02–1.0. Clean images: 0.0 (except color bars ~0.18).
MIN_BLOCK_VAR_FOR_CORROBORATION = 0.02

# Minimum block_var for MVAD to be trusted with corroboration
MIN_BLOCK_VAR_FOR_MVAD = 0.02

# Minimum block boundary density (fraction of 8px boundaries with strong paired edges)
# Real macroblocking: 0.04–1.0. Color bars: ~0.052. Clean images: 0.0–0.11.
# Used to gate edge corroboration AND BRISQUE corroboration.
MIN_BOUNDARY_DENSITY_FOR_EDGE_CORR   = 0.30
MIN_BOUNDARY_DENSITY_FOR_BRISQUE_CORR = 0.08  # color_bars=0.052 < 0.08, real pix ≥ 0.095

# High-confidence BRISQUE standalone path:
# BRISQUE > 60 AND boundary_density > 0.50 → real blocking even without flat block interiors
# (catches checkerboard: brisque=67, boundary_density=1.0)
BRISQUE_HIGH_STANDALONE_THRESHOLD  = 60.0
BRISQUE_HIGH_DENSITY_THRESHOLD     = 0.50

# Minimum block_var for standalone BRISQUE to fire (normal path)
MIN_BLOCK_VAR_FOR_BRISQUE = 0.02

# Minimum block_var for Tier1 composite to fire
MIN_BLOCK_VAR_FOR_TIER1 = 0.02

# Hybrid fallback threshold — raised from 0.30 to eliminate weak false positives
HYBRID_THRESHOLD = 0.42

# Minimum block_var for Hybrid fallback to fire
MIN_BLOCK_VAR_FOR_HYBRID = 0.02

# Very high MVAD confidence path: if MVAD > 0.90, lower block_var requirement
MVAD_HIGH_CONFIDENCE_THRESHOLD = 0.90
MIN_BLOCK_VAR_FOR_HIGH_MVAD    = 0.01  # essentially just needs any block structure hint


class ImageDetector:
    """
    Single image artifact detection.

    Decision pipeline (priority order):
      1. MVAD > threshold AND block_var > MIN AND corroboration AND coverage → ARTIFACT
      2. MVAD > threshold, no valid corroboration/coverage                  → CLEAN
      3. BRISQUE > threshold AND block_var > MIN                            → ARTIFACT
      4. Tier1 > 0.50 AND block_var > MIN                                   → ARTIFACT
      5. Hybrid fallback AND block_var > MIN                                → ARTIFACT
      6. Default                                                            → CLEAN

    Corroborating signals (require block_var > MIN):
      edge:          edge_score > 0.05 AND boundary_density > 0.30
      brisque:       brisque_score > 30.0
      grid:          grid_score > 0.05
      tier1_spatial: tier1_spatial > 0.035
      block_var:     block_var_score > 0.30 (self-corroborating)
    """

    def __init__(self):
        self.spatial_detector  = SpatialDetector()
        self.edge_threshold    = settings.EDGE_VOTE_THRESHOLD
        self.grid_threshold    = settings.GRID_VOTE_THRESHOLD
        self.mvad_threshold    = settings.MVAD_VOTE_THRESHOLD
        self.brisque_threshold = settings.BRISQUE_VOTE_THRESHOLD

    def analyze(self, image: np.ndarray) -> Dict:
        tier1_signals    = self.spatial_detector.compute_tier1_signals(image)
        mvad_blockiness, mvad_pixelation = model_manager.predict(image)
        brisque_score    = compute_brisque_score(image)
        block_var_score  = compute_block_variance_score(image)
        dct_score        = compute_dct_score(image)
        boundary_density = compute_block_boundary_density(image)
        artifact_col_cov = compute_artifact_col_coverage(image)
        mb_ratio_score   = compute_macroblock_ratio_score(image)

        edge_score    = tier1_signals['edge_score']
        grid_score    = tier1_signals['grid_score']
        color_score   = tier1_signals['color_quant_score']
        mvad_score    = max(mvad_blockiness, mvad_pixelation)
        tier1_score   = edge_score * 0.50 + color_score * 0.40 + grid_score * 0.10
        tier1_spatial = edge_score * 0.70 + grid_score * 0.30

        # ── Block structure gate ──────────────────────────────────────────────
        # block_var_score > 0 means actual flat-interior + strong-boundary blocks
        # exist in the image. This is the ground truth for real macroblocking.
        # Without this, all other signals can fire on non-artifact content.
        # mb_ratio_score also counts as block structure evidence for real broadcast
        # macroblocking where blocks have real content but unnaturally sharp boundaries.
        has_block_structure = (
            block_var_score >= MIN_BLOCK_VAR_FOR_CORROBORATION or
            (mb_ratio_score >= 0.06 and artifact_col_cov >= 0.25)
        )

        # ── Corroboration signals (each gated by block structure) ─────────────
        # Edge corroboration: requires block structure AND high boundary density
        # (eliminates smooth stripes, color bars with sparse boundaries)
        edge_corroborates = (
            edge_score > 0.05 and
            has_block_structure and
            boundary_density >= MIN_BOUNDARY_DENSITY_FOR_EDGE_CORR
        )

        # BRISQUE corroboration: requires block structure AND minimum boundary density
        # (eliminates blurred noise, dark/underexposed images, color bars)
        brisque_corroborates = (
            brisque_score > 30.0 and
            has_block_structure and
            boundary_density >= MIN_BOUNDARY_DENSITY_FOR_BRISQUE_CORR
        )

        # Grid corroboration: requires block structure
        grid_corroborates = (
            grid_score > 0.05 and
            has_block_structure
        )

        # Tier1 spatial: requires block structure
        tier1_spatial_corroborates = (
            tier1_spatial > 0.035 and
            has_block_structure
        )

        # Block variance self-corroborates (it IS the block structure signal)
        block_var_corroborates = block_var_score > 0.30

        # Macroblock ratio: real broadcast macroblocking with content-filled blocks
        # Requires artifact_col_cov >= 0.25 to exclude color bars (col_cov=0.175)
        mb_ratio_corroborates = mb_ratio_score >= 0.08 and artifact_col_cov >= 0.25

        corroborating_signals = {
            'edge':          edge_corroborates,
            'brisque':       brisque_corroborates,
            'grid':          grid_corroborates,
            'tier1_spatial': tier1_spatial_corroborates,
            'block_var':     block_var_corroborates,
            'mb_ratio':      mb_ratio_corroborates,
        }
        has_corroboration = any(corroborating_signals.values())
        corroborating_signal_name = next(
            (k for k, v in corroborating_signals.items() if v), None
        )

        # ── Coverage guard ────────────────────────────────────────────────────
        # Requires actual block structure evidence for coverage to pass
        sufficient_coverage = (
            (has_corroboration and has_block_structure) or
            block_var_score >= 0.15 or
            (mb_ratio_score >= 0.10 and artifact_col_cov >= 0.25) or
            (mvad_score >= 0.60 and has_block_structure)
        )

        # ── Decision tree ─────────────────────────────────────────────────────
        # Path 0: Very high MVAD confidence — lower block_var requirement
        # Catches skin-tone pixelated (MVAD=0.994, block_var=0.04)
        very_high_mvad = mvad_score >= MVAD_HIGH_CONFIDENCE_THRESHOLD
        high_mvad_block_structure = block_var_score >= MIN_BLOCK_VAR_FOR_HIGH_MVAD

        if (very_high_mvad and
                high_mvad_block_structure and
                (has_corroboration or brisque_score > 30.0)):
            artifact_detected = True
            confidence        = mvad_score
            decision_maker    = f'MVAD_highconf+{corroborating_signal_name or "brisque"}'

        # Path 1: MVAD + corroboration + coverage (main path)
        elif (mvad_score > self.mvad_threshold and
                has_block_structure and
                has_corroboration and
                sufficient_coverage):
            artifact_detected = True
            confidence        = mvad_score
            decision_maker    = f'MVAD+{corroborating_signal_name}'

        # Path 2: High BRISQUE + high boundary density (catches checkerboard)
        # Must be checked BEFORE MVAD_unconfirmed so it can rescue cases where
        # MVAD fires but has no block structure (checkerboard: MVAD=0.557, brisque=67, density=1.0)
        elif (brisque_score >= BRISQUE_HIGH_STANDALONE_THRESHOLD and
              boundary_density >= BRISQUE_HIGH_DENSITY_THRESHOLD):
            artifact_detected = True
            confidence        = max(brisque_score / 100.0, mvad_score)
            decision_maker    = 'BRISQUE_highconf'

        # Path 3: High BRISQUE + block structure (catches sky_pixelated: brisque=95, block_var=0.17)
        # Requires artifact_col_coverage > 0.25 to exclude color bars (col_cov=0.175)
        elif (brisque_score >= BRISQUE_HIGH_STANDALONE_THRESHOLD and
              block_var_score >= 0.15 and
              artifact_col_cov >= 0.25):
            artifact_detected = True
            confidence        = max(brisque_score / 100.0, mvad_score)
            decision_maker    = 'BRISQUE_blockvar'

        # Path 4: MVAD above threshold but no valid corroboration → suppress
        elif mvad_score > self.mvad_threshold and (
                not has_block_structure or
                not has_corroboration or
                not sufficient_coverage):
            artifact_detected = False
            confidence        = mvad_score * 0.40
            decision_maker    = 'MVAD_unconfirmed'

        # Path 5: Standalone BRISQUE (normal threshold) with block structure
        elif (brisque_score > self.brisque_threshold and
              block_var_score >= MIN_BLOCK_VAR_FOR_BRISQUE):
            artifact_detected = True
            confidence        = brisque_score / 100.0
            decision_maker    = 'BRISQUE'

        # Path 6: Tier1 composite with block structure
        elif (tier1_score > 0.50 and
              block_var_score >= MIN_BLOCK_VAR_FOR_TIER1):
            artifact_detected = True
            confidence        = tier1_score
            decision_maker    = 'Tier1'

        # Path 7: Hybrid fallback (last resort)
        else:
            confidence        = mvad_score * 0.60 + tier1_score * 0.40
            artifact_detected = (
                confidence > HYBRID_THRESHOLD and
                block_var_score >= MIN_BLOCK_VAR_FOR_HYBRID
            )
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
                'boundary_density':   boundary_density,
                'artifact_col_cov':   artifact_col_cov,
                'mb_ratio_score':     mb_ratio_score,
            },
            'voting': {
                'artifact_votes': sum(votes.values()),
                'clean_votes':    4 - sum(votes.values()),
                'details':        vote_details,
            },
            'corroboration': {
                'has_corroboration':   has_corroboration,
                'sufficient_coverage': sufficient_coverage,
                'has_block_structure': has_block_structure,
                'boundary_density':    boundary_density,
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
                f'BlockVar={block_var_score:.3f}, BoundaryDensity={boundary_density:.3f}, '
                f'MBRatio={mb_ratio_score:.3f}, '
                f'HasBlockStructure={has_block_structure}, Coverage={sufficient_coverage}'
            ),
        }


image_detector = ImageDetector()
