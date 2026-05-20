"""
Core detection engine — Tier 1 spatial signals.
All detectors operate on the Y (luma) channel: H.264/HEVC artifacts are primarily in luma.
"""
import cv2
import numpy as np
from typing import Dict
from src.core.config import settings


def _to_luma(frame: np.ndarray) -> np.ndarray:
    """Convert BGR frame to Y (luma) channel."""
    if len(frame.shape) == 2:
        return frame.astype(np.float32)
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    return yuv[:, :, 0].astype(np.float32)


class SpatialDetector:
    """Tier 1 — Fast spatial checks on the luma channel."""

    def __init__(self):
        self.edge_threshold        = settings.EDGE_THRESHOLD
        self.color_quant_threshold = settings.COLOR_QUANT_THRESHOLD
        self.grid_sizes            = [8, 16, 32, 64]
        self.edge_block_sizes      = [8, 16, 32, 64]

    def boundary_edge_pairing(self, frame: np.ndarray) -> float:
        """
        Paired gradient check at macroblock boundaries.
        Natural edge: one side strong. Macroblock: both sides strong and aligned.
        Covers H.264 (8/16px), H.265/HEVC (32px), AV1 (64px).
        Returns: score [0, 1]
        """
        luma = _to_luma(frame)
        h, w = luma.shape
        grad_x = cv2.Sobel(luma, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(luma, cv2.CV_64F, 0, 1, ksize=3)
        edge_scores = []

        for block_size in self.edge_block_sizes:
            for x in range(block_size, w - block_size, block_size):
                paired = np.minimum(np.abs(grad_x[:, x - 1]), np.abs(grad_x[:, x]))
                edge_scores.append(np.mean(paired))
            for y in range(block_size, h - block_size, block_size):
                paired = np.minimum(np.abs(grad_y[y - 1, :]), np.abs(grad_y[y, :]))
                edge_scores.append(np.mean(paired))

        if not edge_scores:
            return 0.0
        return float(min(np.mean(edge_scores) / 255.0, 1.0))

    def grid_periodicity_check(self, frame: np.ndarray) -> float:
        """
        Detects periodic variance dips at 8/16/32/64px intervals.
        Macroblocking creates flat blocks → variance drops at boundaries.
        Returns: score [0, 1]
        """
        luma = _to_luma(frame)
        row_variance = np.var(luma, axis=1)
        col_variance = np.var(luma, axis=0)
        scores = []
        for grid_size in self.grid_sizes:
            r = self._check_grid_alignment(row_variance, grid_size)
            c = self._check_grid_alignment(col_variance, grid_size)
            scores.append((r + c) / 2)
        return float(max(scores)) if scores else 0.0

    def _check_grid_alignment(self, variance_profile: np.ndarray, grid_size: int) -> float:
        n = len(variance_profile)
        positions = list(range(0, n, grid_size))
        if len(positions) < 2:
            return 0.0
        boundary, midpoint = [], []
        for i in range(len(positions) - 1):
            s, e, m = positions[i], positions[i + 1], (positions[i] + positions[i + 1]) // 2
            if s < n and e < n and m < n:
                boundary += [variance_profile[s], variance_profile[e]]
                midpoint.append(variance_profile[m])
        if not boundary or not midpoint:
            return 0.0
        mid_mean = np.mean(midpoint)
        if mid_mean == 0:
            return 0.0
        return float(max(0.0, min(1.0 - np.mean(boundary) / (mid_mean + 1e-6), 1.0)))

    def color_quantization_check(self, frame: np.ndarray) -> float:
        """
        High saturation + low hue variance = color banding artifact.
        Operates in HSV (color domain). Returns: score [0, 1]
        """
        if len(frame.shape) != 3:
            return 0.0
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        block_size = 16
        scores = []
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block    = hsv[y:y + block_size, x:x + block_size]
                mean_sat = np.mean(block[:, :, 1])
                if mean_sat < 50:
                    continue
                if np.var(block[:, :, 0]) < 20:
                    scores.append(mean_sat / 255.0)
        return float(np.mean(scores)) if scores else 0.0

    def compute_tier1_signals(self, frame: np.ndarray) -> Dict[str, float]:
        """Run all Tier 1 checks and return signals dict."""
        edge_score        = self.boundary_edge_pairing(frame)
        grid_score        = self.grid_periodicity_check(frame)
        color_quant_score = self.color_quantization_check(frame)
        return {
            'edge_score':        edge_score,
            'grid_score':        grid_score,
            'color_quant_score': color_quant_score,
            'passes_gate':       (
                edge_score > self.edge_threshold or
                color_quant_score > self.color_quant_threshold
            )
        }


def compute_dct_score(frame: np.ndarray, block_size: int = 8) -> float:
    """
    DCT periodic pattern analysis on luma channel.
    Compares AC energy at grid-aligned block rows vs interior rows.
    Macroblocking creates lower AC energy at codec grid positions.
    Returns: score [0, 1] — informational only, not used in corroboration.
    """
    luma = _to_luma(frame)
    h, w = luma.shape
    block_rows, block_cols = h // block_size, w // block_size
    if block_rows < 4 or block_cols < 4:
        return 0.0

    ac_map = np.full((block_rows, block_cols), -1.0, dtype=np.float32)
    for br in range(block_rows):
        for bc in range(block_cols):
            block = luma[br * block_size:(br + 1) * block_size,
                         bc * block_size:(bc + 1) * block_size]
            if block.mean() < 10 or block.mean() > 245:
                continue
            dct_block = cv2.dct(block.astype(np.float32))
            dc = float(dct_block[0, 0] ** 2)
            ac = float(np.sum(dct_block[1:, :] ** 2) + np.sum(dct_block[:, 1:] ** 2))
            ac_map[br, bc] = ac / (dc + 1e-6)

    grid_ac, interior_ac = [], []
    for br in range(block_rows):
        valid = ac_map[br, :][ac_map[br, :] >= 0]
        if len(valid) == 0:
            continue
        (grid_ac if br % 2 == 0 else interior_ac).append(float(valid.mean()))

    if not grid_ac or not interior_ac:
        return 0.0
    interior_mean = float(np.mean(interior_ac))
    if interior_mean < 1e-6:
        return 0.0
    return float(np.clip(1.0 - float(np.mean(grid_ac)) / (interior_mean + 1e-6), 0.0, 1.0))


def compute_block_variance_score(frame: np.ndarray, block_size: int = 8) -> float:
    """
    Pixel-level flatness detection on luma channel.
    Flags blocks with: flat interior + strong boundary gradient + no interior gradients.
    Distinguishes macroblocked blocks from natural smooth regions (bokeh, sky).
    Returns: score [0, 1]
    """
    luma = _to_luma(frame)
    h, w = luma.shape
    sobelx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
    grad   = np.sqrt(sobelx ** 2 + sobely ** 2)

    artifact_blocks = 0
    total_blocks    = 0
    for by in range(0, h - block_size, block_size):
        for bx in range(0, w - block_size, block_size):
            bluma = luma[by:by + block_size, bx:bx + block_size]
            bgrad = grad[by:by + block_size, bx:bx + block_size]
            boundary_grad = np.concatenate([bgrad[0,:], bgrad[-1,:], bgrad[:,0], bgrad[:,-1]])
            total_blocks += 1
            if (float(np.std(bluma[1:-1, 1:-1])) < 3.0 and
                    float(np.mean(boundary_grad)) > 15.0 and
                    float(grad[by+1:by+block_size-1, bx+1:bx+block_size-1].mean()) < 5.0):
                artifact_blocks += 1

    return float(artifact_blocks / total_blocks) if total_blocks > 0 else 0.0


def compute_frozen_block_score(frame: np.ndarray, prev_frame: np.ndarray, block_size: int = 8) -> float:
    """
    Detects frozen/repeated blocks — a transmission error artifact where decoder
    reuses blocks from the previous frame due to packet loss or bitstream corruption.

    Real stream glitches: 20–65% of blocks are frozen (identical to prev frame).
    Normal motion video: < 5% of blocks are frozen (only truly static regions).

    This is distinct from macroblocking/pixelation — the block content is valid
    but temporally wrong (copied from wrong frame).

    Args:
        frame:      Current frame (BGR)
        prev_frame: Previous frame (BGR)
        block_size: Block size in pixels (default 8)

    Returns:
        Fraction of frozen blocks [0, 1]
    """
    if prev_frame is None:
        return 0.0

    luma_curr = _to_luma(frame)
    luma_prev = _to_luma(prev_frame)

    if luma_curr.shape != luma_prev.shape:
        return 0.0

    h, w = luma_curr.shape
    frozen = 0
    total  = 0

    for by in range(0, h - block_size, block_size):
        for bx in range(0, w - block_size, block_size):
            bc = luma_curr[by:by + block_size, bx:bx + block_size]
            bp = luma_prev[by:by + block_size, bx:bx + block_size]
            total += 1
            # Mean absolute difference < 1.0 luma unit = effectively identical
            if float(np.mean(np.abs(bc.astype(np.float32) - bp.astype(np.float32)))) < 1.0:
                frozen += 1

    return float(frozen / total) if total > 0 else 0.0


def compute_block_boundary_density(frame: np.ndarray, block_size: int = 8, threshold: float = 5.0) -> float:
    """
    Fraction of block-aligned boundaries (at 8px intervals) that have strong
    paired gradient energy on both sides.

    Real macroblocking: nearly ALL 8px boundaries are strong → density ≈ 0.5–1.0
    Color bars:         only bar-edge boundaries are strong  → density ≈ 0.05–0.15
    Smooth stripes:     all boundaries strong BUT block_var=0 (caught by block gate)
    Clean images:       very few strong boundaries           → density ≈ 0.0–0.05

    Returns: fraction [0, 1]
    """
    luma = _to_luma(frame)
    h, w = luma.shape
    sobelx = cv2.Sobel(luma, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(luma, cv2.CV_64F, 0, 1, ksize=3)

    strong = 0
    total  = 0

    # Vertical boundaries (columns at x = block_size, 2*block_size, ...)
    for x in range(block_size, w - block_size, block_size):
        paired = float(np.minimum(np.abs(sobelx[:, x - 1]), np.abs(sobelx[:, x])).mean())
        total += 1
        if paired > threshold:
            strong += 1

    # Horizontal boundaries (rows at y = block_size, 2*block_size, ...)
    for y in range(block_size, h - block_size, block_size):
        paired = float(np.minimum(np.abs(sobely[y - 1, :]), np.abs(sobely[y, :])).mean())
        total += 1
        if paired > threshold:
            strong += 1

    return float(strong / total) if total > 0 else 0.0


def compute_artifact_col_coverage(frame: np.ndarray, block_size: int = 8) -> float:
    """
    Fraction of block-column positions that contain at least one artifact block
    (flat interior + strong boundary + no interior gradients).

    Real macroblocking: artifacts spread across many columns → coverage ≥ 0.35
    Color bars:         artifacts only at bar-edge columns   → coverage ≈ 0.17
    Clean images:       no artifact blocks                   → coverage = 0.0

    This is the key discriminator for the BRISQUE_blockvar path, which can fire
    on color bars (high BRISQUE + non-zero block_var) but should not.

    Returns: fraction [0, 1]
    """
    luma = _to_luma(frame)
    h, w = luma.shape
    sobelx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
    grad   = np.sqrt(sobelx ** 2 + sobely ** 2)

    artifact_cols = set()
    total_cols    = w // block_size

    for by in range(0, h - block_size, block_size):
        for bx in range(0, w - block_size, block_size):
            bluma = luma[by:by + block_size, bx:bx + block_size]
            bgrad = grad[by:by + block_size, bx:bx + block_size]
            boundary_grad = np.concatenate([
                bgrad[0, :], bgrad[-1, :], bgrad[:, 0], bgrad[:, -1]
            ])
            if (float(np.std(bluma[1:-1, 1:-1])) < 3.0 and
                    float(np.mean(boundary_grad)) > 15.0 and
                    float(grad[by + 1:by + block_size - 1,
                               bx + 1:bx + block_size - 1].mean()) < 5.0):
                artifact_cols.add(bx // block_size)

    return float(len(artifact_cols) / total_cols) if total_cols > 0 else 0.0


def compute_brisque_score(frame: np.ndarray) -> float:
    """
    Gradient-ratio quality metric on luma channel.
    Compares gradient energy at 8px boundary rows/cols vs interior rows/cols (3px inside).
    Clean: ratio ≈ 1.0. Blocked: ratio > 2.0.
    Returns: score [0, 100]
    """
    luma = _to_luma(frame)
    h, w = luma.shape
    sobelx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
    grad   = np.sqrt(sobelx ** 2 + sobely ** 2)

    h_b = [i for i in range(0, h, 8) if i < h]
    h_i = [i + 3 for i in range(0, h, 8) if i + 3 < h]
    v_b = [j for j in range(0, w, 8) if j < w]
    v_i = [j + 3 for j in range(0, w, 8) if j + 3 < w]

    boundary = (float(grad[h_b, :].mean()) + float(grad[:, v_b].mean())) / 2
    interior = (float(grad[h_i, :].mean()) + float(grad[:, v_i].mean())) / 2
    ratio    = boundary / (interior + 1e-6)

    lap_var           = float(cv2.Laplacian(luma, cv2.CV_32F).var())
    sharpness_penalty = float(np.clip(100.0 - np.log1p(lap_var) * 8.0, 0.0, 100.0))
    blockiness_score  = float(np.clip((ratio - 1.0) / 3.0 * 100.0, 0.0, 100.0))
    return float(np.clip(0.75 * blockiness_score + 0.25 * sharpness_penalty, 0.0, 100.0))
