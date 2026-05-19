import cv2
import numpy as np
from typing import Tuple, Dict
from src.core.config import settings


class SpatialDetector:
    """Tier 1 - Fast spatial checks that gate 70-80% of clean frames"""
    
    def __init__(self):
        self.edge_threshold = settings.EDGE_THRESHOLD
        self.color_quant_threshold = settings.COLOR_QUANT_THRESHOLD
        self.grid_sizes = settings.GRID_SIZES
    
    def boundary_edge_pairing(self, frame: np.ndarray) -> float:
        """
        Check both sides of macroblock boundaries simultaneously.
        Natural edges: one side strong.
        Macroblock discontinuity: both sides strong and aligned.
        
        Returns: edge score [0, 1], higher = more blocking artifacts
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        h, w = gray.shape
        
        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        edge_scores = []
        
        # Check 8x8 and 16x16 block boundaries (most common in H.264/HEVC)
        for block_size in [8, 16]:
            # Vertical boundaries
            for x in range(block_size, w - block_size, block_size):
                left_edge = np.abs(grad_x[:, x - 1])
                right_edge = np.abs(grad_x[:, x])
                
                # Both sides strong = artifact
                paired_strength = np.minimum(left_edge, right_edge)
                edge_scores.append(np.mean(paired_strength))
            
            # Horizontal boundaries
            for y in range(block_size, h - block_size, block_size):
                top_edge = np.abs(grad_y[y - 1, :])
                bottom_edge = np.abs(grad_y[y, :])
                
                paired_strength = np.minimum(top_edge, bottom_edge)
                edge_scores.append(np.mean(paired_strength))
        
        if not edge_scores:
            return 0.0
        
        # Normalize to [0, 1]
        score = np.mean(edge_scores) / 255.0
        return min(score, 1.0)
    
    def grid_periodicity_check(self, frame: np.ndarray) -> float:
        """
        FFmpeg blockdetect equivalent - detect periodic grid patterns
        at 8, 16, 32px intervals matching codec macroblock sizes.
        
        Returns: periodicity score [0, 1], higher = stronger grid pattern
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        h, w = gray.shape
        
        # Compute row and column variance profiles
        row_variance = np.var(gray, axis=1)
        col_variance = np.var(gray, axis=0)
        
        periodicity_scores = []
        
        for grid_size in self.grid_sizes:
            # Check if variance dips align with grid boundaries
            row_grid_alignment = self._check_grid_alignment(row_variance, grid_size)
            col_grid_alignment = self._check_grid_alignment(col_variance, grid_size)
            
            periodicity_scores.append((row_grid_alignment + col_grid_alignment) / 2)
        
        return max(periodicity_scores) if periodicity_scores else 0.0
    
    def _check_grid_alignment(self, variance_profile: np.ndarray, grid_size: int) -> float:
        """Check if variance dips align with grid boundaries"""
        n = len(variance_profile)
        grid_positions = list(range(0, n, grid_size))
        
        if len(grid_positions) < 2:
            return 0.0
        
        # Sample variance at grid boundaries vs. midpoints
        boundary_variance = []
        midpoint_variance = []
        
        for i in range(len(grid_positions) - 1):
            start = grid_positions[i]
            end = grid_positions[i + 1]
            mid = (start + end) // 2
            
            if start < n and end < n and mid < n:
                boundary_variance.append(variance_profile[start])
                boundary_variance.append(variance_profile[end])
                midpoint_variance.append(variance_profile[mid])
        
        if not boundary_variance or not midpoint_variance:
            return 0.0
        
        # Strong grid = low variance at boundaries, high at midpoints
        boundary_mean = np.mean(boundary_variance)
        midpoint_mean = np.mean(midpoint_variance)
        
        if midpoint_mean == 0:
            return 0.0
        
        ratio = 1.0 - (boundary_mean / (midpoint_mean + 1e-6))
        return max(0.0, min(ratio, 1.0))
    
    def color_quantization_check(self, frame: np.ndarray) -> float:
        """
        High saturation + low spatial variance = quantization artifact.
        Natural vivid scenes have high saturation but fine detail.
        
        Returns: quantization score [0, 1], higher = more quantization
        """
        if len(frame.shape) != 3:
            return 0.0
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        
        # Divide into blocks
        block_size = 16
        quant_scores = []
        
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = hsv[y:y+block_size, x:x+block_size]
                
                # High saturation check
                saturation = block[:, :, 1]
                mean_sat = np.mean(saturation)
                
                if mean_sat < 50:  # Skip low-saturation blocks
                    continue
                
                # Low spatial variance check
                hue = block[:, :, 0]
                hue_variance = np.var(hue)
                
                # High saturation + very low variance = quantization (not just natural flat color)
                # Threshold 20 is much tighter than 100 to avoid flagging natural scenes
                if hue_variance < 20:
                    quant_scores.append(mean_sat / 255.0)
        
        if not quant_scores:
            return 0.0
        
        return np.mean(quant_scores)
    
    def compute_tier1_signals(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Run all Tier 1 checks and return signals.
        If all below threshold, frame is clean - no ML needed.
        """
        edge_score = self.boundary_edge_pairing(frame)
        grid_score = self.grid_periodicity_check(frame)
        color_quant_score = self.color_quantization_check(frame)
        
        return {
            'edge_score': edge_score,
            'grid_score': grid_score,
            'color_quant_score': color_quant_score,
            'passes_gate': edge_score > self.edge_threshold or color_quant_score > self.color_quant_threshold
        }


class CompositeScorer:
    """Combine all signals into final composite score"""
    
    def __init__(self):
        self.weights = {
            'boundary_edge': settings.BOUNDARY_EDGE_WEIGHT,
            'mvad_blockiness': settings.MVAD_BLOCKINESS_WEIGHT,
            'color_quant': settings.COLOR_QUANT_WEIGHT,
            'mvad_pixelation': settings.MVAD_PIXELATION_WEIGHT,
            'brisque': settings.BRISQUE_WEIGHT
        }
    
    def compute_score(self, signals: Dict[str, float]) -> float:
        """
        Weighted composite: 35% edge + 30% MVAD block + 15% color + 10% MVAD pixel + 10% BRISQUE
        """
        score = 0.0
        
        score += signals.get('edge_score', 0.0) * self.weights['boundary_edge']
        score += signals.get('mvad_blockiness', 0.0) * self.weights['mvad_blockiness']
        score += signals.get('color_quant_score', 0.0) * self.weights['color_quant']
        score += signals.get('mvad_pixelation', 0.0) * self.weights['mvad_pixelation']
        score += signals.get('brisque_score', 0.0) * self.weights['brisque']
        
        return min(score, 1.0)
    
    def classify_artifact(self, signals: Dict[str, float]) -> str:
        """
        Determine artifact type based on dominant signal
        """
        edge_contribution = signals.get('edge_score', 0.0) * self.weights['boundary_edge']
        color_contribution = signals.get('color_quant_score', 0.0) * self.weights['color_quant']
        
        # Check for color shift (video only)
        if signals.get('color_shift_detected', False):
            return 'color_shift'
        
        # Dominant signal determines type
        if edge_contribution > color_contribution:
            return 'macroblocking'
        else:
            return 'pixelation'


def compute_brisque_score(frame: np.ndarray) -> float:
    """
    Gradient-ratio heuristic quality score, [0, 100].
    
    Higher = more degraded / more block-artifact energy at 8px boundaries.
    
    Weights: 
    - 0.75 blockiness (primary) - gradient energy at 8px boundaries vs interior
    - 0.25 sharpness penalty (secondary) - Laplacian variance
    
    The sharpness weight is intentionally low — block edges themselves create
    high Laplacian variance, so sharpness alone is not a reliable signal.
    
    Returns:
        Score in [0, 100] range, higher = more artifacts
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) if len(frame.shape) == 3 else frame.astype(np.float32)
    h, w = gray.shape

    # Compute gradient magnitude
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx ** 2 + sobely ** 2)

    # Correct BRISQUE: compare gradient at boundary lines vs 3px inside
    # Horizontal: compare rows at multiples of 8 vs rows 3px inside each block
    h_boundary_rows = [i for i in range(0, h, 8) if i < h]
    h_interior_rows = [i + 3 for i in range(0, h, 8) if i + 3 < h]

    # Vertical: compare cols at multiples of 8 vs cols 3px inside each block
    v_boundary_cols = [j for j in range(0, w, 8) if j < w]
    v_interior_cols = [j + 3 for j in range(0, w, 8) if j + 3 < w]

    # Horizontal blocking: strong gradients along horizontal boundary rows
    h_boundary_energy = float(gradient_mag[h_boundary_rows, :].mean()) if h_boundary_rows else 0.0
    h_interior_energy = float(gradient_mag[h_interior_rows, :].mean()) if h_interior_rows else 1e-6

    # Vertical blocking: strong gradients along vertical boundary cols
    v_boundary_energy = float(gradient_mag[:, v_boundary_cols].mean()) if v_boundary_cols else 0.0
    v_interior_energy = float(gradient_mag[:, v_interior_cols].mean()) if v_interior_cols else 1e-6

    # Combined blockiness ratio
    boundary_energy = (h_boundary_energy + v_boundary_energy) / 2
    interior_energy = (h_interior_energy + v_interior_energy) / 2
    blockiness_ratio = boundary_energy / (interior_energy + 1e-6)

    # Sharpness penalty using Laplacian variance
    lap_var = cv2.Laplacian(gray, cv2.CV_32F).var()
    sharpness_penalty = float(np.clip(100.0 - np.log1p(lap_var) * 8.0, 0.0, 100.0))

    # Blockiness score: normalize ratio to [0, 100]
    blockiness_score = float(np.clip((blockiness_ratio - 1.0) / 3.0 * 100.0, 0.0, 100.0))

    # Weighted combination: 75% blockiness + 25% sharpness
    final_score = float(np.clip(0.75 * blockiness_score + 0.25 * sharpness_penalty, 0.0, 100.0))

    return final_score
