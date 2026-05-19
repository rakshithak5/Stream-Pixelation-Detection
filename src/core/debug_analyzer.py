"""
Debug analyzer - full internal breakdown of image detection workflow.
Shows per-block scores, pixel-level analysis, and step-by-step trace.
"""
import cv2
import numpy as np
from typing import Dict, List
from src.core.detection import compute_brisque_score
from src.models.mvad_wrapper import model_manager
from src.core.config import settings


def analyze_image_debug(image: np.ndarray) -> Dict:
    """
    Full internal breakdown of the detection workflow.
    Returns per-block scores, pixel maps, and step-by-step trace.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    trace = []  # step-by-step log

    # ── STEP 1: Image properties ─────────────────────────────────────────────
    trace.append({
        'step': 1,
        'name': 'Image Properties',
        'details': {
            'width': w,
            'height': h,
            'channels': image.shape[2] if len(image.shape) == 3 else 1,
            'dtype': str(image.dtype),
            'mean_brightness': float(np.mean(gray)),
            'std_brightness':  float(np.std(gray)),
            'min_pixel': int(gray.min()),
            'max_pixel': int(gray.max()),
        }
    })

    # ── STEP 2: Gradient computation ─────────────────────────────────────────
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx ** 2 + sobely ** 2)

    trace.append({
        'step': 2,
        'name': 'Gradient Computation (Sobel)',
        'details': {
            'mean_gradient':   float(gradient_mag.mean()),
            'max_gradient':    float(gradient_mag.max()),
            'std_gradient':    float(gradient_mag.std()),
            'description': (
                'Sobel operator computes gradient magnitude at every pixel. '
                'High gradient = strong edge. Macroblocking creates '
                'artificially strong edges at 8/16px boundaries.'
            )
        }
    })

    # ── STEP 3: Boundary edge pairing (per block boundary) ───────────────────
    block_boundary_scores = {}
    edge_scores_list = []

    for block_size in [8, 16]:
        boundary_strengths = []

        # Vertical boundaries
        for x in range(block_size, w - block_size, block_size):
            left  = np.abs(sobelx[:, x - 1])
            right = np.abs(sobelx[:, x])
            paired = float(np.minimum(left, right).mean())
            boundary_strengths.append({'x': x, 'direction': 'vertical', 'strength': round(paired, 4)})
            edge_scores_list.append(paired)

        # Horizontal boundaries
        for y in range(block_size, h - block_size, block_size):
            top    = np.abs(sobely[y - 1, :])
            bottom = np.abs(sobely[y, :])
            paired = float(np.minimum(top, bottom).mean())
            boundary_strengths.append({'y': y, 'direction': 'horizontal', 'strength': round(paired, 4)})
            edge_scores_list.append(paired)

        block_boundary_scores[f'block_{block_size}px'] = {
            'boundary_count': len(boundary_strengths),
            'mean_strength':  round(float(np.mean([b['strength'] for b in boundary_strengths])), 4) if boundary_strengths else 0,
            'max_strength':   round(float(np.max([b['strength'] for b in boundary_strengths])), 4) if boundary_strengths else 0,
            'top5_strongest': sorted(boundary_strengths, key=lambda b: b['strength'], reverse=True)[:5],
        }

    edge_score = float(np.mean(edge_scores_list) / 255.0) if edge_scores_list else 0.0
    edge_score = min(edge_score, 1.0)

    trace.append({
        'step': 3,
        'name': 'Boundary Edge Pairing',
        'score': round(edge_score, 4),
        'threshold': settings.EDGE_VOTE_THRESHOLD,
        'vote': 'YES' if edge_score > settings.EDGE_VOTE_THRESHOLD else 'NO',
        'details': block_boundary_scores,
        'description': (
            'Checks BOTH sides of each macroblock boundary. '
            'Natural edge: one side strong. '
            'Macroblock artifact: BOTH sides strong and aligned. '
            'Score = mean paired strength / 255.'
        )
    })

    # ── STEP 4: Grid periodicity (per grid size) ─────────────────────────────
    row_variance = np.var(gray, axis=1)
    col_variance = np.var(gray, axis=0)
    grid_detail  = {}
    grid_scores  = []

    for grid_size in [8, 16, 32]:
        # Row alignment
        row_boundary, row_midpoint = [], []
        positions = list(range(0, h, grid_size))
        for i in range(len(positions) - 1):
            s, e = positions[i], positions[i + 1]
            mid = (s + e) // 2
            if s < h and e < h and mid < h:
                row_boundary += [float(row_variance[s]), float(row_variance[e])]
                row_midpoint.append(float(row_variance[mid]))

        # Col alignment
        col_boundary, col_midpoint = [], []
        positions = list(range(0, w, grid_size))
        for i in range(len(positions) - 1):
            s, e = positions[i], positions[i + 1]
            mid = (s + e) // 2
            if s < w and e < w and mid < w:
                col_boundary += [float(col_variance[s]), float(col_variance[e])]
                col_midpoint.append(float(col_variance[mid]))

        row_score = max(0.0, 1.0 - (np.mean(row_boundary) / (np.mean(row_midpoint) + 1e-6))) if row_boundary and row_midpoint else 0.0
        col_score = max(0.0, 1.0 - (np.mean(col_boundary) / (np.mean(col_midpoint) + 1e-6))) if col_boundary and col_midpoint else 0.0
        combined  = (row_score + col_score) / 2

        grid_detail[f'grid_{grid_size}px'] = {
            'row_alignment_score': round(row_score, 4),
            'col_alignment_score': round(col_score, 4),
            'combined_score':      round(combined, 4),
            'interpretation': (
                'High score = variance DIPS at grid boundaries (flat blocks). '
                'Low score = variance uniform (no grid pattern).'
            )
        }
        grid_scores.append(combined)

    grid_score = float(max(grid_scores)) if grid_scores else 0.0

    trace.append({
        'step': 4,
        'name': 'Grid Periodicity Check',
        'score': round(grid_score, 4),
        'threshold': settings.GRID_VOTE_THRESHOLD,
        'vote': 'YES' if grid_score > settings.GRID_VOTE_THRESHOLD else 'NO',
        'details': grid_detail,
        'description': (
            'Detects repeating patterns at 8/16/32px intervals. '
            'Macroblocking creates flat blocks → variance dips at boundaries. '
            'Score = 1 - (boundary_variance / midpoint_variance).'
        )
    })

    # ── STEP 5: Color quantization (per block) ───────────────────────────────
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    block_size = 16
    quant_blocks = []
    quant_scores_list = []

    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = hsv[y:y + block_size, x:x + block_size]
            sat   = block[:, :, 1]
            hue   = block[:, :, 0]
            mean_sat   = float(np.mean(sat))
            hue_var    = float(np.var(hue))
            is_artifact = mean_sat > 50 and hue_var < 100

            if mean_sat > 50:
                quant_blocks.append({
                    'block_x': x, 'block_y': y,
                    'mean_saturation': round(mean_sat, 2),
                    'hue_variance':    round(hue_var, 2),
                    'flagged':         is_artifact,
                })
                if is_artifact:
                    quant_scores_list.append(mean_sat / 255.0)

    color_score = float(np.mean(quant_scores_list)) if quant_scores_list else 0.0
    flagged_blocks = [b for b in quant_blocks if b['flagged']]

    trace.append({
        'step': 5,
        'name': 'Color Quantization Check',
        'score': round(color_score, 4),
        'details': {
            'total_saturated_blocks': len(quant_blocks),
            'flagged_blocks_count':   len(flagged_blocks),
            'flagged_percentage':     round(len(flagged_blocks) / max(len(quant_blocks), 1) * 100, 1),
            'top5_flagged_blocks':    sorted(flagged_blocks, key=lambda b: b['mean_saturation'], reverse=True)[:5],
            'sample_clean_blocks':    [b for b in quant_blocks if not b['flagged']][:3],
        },
        'description': (
            'Detects color banding: high saturation + low hue variance = flat color block. '
            'Natural scenes have high saturation BUT varied hue. '
            'Compressed artifacts have uniform hue within a block.'
        )
    })

    # ── STEP 6: BRISQUE gradient-ratio ───────────────────────────────────────
    # Correct: compare full boundary rows/cols vs interior rows/cols (3px inside)
    h_boundary_rows = [i for i in range(0, h, 8) if i < h]
    h_interior_rows = [i + 3 for i in range(0, h, 8) if i + 3 < h]
    v_boundary_cols = [j for j in range(0, w, 8) if j < w]
    v_interior_cols = [j + 3 for j in range(0, w, 8) if j + 3 < w]

    h_b_energy = float(gradient_mag[h_boundary_rows, :].mean()) if h_boundary_rows else 0.0
    h_i_energy = float(gradient_mag[h_interior_rows, :].mean()) if h_interior_rows else 1e-6
    v_b_energy = float(gradient_mag[:, v_boundary_cols].mean()) if v_boundary_cols else 0.0
    v_i_energy = float(gradient_mag[:, v_interior_cols].mean()) if v_interior_cols else 1e-6

    boundary_energy  = (h_b_energy + v_b_energy) / 2
    interior_energy  = (h_i_energy + v_i_energy) / 2
    blockiness_ratio = boundary_energy / (interior_energy + 1e-6)

    lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    sharpness_penalty = float(np.clip(100.0 - np.log1p(lap_var) * 8.0, 0.0, 100.0))
    blockiness_score  = float(np.clip((blockiness_ratio - 1.0) / 3.0 * 100.0, 0.0, 100.0))
    brisque_score     = float(np.clip(0.75 * blockiness_score + 0.25 * sharpness_penalty, 0.0, 100.0))

    trace.append({
        'step': 6,
        'name': 'BRISQUE Gradient-Ratio',
        'score': round(brisque_score, 2),
        'threshold': settings.BRISQUE_VOTE_THRESHOLD,
        'vote': 'YES' if brisque_score > settings.BRISQUE_VOTE_THRESHOLD else 'NO',
        'details': {
            'h_boundary_rows':   len(h_boundary_rows),
            'h_interior_rows':   len(h_interior_rows),
            'v_boundary_cols':   len(v_boundary_cols),
            'v_interior_cols':   len(v_interior_cols),
            'h_boundary_energy': round(h_b_energy, 4),
            'h_interior_energy': round(h_i_energy, 4),
            'v_boundary_energy': round(v_b_energy, 4),
            'v_interior_energy': round(v_i_energy, 4),
            'boundary_gradient_energy': round(boundary_energy, 4),
            'interior_gradient_energy': round(interior_energy, 4),
            'blockiness_ratio':      round(blockiness_ratio, 4),
            'blockiness_score_0_100': round(blockiness_score, 2),
            'laplacian_variance':    round(lap_var, 2),
            'sharpness_penalty':     round(sharpness_penalty, 2),
            'formula': '0.75 * blockiness_score + 0.25 * sharpness_penalty',
            'interpretation': (
                f'Ratio={blockiness_ratio:.3f}. '
                'Clean image: ratio ≈ 1.0 (boundaries same as interior). '
                'Blocked image: ratio > 2.0 (boundaries much stronger). '
                f'Your image: {"BLOCKED" if blockiness_ratio > 1.5 else "CLEAN"} '
                f'(ratio={blockiness_ratio:.2f})'
            )
        },
        'description': (
            'Compares gradient energy AT 8px boundaries vs interior pixels. '
            'Macroblocking creates strong edges exactly at 8px grid lines. '
            'Score 0-100: higher = more blocking.'
        )
    })

    # ── STEP 7: MVAD ML model ────────────────────────────────────────────────
    mvad_blockiness, mvad_pixelation = model_manager.predict(image)
    mvad_score = max(mvad_blockiness, mvad_pixelation)

    trace.append({
        'step': 7,
        'name': 'MVAD ML Model (RMViT)',
        'score': round(mvad_score, 4),
        'threshold': settings.MVAD_VOTE_THRESHOLD,
        'vote': 'YES' if mvad_score > settings.MVAD_VOTE_THRESHOLD else 'NO',
        'details': {
            'mvad_blockiness':  round(mvad_blockiness, 4),
            'mvad_pixelation':  round(mvad_pixelation, 4),
            'dominant_artifact': 'blockiness' if mvad_blockiness >= mvad_pixelation else 'pixelation',
            'input_size':       '224x224 (4 frames, ImageNet normalized)',
            'model_architecture': 'RMViT (Region-aware Multi-scale Vision Transformer)',
            'trained_on':       '10 artifact types: blockiness, pixelation, banding, blur, etc.',
        },
        'description': (
            'Deep learning model processes 4 frames at 224x224. '
            'Detects subtle patterns invisible to traditional methods. '
            'Most reliable signal for subtle macroblocking.'
        )
    })

    # ── STEP 8: Tier 1 composite ─────────────────────────────────────────────
    tier1_score = edge_score * 0.50 + color_score * 0.40 + grid_score * 0.10

    trace.append({
        'step': 8,
        'name': 'Tier 1 Composite Score',
        'score': round(tier1_score, 4),
        'threshold': 0.50,
        'vote': 'YES' if tier1_score > 0.50 else 'NO',
        'details': {
            'edge_contribution':  round(edge_score * 0.50, 4),
            'color_contribution': round(color_score * 0.40, 4),
            'grid_contribution':  round(grid_score * 0.10, 4),
            'formula': 'edge*0.50 + color*0.40 + grid*0.10',
            'why_these_weights': (
                'Edge gets 50% - most reliable spatial signal for blocking. '
                'Color gets 40% - catches quantization/banding. '
                'Grid gets 10% - supplementary periodicity check.'
            )
        },
        'description': 'Weighted combination of spatial signals. Triggers if > 0.50.'
    })

    # ── STEP 9: Final hybrid decision ────────────────────────────────────────
    if mvad_score > settings.MVAD_VOTE_THRESHOLD:
        artifact_detected = True
        confidence        = mvad_score
        decision_maker    = 'MVAD'
        decision_reason   = f'MVAD score {mvad_score:.3f} > threshold {settings.MVAD_VOTE_THRESHOLD}'
    elif brisque_score > settings.BRISQUE_VOTE_THRESHOLD:
        artifact_detected = True
        confidence        = brisque_score / 100.0
        decision_maker    = 'BRISQUE'
        decision_reason   = f'BRISQUE score {brisque_score:.1f} > threshold {settings.BRISQUE_VOTE_THRESHOLD}'
    elif tier1_score > 0.50:
        artifact_detected = True
        confidence        = tier1_score
        decision_maker    = 'Tier1'
        decision_reason   = f'Tier1 score {tier1_score:.3f} > threshold 0.50'
    else:
        confidence        = mvad_score * 0.60 + tier1_score * 0.40
        artifact_detected = confidence > 0.30
        decision_maker    = 'Hybrid'
        decision_reason   = (
            f'All below thresholds. '
            f'Weighted: {mvad_score:.3f}*0.60 + {tier1_score:.3f}*0.40 = {confidence:.3f} '
            f'({">" if artifact_detected else "<"} 0.30)'
        )

    severity = 'none'
    if artifact_detected:
        severity = 'high' if confidence > 0.7 else ('medium' if confidence > 0.5 else 'low')

    artifact_type = None
    if artifact_detected:
        artifact_type = 'pixelation' if mvad_pixelation > mvad_blockiness else 'macroblocking'

    trace.append({
        'step': 9,
        'name': 'Final Hybrid Decision',
        'artifact_detected': artifact_detected,
        'confidence':        round(confidence, 4),
        'artifact_type':     artifact_type,
        'severity':          severity,
        'decision_maker':    decision_maker,
        'decision_reason':   decision_reason,
        'decision_tree': {
            'check_1': f'MVAD ({mvad_score:.3f}) > {settings.MVAD_VOTE_THRESHOLD}? {"✅ YES → ARTIFACT" if mvad_score > settings.MVAD_VOTE_THRESHOLD else "❌ NO → next check"}',
            'check_2': f'BRISQUE ({brisque_score:.1f}) > {settings.BRISQUE_VOTE_THRESHOLD}? {"✅ YES → ARTIFACT" if brisque_score > settings.BRISQUE_VOTE_THRESHOLD else "❌ NO → next check"}',
            'check_3': f'Tier1 ({tier1_score:.3f}) > 0.50? {"✅ YES → ARTIFACT" if tier1_score > 0.50 else "❌ NO → next check"}',
            'check_4': f'Hybrid ({confidence:.3f}) > 0.30? {"✅ YES → ARTIFACT" if artifact_detected and decision_maker == "Hybrid" else "❌ NO → CLEAN"}',
        }
    })

    # ── STEP 10: Per-block artifact map ──────────────────────────────────────
    block_map = _compute_block_map(gray, gradient_mag, hsv, h, w)

    trace.append({
        'step': 10,
        'name': 'Per-Block Artifact Map (8x8 grid)',
        'details': {
            'block_size': 8,
            'grid_rows':  h // 8,
            'grid_cols':  w // 8,
            'total_blocks': (h // 8) * (w // 8),
            'flagged_blocks': sum(1 for row in block_map for b in row if b['flagged']),
            'flagged_percentage': round(
                sum(1 for row in block_map for b in row if b['flagged']) /
                max((h // 8) * (w // 8), 1) * 100, 1
            ),
            'top10_worst_blocks': sorted(
                [b for row in block_map for b in row],
                key=lambda b: b['artifact_score'], reverse=True
            )[:10],
            'block_map_summary': _summarize_block_map(block_map),
        },
        'description': (
            'Each 8x8 block scored independently. '
            'artifact_score = boundary gradient energy / interior gradient energy. '
            'Score > 1.5 = likely artifact block.'
        )
    })

    return {
        'image_size': {'width': w, 'height': h},
        'final_result': {
            'artifact_detected': artifact_detected,
            'confidence':        round(confidence, 4),
            'artifact_type':     artifact_type,
            'severity':          severity,
            'decision_maker':    decision_maker,
        },
        'all_scores': {
            'edge_score':         round(edge_score, 4),
            'grid_score':         round(grid_score, 4),
            'color_quant_score':  round(color_score, 4),
            'tier1_composite':    round(tier1_score, 4),
            'mvad_blockiness':    round(mvad_blockiness, 4),
            'mvad_pixelation':    round(mvad_pixelation, 4),
            'brisque':            round(brisque_score, 2),
        },
        'thresholds': {
            'mvad':    settings.MVAD_VOTE_THRESHOLD,
            'brisque': settings.BRISQUE_VOTE_THRESHOLD,
            'edge':    settings.EDGE_VOTE_THRESHOLD,
            'grid':    settings.GRID_VOTE_THRESHOLD,
            'tier1':   0.50,
            'hybrid':  0.30,
        },
        'workflow_trace': trace,
    }


def _compute_block_map(gray, gradient_mag, hsv, h, w, block_size=8):
    """Score every 8x8 block independently."""
    block_map = []
    for by in range(0, h - block_size, block_size):
        row = []
        for bx in range(0, w - block_size, block_size):
            block_grad = gradient_mag[by:by + block_size, bx:bx + block_size]

            # Boundary = first/last row and col of this block
            boundary = np.zeros((block_size, block_size), dtype=bool)
            boundary[0, :] = True
            boundary[-1, :] = True
            boundary[:, 0] = True
            boundary[:, -1] = True
            interior = ~boundary

            b_energy = float(block_grad[boundary].mean()) if boundary.any() else 0.0
            i_energy = float(block_grad[interior].mean()) if interior.any() else 0.0

            # Guard against near-zero interior (uniform/flat blocks)
            # A flat block is not an artifact - skip it
            if i_energy < 0.5:
                ratio = 1.0  # treat as clean
            else:
                ratio = b_energy / i_energy

            # Color info
            block_hsv = hsv[by:by + block_size, bx:bx + block_size]
            mean_sat  = float(block_hsv[:, :, 1].mean())
            hue_var   = float(block_hsv[:, :, 0].var())

            artifact_score = round(ratio, 3)
            row.append({
                'bx': bx, 'by': by,
                'artifact_score':    artifact_score,
                'boundary_energy':   round(b_energy, 3),
                'interior_energy':   round(i_energy, 3),
                'mean_saturation':   round(mean_sat, 1),
                'hue_variance':      round(hue_var, 1),
                'flagged':           ratio > 1.5,
            })
        block_map.append(row)
    return block_map


def _summarize_block_map(block_map):
    """Return a compact row-by-row summary (max score per row)."""
    summary = []
    for i, row in enumerate(block_map):
        scores = [b['artifact_score'] for b in row]
        summary.append({
            'row': i,
            'max_score':  round(max(scores), 3),
            'mean_score': round(float(np.mean(scores)), 3),
            'flagged_in_row': sum(1 for b in row if b['flagged']),
        })
    return summary
