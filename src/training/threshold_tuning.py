"""Per-SWC sigmoid threshold tuning on the validation set.

After training a multi-label model, sweeps the decision threshold
independently for each SWC label over a grid and selects the threshold
that maximises per-label F1 on the validation set.  The optimal
thresholds are saved to a JSON file for use at test time.

Usage::

    from src.training.threshold_tuning import tune_thresholds, apply_thresholds

    optimal = tune_thresholds(
        logits_val,    # Tensor [N, num_labels]
        targets_val,   # Tensor [N, num_labels]
        mask_val,      # Tensor [N, num_labels] bool
        swc_ids=[101, 103, 104, 107, 113, 114, 115, 120, 128, 135],
        output_path=Path("reports/phase7_balanced/optimal_thresholds.json"),
    )
    # optimal: Dict[str, float]  e.g. {"101": 0.35, "103": 0.50, ...}

    preds = apply_thresholds(logits_test, optimal, swc_ids)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Default sweep grid: 0.05 to 0.90 in steps of 0.05
DEFAULT_THRESHOLD_GRID = [round(v, 2) for v in np.arange(0.05, 0.91, 0.05).tolist()]


def _f1_for_threshold(
    probs: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    threshold: float,
) -> float:
    """Compute binary F1 for a single label given a threshold."""
    assessed = mask.astype(bool)
    if assessed.sum() == 0:
        return 0.0
    p = probs[assessed]
    t = targets[assessed]
    preds = (p >= threshold).astype(int)
    tp = int(((preds == 1) & (t == 1)).sum())
    fp = int(((preds == 1) & (t == 0)).sum())
    fn = int(((preds == 0) & (t == 1)).sum())
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return float(f1)


def tune_thresholds(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    swc_ids: Sequence[int],
    threshold_grid: Optional[List[float]] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, float]:
    """Find per-SWC optimal decision threshold on the validation set.

    Args:
        logits: Float tensor of shape [N, num_labels].
        targets: Float tensor of shape [N, num_labels] with 0/1 labels.
        mask: Bool tensor of shape [N, num_labels]; True = assessed.
        swc_ids: Ordered list matching the label dimension.
        threshold_grid: Candidate threshold values to sweep (default: 0.05…0.90).
        output_path: If provided, saves the result JSON here.

    Returns:
        Dict mapping str(swc_id) -> optimal_threshold (float).
    """
    if threshold_grid is None:
        threshold_grid = DEFAULT_THRESHOLD_GRID

    probs_np = torch.sigmoid(logits).detach().cpu().numpy()  # [N, L]
    targets_np = targets.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy().astype(bool)

    num_labels = probs_np.shape[1]
    if len(swc_ids) != num_labels:
        raise ValueError(
            f"len(swc_ids)={len(swc_ids)} != num_labels={num_labels}"
        )

    optimal: Dict[str, float] = {}
    tuning_details: List[Dict] = []

    for col_idx, swc_id in enumerate(swc_ids):
        p_col = probs_np[:, col_idx]
        t_col = targets_np[:, col_idx]
        m_col = mask_np[:, col_idx]

        best_thr = 0.5
        best_f1 = -1.0
        sweep_results = []

        for thr in threshold_grid:
            f1 = _f1_for_threshold(p_col, t_col, m_col, thr)
            sweep_results.append({"threshold": thr, "f1": round(f1, 4)})
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr

        optimal[str(swc_id)] = float(best_thr)
        n_pos = int(t_col[m_col.astype(bool)].sum()) if m_col.any() else 0
        tuning_details.append(
            {
                "swc_id": swc_id,
                "optimal_threshold": best_thr,
                "val_f1_at_optimal": round(best_f1, 4),
                "val_positives": n_pos,
                "sweep": sweep_results,
            }
        )
        logger.info(
            "SWC-%d: optimal_threshold=%.2f  val_F1=%.4f  (val_pos=%d)",
            swc_id, best_thr, best_f1, n_pos,
        )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "optimal_thresholds": optimal,
            "details": tuning_details,
            "threshold_grid": threshold_grid,
        }
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("Saved threshold tuning results to %s", output_path)

    return optimal


def apply_thresholds(
    logits: torch.Tensor,
    optimal_thresholds: Dict[str, float],
    swc_ids: Sequence[int],
) -> torch.Tensor:
    """Apply per-SWC optimal thresholds to produce binary predictions.

    Args:
        logits: Float tensor [N, num_labels].
        optimal_thresholds: Dict str(swc_id) -> threshold.
        swc_ids: Ordered list of SWC IDs.

    Returns:
        Binary predictions tensor [N, num_labels] (int, 0 or 1).
    """
    probs = torch.sigmoid(logits)  # [N, L]
    preds = torch.zeros_like(probs, dtype=torch.long)
    for col_idx, swc_id in enumerate(swc_ids):
        thr = float(optimal_thresholds.get(str(swc_id), 0.5))
        preds[:, col_idx] = (probs[:, col_idx] >= thr).long()
    return preds


def load_thresholds(path: Path) -> Dict[str, float]:
    """Load optimal thresholds from a previously saved JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Threshold file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return {str(k): float(v) for k, v in payload["optimal_thresholds"].items()}
