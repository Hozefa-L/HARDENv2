from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from sklearn.metrics import average_precision_score


@dataclass(frozen=True)
class SplitMetrics:
    split: str
    multilabel_loss: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    micro_precision: float
    micro_recall: float
    micro_f1: float
    micro_average_precision: Optional[float]
    macro_average_precision: Optional[float]
    macro_accuracy: float
    macro_mcc: float
    subset_accuracy: float
    assessed_label_count: int
    assessed_positive_count: int
    per_swc: List[Dict[str, Any]]


def sigmoid(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    positive = x >= 0
    negative = ~positive
    out = np.empty_like(x, dtype=np.float64)
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[negative])
    out[negative] = exp_x / (1.0 + exp_x)
    return out


def _binary_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    truth = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    tp = int(np.logical_and(truth == 1, pred == 1).sum())
    fp = int(np.logical_and(truth == 0, pred == 1).sum())
    fn = int(np.logical_and(truth == 1, pred == 0).sum())
    tn = int(np.logical_and(truth == 0, pred == 0).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _prf_from_counts(counts: Mapping[str, int]) -> Dict[str, float]:
    tp = float(counts["tp"])
    fp = float(counts["fp"])
    fn = float(counts["fn"])
    tn = float(counts["tn"])
    precision = tp / (tp + fp) if (tp + fp) > 0.0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0.0 else 0.0
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0.0 else 0.0
    # Matthews Correlation Coefficient
    mcc_denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = (tp * tn - fp * fn) / mcc_denom if mcc_denom > 0.0 else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "mcc": float(mcc),
    }


def _average_precision_if_defined(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    labels = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(y_score, dtype=np.float64)
    if labels.size == 0:
        return None
    unique = np.unique(labels)
    if unique.size < 2:
        return None
    return float(average_precision_score(labels, scores))


def _masked_bce_from_logits(logits: np.ndarray, targets: np.ndarray, target_mask: np.ndarray) -> float:
    x = np.asarray(logits, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    mask = np.asarray(target_mask, dtype=bool)
    if x.shape != y.shape or x.shape != mask.shape:
        raise ValueError("`logits`, `targets`, and `target_mask` must have identical shapes.")
    if int(mask.sum()) == 0:
        raise ValueError("No assessed labels in `target_mask`; masked BCE is undefined.")

    # Stable BCE-with-logits formula:
    # max(x, 0) - x * y + log(1 + exp(-|x|))
    per_element = np.maximum(x, 0.0) - x * y + np.log1p(np.exp(-np.abs(x)))
    return float(per_element[mask].mean())


def _masked_bce_from_probabilities(probabilities: np.ndarray, targets: np.ndarray, target_mask: np.ndarray) -> float:
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    mask = np.asarray(target_mask, dtype=bool)
    if p.shape != y.shape or p.shape != mask.shape:
        raise ValueError("`probabilities`, `targets`, and `target_mask` must have identical shapes.")
    if int(mask.sum()) == 0:
        raise ValueError("No assessed labels in `target_mask`; masked BCE is undefined.")
    clipped = np.clip(p, 1e-8, 1.0 - 1e-8)
    per_element = -(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped))
    return float(per_element[mask].mean())


def optimize_per_swc_thresholds(
    *,
    probabilities: np.ndarray,
    targets: np.ndarray,
    target_mask: np.ndarray,
    swc_ids: Sequence[int],
    threshold_grid: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Find per-SWC decision threshold that maximises F1 on a validation set.

    Returns a dict with:
      - ``thresholds``: list of per-SWC optimal thresholds (same order as *swc_ids*)
      - ``per_swc``: list of dicts with swc_id, best_threshold, best_f1
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(targets, dtype=np.float64)
    mask = np.asarray(target_mask, dtype=bool)

    if probs.shape != truth.shape or probs.shape != mask.shape:
        raise ValueError("Shape mismatch between probabilities, targets, and target_mask.")
    if probs.ndim != 2 or probs.shape[1] != len(swc_ids):
        raise ValueError("SWC dimension mismatch.")

    if threshold_grid is None:
        threshold_grid = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                          0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

    optimal_thresholds: List[float] = []
    per_swc_info: List[Dict[str, Any]] = []

    for label_idx, swc_id in enumerate(swc_ids):
        label_mask = mask[:, label_idx]
        assessed_count = int(label_mask.sum())

        if assessed_count == 0:
            optimal_thresholds.append(0.5)
            per_swc_info.append({"swc_id": int(swc_id), "best_threshold": 0.5, "best_f1": 0.0, "assessed": 0})
            continue

        y_true = truth[label_mask, label_idx].astype(np.int64)
        y_score = probs[label_mask, label_idx]

        best_f1 = -1.0
        best_thr = 0.5

        for thr in threshold_grid:
            y_pred = (y_score >= thr).astype(np.int64)
            counts = _binary_counts(y_true, y_pred)
            prf = _prf_from_counts(counts)
            if prf["f1"] > best_f1:
                best_f1 = prf["f1"]
                best_thr = thr

        optimal_thresholds.append(best_thr)
        per_swc_info.append({
            "swc_id": int(swc_id),
            "best_threshold": best_thr,
            "best_f1": float(best_f1),
            "assessed": assessed_count,
        })

    return {
        "thresholds": optimal_thresholds,
        "per_swc": per_swc_info,
    }


def compute_masked_multilabel_metrics(
    *,
    probabilities: np.ndarray,
    targets: np.ndarray,
    target_mask: np.ndarray,
    swc_ids: Sequence[int],
    threshold: float = 0.5,
    per_swc_thresholds: Optional[Sequence[float]] = None,
    logits: Optional[np.ndarray] = None,
    split_name: str = "",
) -> Dict[str, Any]:
    """Compute masked multilabel metrics.

    If *per_swc_thresholds* is provided (list of floats, one per SWC), each
    label uses its own decision threshold.  Otherwise the scalar *threshold*
    is applied uniformly.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(targets, dtype=np.float64)
    mask = np.asarray(target_mask, dtype=bool)

    if probs.shape != truth.shape or probs.shape != mask.shape:
        raise ValueError("`probabilities`, `targets`, and `target_mask` must have identical shapes.")
    if probs.ndim != 2:
        raise ValueError(f"Expected rank-2 arrays, got shape {probs.shape}.")
    if not 0.0 < threshold < 1.0:
        raise ValueError("`threshold` must be in (0.0, 1.0).")
    if probs.shape[1] != len(swc_ids):
        raise ValueError(
            f"SWC dimension mismatch: probs has {probs.shape[1]} labels but swc_ids has {len(swc_ids)} entries."
        )
    if per_swc_thresholds is not None and len(per_swc_thresholds) != len(swc_ids):
        raise ValueError("`per_swc_thresholds` length must match `swc_ids` length.")

    assessed_total = int(mask.sum())
    if assessed_total == 0:
        raise ValueError("No assessed labels in `target_mask`; cannot compute masked metrics.")

    # Build per-label prediction matrix using per-SWC or scalar threshold
    if per_swc_thresholds is not None:
        thr_array = np.array(per_swc_thresholds, dtype=np.float64).reshape(1, -1)
        preds = (probs >= thr_array).astype(np.int64)
    else:
        preds = (probs >= float(threshold)).astype(np.int64)
    per_swc_rows: List[Dict[str, Any]] = []
    macro_precision_values: List[float] = []
    macro_recall_values: List[float] = []
    macro_f1_values: List[float] = []
    macro_accuracy_values: List[float] = []
    macro_mcc_values: List[float] = []
    ap_values: List[float] = []

    for label_idx, swc_id in enumerate(swc_ids):
        label_mask = mask[:, label_idx]
        assessed_count = int(label_mask.sum())
        if assessed_count == 0:
            row = {
                "swc_id": int(swc_id),
                "support_assessed": 0,
                "support_positive": 0,
                "support_negative": 0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "accuracy": 0.0,
                "mcc": 0.0,
                "average_precision": None,
                "average_precision_defined": False,
            }
            per_swc_rows.append(row)
            continue

        y_true = truth[label_mask, label_idx].astype(np.int64)
        y_pred = preds[label_mask, label_idx].astype(np.int64)
        y_score = probs[label_mask, label_idx].astype(np.float64)

        counts = _binary_counts(y_true, y_pred)
        prf = _prf_from_counts(counts)
        ap = _average_precision_if_defined(y_true, y_score)

        macro_precision_values.append(prf["precision"])
        macro_recall_values.append(prf["recall"])
        macro_f1_values.append(prf["f1"])
        macro_accuracy_values.append(prf["accuracy"])
        macro_mcc_values.append(prf["mcc"])
        if ap is not None:
            ap_values.append(float(ap))

        row = {
            "swc_id": int(swc_id),
            "support_assessed": assessed_count,
            "support_positive": int((y_true == 1).sum()),
            "support_negative": int((y_true == 0).sum()),
            "tp": int(counts["tp"]),
            "fp": int(counts["fp"]),
            "fn": int(counts["fn"]),
            "tn": int(counts["tn"]),
            "precision": float(prf["precision"]),
            "recall": float(prf["recall"]),
            "f1": float(prf["f1"]),
            "accuracy": float(prf["accuracy"]),
            "mcc": float(prf["mcc"]),
            "average_precision": float(ap) if ap is not None else None,
            "average_precision_defined": bool(ap is not None),
        }
        per_swc_rows.append(row)

    masked_truth = truth[mask].astype(np.int64)
    masked_pred = preds[mask].astype(np.int64)
    masked_scores = probs[mask].astype(np.float64)
    micro_counts = _binary_counts(masked_truth, masked_pred)
    micro_prf = _prf_from_counts(micro_counts)
    micro_ap = _average_precision_if_defined(masked_truth, masked_scores)

    if logits is not None:
        multilabel_loss = _masked_bce_from_logits(logits=np.asarray(logits, dtype=np.float64), targets=truth, target_mask=mask)
    else:
        multilabel_loss = _masked_bce_from_probabilities(probabilities=probs, targets=truth, target_mask=mask)

    macro_precision = float(np.mean(macro_precision_values)) if macro_precision_values else 0.0
    macro_recall = float(np.mean(macro_recall_values)) if macro_recall_values else 0.0
    macro_f1 = float(np.mean(macro_f1_values)) if macro_f1_values else 0.0
    macro_accuracy = float(np.mean(macro_accuracy_values)) if macro_accuracy_values else 0.0
    macro_mcc = float(np.mean(macro_mcc_values)) if macro_mcc_values else 0.0
    macro_ap = float(np.mean(ap_values)) if ap_values else None

    # Subset accuracy: fraction of samples where ALL assessed labels are correct
    n_samples = probs.shape[0]
    exact_matches = 0
    for i in range(n_samples):
        sample_mask = mask[i, :]
        if not sample_mask.any():
            continue
        if np.array_equal(preds[i, sample_mask], truth[i, sample_mask].astype(np.int64)):
            exact_matches += 1
    n_assessed_samples = int((mask.any(axis=1)).sum())
    subset_accuracy = float(exact_matches / n_assessed_samples) if n_assessed_samples > 0 else 0.0

    payload = SplitMetrics(
        split=str(split_name),
        multilabel_loss=float(multilabel_loss),
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        micro_precision=float(micro_prf["precision"]),
        micro_recall=float(micro_prf["recall"]),
        micro_f1=float(micro_prf["f1"]),
        micro_average_precision=float(micro_ap) if micro_ap is not None else None,
        macro_average_precision=macro_ap,
        macro_accuracy=macro_accuracy,
        macro_mcc=macro_mcc,
        subset_accuracy=subset_accuracy,
        assessed_label_count=int(assessed_total),
        assessed_positive_count=int((truth[mask] == 1).sum()),
        per_swc=per_swc_rows,
    )

    return {
        "split": payload.split,
        "multilabel_loss": payload.multilabel_loss,
        "macro_precision": payload.macro_precision,
        "macro_recall": payload.macro_recall,
        "macro_f1": payload.macro_f1,
        "micro_precision": payload.micro_precision,
        "micro_recall": payload.micro_recall,
        "micro_f1": payload.micro_f1,
        "micro_average_precision": payload.micro_average_precision,
        "macro_average_precision": payload.macro_average_precision,
        "macro_accuracy": payload.macro_accuracy,
        "macro_mcc": payload.macro_mcc,
        "subset_accuracy": payload.subset_accuracy,
        "assessed_label_count": payload.assessed_label_count,
        "assessed_positive_count": payload.assessed_positive_count,
        "per_swc": payload.per_swc,
    }
