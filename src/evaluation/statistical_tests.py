"""Statistical significance tests for experiment comparison.

Provides Wilcoxon signed-rank tests for paired model comparisons,
Bonferroni correction for multiple testing, and Cohen's d effect size.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def _threshold_predictions(
    *,
    probabilities: np.ndarray,
    threshold: float = 0.5,
    per_swc_thresholds: Optional[Sequence[float]] = None,
) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim < 2:
        raise ValueError("Expected probability array with at least 2 dimensions.")
    if per_swc_thresholds is not None:
        thresholds = np.asarray(per_swc_thresholds, dtype=np.float64)
        if thresholds.shape != (probs.shape[-1],):
            raise ValueError("`per_swc_thresholds` length must match the label dimension.")
        reshape_dims = (1,) * (probs.ndim - 1) + (thresholds.shape[0],)
        return probs >= thresholds.reshape(reshape_dims)
    if not 0.0 < threshold < 1.0:
        raise ValueError("`threshold` must be in (0.0, 1.0).")
    return probs >= float(threshold)


def _masked_macro_f1_from_predictions(
    *,
    predictions: np.ndarray,
    targets: np.ndarray,
    target_mask: np.ndarray,
) -> np.ndarray:
    preds = np.asarray(predictions, dtype=bool)
    truth = np.asarray(targets, dtype=bool)
    mask = np.asarray(target_mask, dtype=bool)
    if truth.shape != mask.shape:
        raise ValueError("`targets` and `target_mask` must have identical shapes.")
    if preds.shape == truth.shape:
        truth_expanded = truth
        mask_expanded = mask
    elif preds.shape[-2:] == truth.shape:
        leading_dims = preds.ndim - 2
        expand_shape = (1,) * leading_dims + truth.shape
        truth_expanded = truth.reshape(expand_shape)
        mask_expanded = mask.reshape(expand_shape)
    else:
        raise ValueError("Prediction trailing dimensions must match target shape.")

    tp = np.sum(preds & truth_expanded & mask_expanded, axis=-2, dtype=np.int64)
    fp = np.sum(preds & ~truth_expanded & mask_expanded, axis=-2, dtype=np.int64)
    fn = np.sum(~preds & truth_expanded & mask_expanded, axis=-2, dtype=np.int64)

    numer = 2.0 * tp.astype(np.float64)
    denom = numer + fp.astype(np.float64) + fn.astype(np.float64)
    f1 = np.divide(numer, denom, out=np.zeros_like(numer, dtype=np.float64), where=denom > 0.0)
    return np.mean(f1, axis=-1, dtype=np.float64)


def wilcoxon_signed_rank_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    alternative: str = "two-sided",
) -> Dict[str, Any]:
    """Wilcoxon signed-rank test for paired samples.

    Args:
        scores_a: Metric values for model A (one per seed/fold).
        scores_b: Metric values for model B (one per seed/fold).
        alternative: 'two-sided', 'greater', or 'less'.

    Returns:
        Dict with statistic, p_value, n_pairs, and interpretation.
    """
    if not HAS_SCIPY:
        raise ImportError("scipy is required for statistical tests. Install with: pip install scipy")
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("scores_a and scores_b must be 1D arrays of equal length.")
    if len(a) < 3:
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "n_pairs": int(len(a)),
            "significant_005": False,
            "note": "Too few samples for Wilcoxon test (need >= 3).",
        }
    # Remove ties (differences == 0)
    diff = a - b
    nonzero = np.abs(diff) > 1e-12
    if nonzero.sum() < 3:
        return {
            "statistic": float("nan"),
            "p_value": 1.0,
            "n_pairs": int(len(a)),
            "significant_005": False,
            "note": "All pairs are tied or fewer than 3 non-tied pairs.",
        }
    result = scipy_stats.wilcoxon(a, b, alternative=alternative, zero_method="wilcox")
    p_value = float(result.pvalue)
    return {
        "statistic": float(result.statistic),
        "p_value": p_value,
        "n_pairs": int(len(a)),
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
    }


def bonferroni_correction(
    p_values: Sequence[float],
    num_comparisons: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Apply Bonferroni correction to a list of p-values.

    Args:
        p_values: Raw p-values from individual tests.
        num_comparisons: Total number of comparisons (defaults to len(p_values)).

    Returns:
        List of dicts with original_p, corrected_p, significant_005.
    """
    n = num_comparisons if num_comparisons is not None else len(p_values)
    results = []
    for p in p_values:
        corrected = min(float(p) * n, 1.0)
        results.append({
            "original_p": float(p),
            "corrected_p": corrected,
            "num_comparisons": n,
            "significant_005": corrected < 0.05,
            "significant_001": corrected < 0.01,
        })
    return results


def cohens_d(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
) -> Dict[str, Any]:
    """Compute Cohen's d effect size for paired samples.

    Uses the pooled standard deviation as denominator.
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    diff_mean = float(np.mean(a) - np.mean(b))
    pooled_std = float(np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0))
    if pooled_std < 1e-12:
        d = 0.0 if abs(diff_mean) < 1e-12 else float("inf") * np.sign(diff_mean)
    else:
        d = diff_mean / pooled_std

    # Interpret magnitude
    abs_d = abs(d)
    if abs_d < 0.2:
        magnitude = "negligible"
    elif abs_d < 0.5:
        magnitude = "small"
    elif abs_d < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"

    return {
        "d": float(d),
        "magnitude": magnitude,
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "std_a": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
        "std_b": float(np.std(b, ddof=1)) if len(b) > 1 else 0.0,
    }


def pairwise_significance_table(
    model_scores: Dict[str, List[float]],
    reference_model: Optional[str] = None,
    metric_name: str = "macro_f1",
) -> Dict[str, Any]:
    """Generate pairwise significance table comparing models.

    Args:
        model_scores: Dict mapping model_name -> list of metric values (one per seed).
        reference_model: If provided, only compare reference vs others.
            If None, compare all pairs.
        metric_name: Name of the metric being compared (for labeling).

    Returns:
        Dict with comparisons list and summary.
    """
    model_names = sorted(model_scores.keys())
    comparisons: List[Dict[str, Any]] = []

    if reference_model is not None:
        if reference_model not in model_scores:
            raise ValueError(f"Reference model '{reference_model}' not in model_scores.")
        ref_scores = model_scores[reference_model]
        other_models = [m for m in model_names if m != reference_model]
        pairs = [(reference_model, other) for other in other_models]
    else:
        pairs = [(a, b) for i, a in enumerate(model_names) for b in model_names[i + 1:]]

    raw_p_values: List[float] = []
    for model_a, model_b in pairs:
        wsr = wilcoxon_signed_rank_test(model_scores[model_a], model_scores[model_b])
        cd = cohens_d(model_scores[model_a], model_scores[model_b])
        comparisons.append({
            "model_a": model_a,
            "model_b": model_b,
            "metric": metric_name,
            "wilcoxon": wsr,
            "cohens_d": cd,
        })
        raw_p_values.append(wsr["p_value"])

    # Apply Bonferroni correction
    corrected = bonferroni_correction(raw_p_values, num_comparisons=len(pairs))
    for comp, corr in zip(comparisons, corrected):
        comp["bonferroni"] = corr

    return {
        "metric": metric_name,
        "num_models": len(model_names),
        "num_comparisons": len(pairs),
        "reference_model": reference_model,
        "comparisons": comparisons,
    }


def masked_macro_f1_from_probabilities(
    *,
    probabilities: np.ndarray,
    targets: np.ndarray,
    target_mask: np.ndarray,
    swc_ids: Sequence[int],
    threshold: float = 0.5,
    per_swc_thresholds: Optional[Sequence[float]] = None,
) -> float:
    probs = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(targets, dtype=np.float64)
    mask = np.asarray(target_mask, dtype=bool)
    if probs.shape != truth.shape or probs.shape != mask.shape:
        raise ValueError("All probability/target/mask arrays must have identical shapes.")
    if probs.ndim != 2:
        raise ValueError("Expected rank-2 probability arrays.")
    if probs.shape[1] != len(swc_ids):
        raise ValueError("SWC dimension mismatch.")

    preds = _threshold_predictions(
        probabilities=probs,
        threshold=threshold,
        per_swc_thresholds=per_swc_thresholds,
    )
    return float(
        _masked_macro_f1_from_predictions(
            predictions=preds,
            targets=truth,
            target_mask=mask,
        )
    )


def paired_bootstrap_macro_f1_difference(
    *,
    probabilities_a: np.ndarray,
    probabilities_b: np.ndarray,
    targets: np.ndarray,
    target_mask: np.ndarray,
    swc_ids: Sequence[int],
    per_swc_thresholds_a: Optional[Sequence[float]] = None,
    per_swc_thresholds_b: Optional[Sequence[float]] = None,
    n_bootstrap: int = 5000,
    ci_alpha: float = 0.05,
    random_state: int = 0,
) -> Dict[str, Any]:
    a = np.asarray(probabilities_a, dtype=np.float64)
    b = np.asarray(probabilities_b, dtype=np.float64)
    truth = np.asarray(targets, dtype=np.float64)
    mask = np.asarray(target_mask, dtype=bool)
    if a.shape != b.shape or a.shape != truth.shape or a.shape != mask.shape:
        raise ValueError("All probability/target/mask arrays must have identical shapes.")
    if a.ndim != 2:
        raise ValueError("Expected rank-2 probability arrays.")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive.")
    if not 0.0 < ci_alpha < 1.0:
        raise ValueError("ci_alpha must be in (0, 1).")

    preds_a = _threshold_predictions(
        probabilities=a,
        per_swc_thresholds=per_swc_thresholds_a,
    )
    preds_b = _threshold_predictions(
        probabilities=b,
        per_swc_thresholds=per_swc_thresholds_b,
    )

    observed = float(
        _masked_macro_f1_from_predictions(
            predictions=preds_a,
            targets=truth,
            target_mask=mask,
        )
        - _masked_macro_f1_from_predictions(
            predictions=preds_b,
            targets=truth,
            target_mask=mask,
        )
    )

    rng = np.random.default_rng(random_state)
    sample_idx = rng.integers(0, a.shape[0], size=(n_bootstrap, a.shape[0]), dtype=np.int64)
    deltas = _masked_macro_f1_from_predictions(
        predictions=preds_a[sample_idx],
        targets=truth[sample_idx],
        target_mask=mask[sample_idx],
    ) - _masked_macro_f1_from_predictions(
        predictions=preds_b[sample_idx],
        targets=truth[sample_idx],
        target_mask=mask[sample_idx],
    )

    lower = float(np.quantile(deltas, ci_alpha / 2.0))
    upper = float(np.quantile(deltas, 1.0 - ci_alpha / 2.0))
    return {
        "observed_difference": float(observed),
        "ci_alpha": float(ci_alpha),
        "ci_lower": lower,
        "ci_upper": upper,
        "n_bootstrap": int(n_bootstrap),
    }


def paired_permutation_macro_f1_test(
    *,
    probabilities_a: np.ndarray,
    probabilities_b: np.ndarray,
    targets: np.ndarray,
    target_mask: np.ndarray,
    swc_ids: Sequence[int],
    per_swc_thresholds_a: Optional[Sequence[float]] = None,
    per_swc_thresholds_b: Optional[Sequence[float]] = None,
    n_permutations: int = 5000,
    random_state: int = 0,
) -> Dict[str, Any]:
    a = np.asarray(probabilities_a, dtype=np.float64)
    b = np.asarray(probabilities_b, dtype=np.float64)
    truth = np.asarray(targets, dtype=np.float64)
    mask = np.asarray(target_mask, dtype=bool)
    if a.shape != b.shape or a.shape != truth.shape or a.shape != mask.shape:
        raise ValueError("All probability/target/mask arrays must have identical shapes.")
    if a.ndim != 2:
        raise ValueError("Expected rank-2 probability arrays.")
    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive.")

    preds_a = _threshold_predictions(
        probabilities=a,
        per_swc_thresholds=per_swc_thresholds_a,
    )
    preds_b = _threshold_predictions(
        probabilities=b,
        per_swc_thresholds=per_swc_thresholds_b,
    )

    observed = float(
        _masked_macro_f1_from_predictions(
            predictions=preds_a,
            targets=truth,
            target_mask=mask,
        )
        - _masked_macro_f1_from_predictions(
            predictions=preds_b,
            targets=truth,
            target_mask=mask,
        )
    )

    rng = np.random.default_rng(random_state)
    swap_mask = rng.random((n_permutations, a.shape[0], 1)) < 0.5
    perm_a = np.where(swap_mask, preds_b[None, :, :], preds_a[None, :, :])
    perm_b = np.where(swap_mask, preds_a[None, :, :], preds_b[None, :, :])
    deltas = _masked_macro_f1_from_predictions(
        predictions=perm_a,
        targets=truth,
        target_mask=mask,
    ) - _masked_macro_f1_from_predictions(
        predictions=perm_b,
        targets=truth,
        target_mask=mask,
    )

    extreme = int(np.count_nonzero(np.abs(deltas) >= abs(observed)))
    p_value = float((extreme + 1) / (n_permutations + 1))
    return {
        "observed_difference": float(observed),
        "p_value": p_value,
        "n_permutations": int(n_permutations),
        "significant_005": p_value < 0.05,
    }


def save_significance_report(
    report: Dict[str, Any],
    output_path: Path,
) -> None:
    """Save significance report to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert numpy types for JSON serialization
    def _convert(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    class _Encoder(json.JSONEncoder):
        def default(self, o: Any) -> Any:
            return _convert(o)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, cls=_Encoder)
