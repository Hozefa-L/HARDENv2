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
