"""
5-fold stratified cross-validation for Level 0.5 enriched classical baselines.

Generates mean±std per-SWC F1 for XGBoost, Random Forest, and Logistic Regression
on the full enriched feature set (graph + TF-IDF + pattern features).

Usage:
    python -m src.evaluation.cross_validation [--n-folds 5] [--seed 42] [--output-dir reports/phase6_level0.5]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FEATURE_INDEX_PATH = Path("data/features/main_benchmark/phase3_feature_index.parquet")
GRAPH_FEATURES_PATH = Path("data/features/main_benchmark/graph_level_features.parquet")
TFIDF_FEATURES_PATH = Path("data/features/main_benchmark/tfidf_features.parquet")
PATTERN_FEATURES_PATH = Path("data/features/main_benchmark/pattern_features.parquet")

SWC_IDS = [101, 103, 104, 107, 113, 114, 115, 120, 128, 132, 135]
SWC_NAMES = {
    101: "IntOverflow", 103: "FloatPragma", 104: "UncheckedCall",
    107: "Reentrancy", 113: "DoSGasLimit", 114: "TXOrigin",
    115: "AuthOrigin", 120: "WeakPRNG", 128: "DoSBlockGas",
    132: "UnexpEther", 135: "CodeSize",
}

CONTRACT_ID = "fp_runtime_unified"


def _load_data() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Load and merge all features. Returns (merged_df, enriched_X, Y, mask)."""
    fi = pd.read_parquet(FEATURE_INDEX_PATH)
    fi[CONTRACT_ID] = fi[CONTRACT_ID].astype(str).str.strip()

    target_cols = [f"swc_{s}" for s in SWC_IDS]
    mask_cols = [f"swc_{s}_assessed" for s in SWC_IDS]

    # Graph features
    gf = pd.read_parquet(GRAPH_FEATURES_PATH)
    gf[CONTRACT_ID] = gf[CONTRACT_ID].astype(str).str.strip()
    gf_cols = sorted([c for c in gf.columns if c.startswith("gf_")])

    # TF-IDF features
    tfidf = pd.read_parquet(TFIDF_FEATURES_PATH)
    tfidf[CONTRACT_ID] = tfidf[CONTRACT_ID].astype(str).str.strip()
    tfidf_cols = sorted([c for c in tfidf.columns if c.startswith("tfidf_")])

    # Pattern features
    pat = pd.read_parquet(PATTERN_FEATURES_PATH)
    pat[CONTRACT_ID] = pat[CONTRACT_ID].astype(str).str.strip()
    pat_cols = sorted([c for c in pat.columns if c.startswith("pat_")])

    merged = fi[[CONTRACT_ID] + target_cols + mask_cols].copy()
    merged = merged.merge(gf[[CONTRACT_ID] + gf_cols], on=CONTRACT_ID, how="inner")
    merged = merged.merge(tfidf[[CONTRACT_ID] + tfidf_cols], on=CONTRACT_ID, how="inner")
    merged = merged.merge(pat[[CONTRACT_ID] + pat_cols], on=CONTRACT_ID, how="inner")

    feature_cols = gf_cols + tfidf_cols + pat_cols
    X = merged[feature_cols].fillna(0.0).values.astype(np.float64)
    Y = merged[target_cols].fillna(0).values.astype(np.int32)
    mask = merged[mask_cols].fillna(0).values.astype(bool)

    logger.info("Loaded %d contracts, %d features, %d SWCs", X.shape[0], X.shape[1], Y.shape[1])
    return merged, X, Y, mask


def _train_predict_lr(
    X_train: np.ndarray, Y_train: np.ndarray, mask_train: np.ndarray,
    X_test: np.ndarray, n_swcs: int,
) -> np.ndarray:
    """Per-label Logistic Regression with masked training."""
    preds = np.zeros((X_test.shape[0], n_swcs), dtype=np.float64)
    for j in range(n_swcs):
        valid = mask_train[:, j]
        if valid.sum() < 2 or Y_train[valid, j].sum() == 0:
            continue
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
        clf.fit(X_train[valid], Y_train[valid, j])
        preds[:, j] = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test)
    return preds


def _train_predict_rf(
    X_train: np.ndarray, Y_train: np.ndarray, mask_train: np.ndarray,
    X_test: np.ndarray, n_swcs: int,
) -> np.ndarray:
    """Per-label Random Forest with masked training."""
    preds = np.zeros((X_test.shape[0], n_swcs), dtype=np.float64)
    for j in range(n_swcs):
        valid = mask_train[:, j]
        if valid.sum() < 2 or Y_train[valid, j].sum() == 0:
            continue
        clf = RandomForestClassifier(
            n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42,
        )
        clf.fit(X_train[valid], Y_train[valid, j])
        preds[:, j] = clf.predict_proba(X_test)[:, 1]
    return preds


def _train_predict_xgb(
    X_train: np.ndarray, Y_train: np.ndarray, mask_train: np.ndarray,
    X_test: np.ndarray, n_swcs: int,
) -> np.ndarray:
    """Per-label XGBoost with masked training and auto scale_pos_weight."""
    preds = np.zeros((X_test.shape[0], n_swcs), dtype=np.float64)
    for j in range(n_swcs):
        valid = mask_train[:, j]
        if valid.sum() < 2 or Y_train[valid, j].sum() == 0:
            continue
        n_pos = Y_train[valid, j].sum()
        n_neg = valid.sum() - n_pos
        spw = max(1.0, n_neg / max(1, n_pos))
        clf = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            scale_pos_weight=spw, use_label_encoder=False,
            eval_metric="logloss", verbosity=0, random_state=42,
        )
        clf.fit(X_train[valid], Y_train[valid, j])
        preds[:, j] = clf.predict_proba(X_test)[:, 1]
    return preds


MODEL_REGISTRY = {
    "logistic_regression": _train_predict_lr,
    "random_forest": _train_predict_rf,
}
if HAS_XGB:
    MODEL_REGISTRY["xgboost"] = _train_predict_xgb


def _threshold_tune(
    Y_val: np.ndarray, mask_val: np.ndarray, proba_val: np.ndarray,
) -> np.ndarray:
    """Per-SWC threshold tuning on validation set, optimizing F1."""
    n_swcs = Y_val.shape[1]
    thresholds = np.full(n_swcs, 0.5)
    for j in range(n_swcs):
        valid = mask_val[:, j]
        if valid.sum() < 2 or Y_val[valid, j].sum() == 0:
            continue
        best_f1 = -1.0
        best_t = 0.5
        for t in np.arange(0.1, 0.91, 0.05):
            yp = (proba_val[valid, j] >= t).astype(int)
            f = f1_score(Y_val[valid, j], yp, zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_t = t
        thresholds[j] = best_t
    return thresholds


def _compute_metrics(
    Y_true: np.ndarray, mask: np.ndarray, proba: np.ndarray, thresholds: np.ndarray,
) -> Dict[str, Any]:
    """Compute per-SWC and aggregate metrics."""
    n_swcs = Y_true.shape[1]
    per_swc = []
    f1s = []
    for j in range(n_swcs):
        valid = mask[:, j]
        swc_id = SWC_IDS[j]
        if valid.sum() == 0 or Y_true[valid, j].sum() == 0:
            per_swc.append({
                "swc_id": swc_id, "name": SWC_NAMES[swc_id],
                "f1": 0.0, "precision": 0.0, "recall": 0.0,
                "support": int(valid.sum()), "positives": 0,
            })
            f1s.append(0.0)
            continue
        yp = (proba[valid, j] >= thresholds[j]).astype(int)
        yt = Y_true[valid, j]
        f = f1_score(yt, yp, zero_division=0)
        p = precision_score(yt, yp, zero_division=0)
        r = recall_score(yt, yp, zero_division=0)
        per_swc.append({
            "swc_id": swc_id, "name": SWC_NAMES[swc_id],
            "f1": round(float(f), 4), "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "support": int(valid.sum()), "positives": int(yt.sum()),
        })
        f1s.append(float(f))

    macro_f1 = float(np.mean(f1s))
    # micro F1 across all valid labels
    all_yt, all_yp = [], []
    for j in range(n_swcs):
        valid = mask[:, j]
        if valid.sum() == 0:
            continue
        all_yt.append(Y_true[valid, j])
        all_yp.append((proba[valid, j] >= thresholds[j]).astype(int))
    if all_yt:
        micro_f1 = float(f1_score(np.concatenate(all_yt), np.concatenate(all_yp), zero_division=0))
    else:
        micro_f1 = 0.0

    return {
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "per_swc": per_swc,
    }


def run_cv(
    n_folds: int = 5,
    seed: int = 42,
    output_dir: Optional[Path] = None,
    models: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run stratified k-fold CV for all models."""
    merged, X, Y, mask = _load_data()

    if models is None:
        models = list(MODEL_REGISTRY.keys())

    # Create CV folds using multilabel stratified split
    # Use binarized Y for stratification (treat unlabeled as 0)
    Y_strat = Y.copy()
    cv = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = list(cv.split(X, Y_strat))

    results: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_folds": n_folds,
        "seed": seed,
        "n_contracts": X.shape[0],
        "n_features": X.shape[1],
        "swc_ids": SWC_IDS,
        "models": {},
    }

    for model_name in models:
        if model_name not in MODEL_REGISTRY:
            logger.warning("Unknown model: %s, skipping", model_name)
            continue

        train_fn = MODEL_REGISTRY[model_name]
        logger.info("Running %d-fold CV for %s ...", n_folds, model_name)

        fold_metrics: List[Dict[str, Any]] = []
        for fold_idx, (train_val_idx, test_idx) in enumerate(folds):
            # Split train_val into train + val (90/10)
            n_tv = len(train_val_idx)
            np.random.seed(seed + fold_idx)
            perm = np.random.permutation(n_tv)
            val_size = max(1, int(0.1 * n_tv))
            val_local_idx = perm[:val_size]
            train_local_idx = perm[val_size:]

            train_idx = train_val_idx[train_local_idx]
            val_idx = train_val_idx[val_local_idx]

            X_train, Y_train, mask_train = X[train_idx], Y[train_idx], mask[train_idx]
            X_val, Y_val, mask_val = X[val_idx], Y[val_idx], mask[val_idx]
            X_test, Y_test, mask_test = X[test_idx], Y[test_idx], mask[test_idx]

            # Train and predict
            proba_val = train_fn(X_train, Y_train, mask_train, X_val, Y.shape[1])
            proba_test = train_fn(X_train, Y_train, mask_train, X_test, Y.shape[1])

            # Threshold tuning on val
            thresholds = _threshold_tune(Y_val, mask_val, proba_val)

            # Evaluate on test
            metrics = _compute_metrics(Y_test, mask_test, proba_test, thresholds)
            metrics["fold"] = fold_idx + 1
            metrics["train_size"] = len(train_idx)
            metrics["val_size"] = len(val_idx)
            metrics["test_size"] = len(test_idx)
            fold_metrics.append(metrics)

            logger.info(
                "  Fold %d: macro_f1=%.4f micro_f1=%.4f",
                fold_idx + 1, metrics["macro_f1"], metrics["micro_f1"],
            )

        # Aggregate across folds
        macro_f1s = [m["macro_f1"] for m in fold_metrics]
        micro_f1s = [m["micro_f1"] for m in fold_metrics]

        per_swc_agg = []
        for j, swc_id in enumerate(SWC_IDS):
            swc_f1s = [m["per_swc"][j]["f1"] for m in fold_metrics]
            swc_precs = [m["per_swc"][j]["precision"] for m in fold_metrics]
            swc_recs = [m["per_swc"][j]["recall"] for m in fold_metrics]
            per_swc_agg.append({
                "swc_id": swc_id,
                "name": SWC_NAMES[swc_id],
                "f1_mean": round(float(np.mean(swc_f1s)), 4),
                "f1_std": round(float(np.std(swc_f1s)), 4),
                "precision_mean": round(float(np.mean(swc_precs)), 4),
                "recall_mean": round(float(np.mean(swc_recs)), 4),
                "fold_f1s": [round(f, 4) for f in swc_f1s],
            })

        model_results = {
            "macro_f1_mean": round(float(np.mean(macro_f1s)), 4),
            "macro_f1_std": round(float(np.std(macro_f1s)), 4),
            "micro_f1_mean": round(float(np.mean(micro_f1s)), 4),
            "micro_f1_std": round(float(np.std(micro_f1s)), 4),
            "per_swc": per_swc_agg,
            "fold_details": fold_metrics,
        }
        results["models"][model_name] = model_results

        logger.info(
            "%s: macro_f1=%.4f±%.4f  micro_f1=%.4f±%.4f",
            model_name,
            model_results["macro_f1_mean"], model_results["macro_f1_std"],
            model_results["micro_f1_mean"], model_results["micro_f1_std"],
        )

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "cv_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved CV results to %s", out_path)

        _write_summary_table(results, output_dir / "cv_summary.md")

    return results


def _write_summary_table(results: Dict[str, Any], path: Path) -> None:
    """Write a markdown summary table."""
    lines = [
        f"# {results['n_folds']}-Fold Cross-Validation Summary",
        f"",
        f"- **Contracts:** {results['n_contracts']}",
        f"- **Features:** {results['n_features']}",
        f"- **Seed:** {results['seed']}",
        f"- **Generated:** {results['generated_at_utc']}",
        f"",
    ]

    for model_name, mr in results["models"].items():
        lines.append(f"## {model_name}")
        lines.append(f"")
        lines.append(
            f"**Overall:** macro_f1={mr['macro_f1_mean']:.4f}±{mr['macro_f1_std']:.4f} "
            f"| micro_f1={mr['micro_f1_mean']:.4f}±{mr['micro_f1_std']:.4f}"
        )
        lines.append(f"")
        lines.append(f"| SWC | Name | F1 (mean±std) | Precision | Recall |")
        lines.append(f"|-----|------|---------------|-----------|--------|")
        for s in mr["per_swc"]:
            lines.append(
                f"| {s['swc_id']} | {s['name']} | "
                f"{s['f1_mean']:.3f}±{s['f1_std']:.3f} | "
                f"{s['precision_mean']:.3f} | {s['recall_mean']:.3f} |"
            )
        lines.append(f"")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Saved summary to %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="5-fold CV for Level 0.5 baselines")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="reports/phase6_level0.5")
    parser.add_argument(
        "--model", type=str, nargs="*", default=None,
        help="Model names to run (default: all). Options: logistic_regression, random_forest, xgboost",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    run_cv(
        n_folds=args.n_folds,
        seed=args.seed,
        output_dir=Path(args.output_dir),
        models=args.model,
    )


if __name__ == "__main__":
    main()
