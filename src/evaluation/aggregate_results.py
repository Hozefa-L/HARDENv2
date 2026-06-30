from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHASE6_CONFIG_PATH = PROJECT_ROOT / "configs/phase6.yaml"


def _resolve_path(path_value: Any) -> Path:
    raw = Path(str(path_value))
    if raw.is_absolute():
        return raw
    return (PROJECT_ROOT / raw).resolve()


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _safe_read_mapping(path: Path, context: str) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {context}: {path}")
    with path.open("r", encoding="utf-8") as fp:
        if path.suffix in {".yaml", ".yml"}:
            payload = yaml.safe_load(fp)
        else:
            payload = json.load(fp)
    if not isinstance(payload, dict):
        raise ValueError(f"{context} must be a mapping: {path}")
    return payload


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def _load_phase6_output_paths(config_path: Path) -> Dict[str, Path]:
    cfg = _safe_read_mapping(config_path, "Phase 6 config")
    outputs = cfg.get("outputs", {})
    if outputs is None:
        outputs = {}
    if not isinstance(outputs, dict):
        raise ValueError("`outputs` in Phase 6 config must be a mapping when provided.")

    return {
        "run_manifest_json": _resolve_path(outputs.get("run_manifest_json") or "reports/phase6/phase6_run_manifest.json"),
        "results_summary_json": _resolve_path(outputs.get("results_summary_json") or "reports/phase6/results_summary.json"),
        "ablation_summary_json": _resolve_path(outputs.get("ablation_summary_json") or "reports/phase6/ablation_summary.json"),
        "per_swc_metrics_parquet": _resolve_path(outputs.get("per_swc_metrics_parquet") or "reports/phase6/per_swc_metrics.parquet"),
    }


def _aggregate_metric_rows(metric_rows: pd.DataFrame) -> pd.DataFrame:
    if metric_rows.empty:
        return pd.DataFrame(
            columns=[
                "split",
                "dataset_variant",
                "model_group",
                "model_variant",
                "seed_count",
                "run_count",
                "macro_f1_mean",
                "macro_f1_std",
                "micro_f1_mean",
                "micro_f1_std",
                "multilabel_loss_mean",
                "multilabel_loss_std",
                "macro_average_precision_mean",
                "macro_average_precision_std",
                "micro_average_precision_mean",
                "micro_average_precision_std",
            ]
        )

    aggregated = (
        metric_rows.groupby(["split", "dataset_variant", "model_group", "model_variant"], dropna=False)
        .agg(
            seed_count=("seed", "nunique"),
            run_count=("run_id", "count"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            micro_f1_mean=("micro_f1", "mean"),
            micro_f1_std=("micro_f1", "std"),
            multilabel_loss_mean=("multilabel_loss", "mean"),
            multilabel_loss_std=("multilabel_loss", "std"),
            macro_average_precision_mean=("macro_average_precision", "mean"),
            macro_average_precision_std=("macro_average_precision", "std"),
            micro_average_precision_mean=("micro_average_precision", "mean"),
            micro_average_precision_std=("micro_average_precision", "std"),
        )
        .reset_index()
    )
    for col in [c for c in aggregated.columns if c.endswith("_std")]:
        aggregated[col] = aggregated[col].fillna(0.0)
    return aggregated


def _ablation_rows(
    *,
    aggregated_test: pd.DataFrame,
    base_variant: str,
    compare_variant: str,
    label: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if aggregated_test.empty:
        return rows

    model_keys = (
        aggregated_test[["model_group", "model_variant"]]
        .drop_duplicates()
        .sort_values(["model_group", "model_variant"])
        .itertuples(index=False, name=None)
    )
    for model_group, model_variant in model_keys:
        base = aggregated_test[
            (aggregated_test["dataset_variant"] == base_variant)
            & (aggregated_test["model_group"] == model_group)
            & (aggregated_test["model_variant"] == model_variant)
        ]
        comp = aggregated_test[
            (aggregated_test["dataset_variant"] == compare_variant)
            & (aggregated_test["model_group"] == model_group)
            & (aggregated_test["model_variant"] == model_variant)
        ]
        if base.empty or comp.empty:
            continue

        base_row = base.iloc[0]
        comp_row = comp.iloc[0]
        rows.append(
            {
                "comparison": label,
                "model_group": str(model_group),
                "model_variant": str(model_variant),
                "base_variant": base_variant,
                "compare_variant": compare_variant,
                "base_seed_count": int(base_row["seed_count"]),
                "compare_seed_count": int(comp_row["seed_count"]),
                "base_macro_f1_mean": float(base_row["macro_f1_mean"]),
                "compare_macro_f1_mean": float(comp_row["macro_f1_mean"]),
                "delta_macro_f1": float(comp_row["macro_f1_mean"] - base_row["macro_f1_mean"]),
                "base_micro_f1_mean": float(base_row["micro_f1_mean"]),
                "compare_micro_f1_mean": float(comp_row["micro_f1_mean"]),
                "delta_micro_f1": float(comp_row["micro_f1_mean"] - base_row["micro_f1_mean"]),
                "base_multilabel_loss_mean": float(base_row["multilabel_loss_mean"]),
                "compare_multilabel_loss_mean": float(comp_row["multilabel_loss_mean"]),
                "delta_multilabel_loss": float(comp_row["multilabel_loss_mean"] - base_row["multilabel_loss_mean"]),
                "base_macro_average_precision_mean": (
                    float(base_row["macro_average_precision_mean"])
                    if pd.notna(base_row["macro_average_precision_mean"])
                    else None
                ),
                "compare_macro_average_precision_mean": (
                    float(comp_row["macro_average_precision_mean"])
                    if pd.notna(comp_row["macro_average_precision_mean"])
                    else None
                ),
                "delta_macro_average_precision": (
                    float(comp_row["macro_average_precision_mean"] - base_row["macro_average_precision_mean"])
                    if pd.notna(base_row["macro_average_precision_mean"]) and pd.notna(comp_row["macro_average_precision_mean"])
                    else None
                ),
            }
        )
    return rows


def aggregate_phase6_results(
    config_path: Path = DEFAULT_PHASE6_CONFIG_PATH,
    *,
    run_manifest_override: Optional[Path] = None,
) -> Dict[str, Any]:
    resolved_config = config_path.resolve()
    output_paths = _load_phase6_output_paths(resolved_config)
    run_manifest_path = run_manifest_override.resolve() if run_manifest_override else output_paths["run_manifest_json"]

    run_manifest = _safe_read_mapping(run_manifest_path, "Phase 6 run manifest")
    runs_raw = run_manifest.get("runs", [])
    if not isinstance(runs_raw, list):
        raise ValueError("Phase 6 run manifest must contain a list `runs`.")

    metric_rows: List[Dict[str, Any]] = []
    per_swc_rows: List[Dict[str, Any]] = []
    completed_runs: List[Dict[str, Any]] = []
    failed_runs: List[Dict[str, Any]] = []
    unavailable_runs: List[Dict[str, Any]] = []
    pending_runs: List[Dict[str, Any]] = []
    missing_metrics_runs: List[Dict[str, Any]] = []
    missing_split_metrics_runs: List[Dict[str, Any]] = []

    for run in runs_raw:
        if not isinstance(run, dict):
            continue
        status = str(run.get("status", "unknown"))
        run_id = str(run.get("run_id", ""))
        base_row = {
            "run_id": run_id,
            "model_group": str(run.get("model_group", "")),
            "model_variant": str(run.get("model_variant", "")),
            "dataset_variant": str(run.get("dataset_variant", "")),
            "seed": int(run.get("seed", 0)),
            "status": status,
            "error": run.get("error"),
        }

        if status == "completed":
            completed_runs.append(base_row)
            metrics_rel = run.get("metrics_json")
            if not metrics_rel:
                missing_metrics_runs.append(base_row)
                continue
            metrics_path = _resolve_path(metrics_rel)
            if not metrics_path.exists():
                missing_metrics_runs.append({**base_row, "missing_metrics_path": _rel(metrics_path)})
                continue
            payload = _safe_read_mapping(metrics_path, f"run metrics `{run_id}`")
            split_metrics = payload.get("split_metrics", {})
            if not isinstance(split_metrics, dict):
                raise ValueError(f"Run metrics payload missing mapping `split_metrics`: {metrics_path}")
            observed_splits = {str(name) for name in split_metrics.keys()}
            missing_splits = sorted(set(["train", "val", "test"]) - observed_splits)
            if missing_splits:
                missing_split_metrics_runs.append({**base_row, "missing_splits": missing_splits})

            for split_name, split_payload in split_metrics.items():
                if not isinstance(split_payload, dict):
                    continue
                row = {
                    **base_row,
                    "split": str(split_name),
                    "multilabel_loss": float(split_payload.get("multilabel_loss", 0.0)),
                    "macro_precision": float(split_payload.get("macro_precision", 0.0)),
                    "macro_recall": float(split_payload.get("macro_recall", 0.0)),
                    "macro_f1": float(split_payload.get("macro_f1", 0.0)),
                    "micro_precision": float(split_payload.get("micro_precision", 0.0)),
                    "micro_recall": float(split_payload.get("micro_recall", 0.0)),
                    "micro_f1": float(split_payload.get("micro_f1", 0.0)),
                    "macro_average_precision": (
                        float(split_payload["macro_average_precision"])
                        if split_payload.get("macro_average_precision") is not None
                        else None
                    ),
                    "micro_average_precision": (
                        float(split_payload["micro_average_precision"])
                        if split_payload.get("micro_average_precision") is not None
                        else None
                    ),
                    "assessed_label_count": int(split_payload.get("assessed_label_count", 0)),
                    "assessed_positive_count": int(split_payload.get("assessed_positive_count", 0)),
                    "metrics_json": _rel(metrics_path),
                }
                metric_rows.append(row)

                per_swc = split_payload.get("per_swc", [])
                if isinstance(per_swc, list):
                    for swc_row in per_swc:
                        if not isinstance(swc_row, dict):
                            continue
                        per_swc_rows.append(
                            {
                                **base_row,
                                "split": str(split_name),
                                "swc_id": int(swc_row.get("swc_id")),
                                "support_assessed": int(swc_row.get("support_assessed", 0)),
                                "support_positive": int(swc_row.get("support_positive", 0)),
                                "support_negative": int(swc_row.get("support_negative", 0)),
                                "tp": int(swc_row.get("tp", 0)),
                                "fp": int(swc_row.get("fp", 0)),
                                "fn": int(swc_row.get("fn", 0)),
                                "tn": int(swc_row.get("tn", 0)),
                                "precision": float(swc_row.get("precision", 0.0)),
                                "recall": float(swc_row.get("recall", 0.0)),
                                "f1": float(swc_row.get("f1", 0.0)),
                                "average_precision": (
                                    float(swc_row["average_precision"])
                                    if swc_row.get("average_precision") is not None
                                    else None
                                ),
                                "average_precision_defined": bool(swc_row.get("average_precision_defined", False)),
                            }
                        )
        elif status == "failed":
            failed_runs.append(base_row)
        elif status == "unavailable":
            unavailable_runs.append(base_row)
        else:
            pending_runs.append(base_row)

    metric_df = pd.DataFrame(metric_rows)
    per_swc_df = pd.DataFrame(per_swc_rows)
    if per_swc_df.empty:
        per_swc_df = pd.DataFrame(
            columns=[
                "run_id",
                "model_group",
                "model_variant",
                "dataset_variant",
                "seed",
                "status",
                "error",
                "split",
                "swc_id",
                "support_assessed",
                "support_positive",
                "support_negative",
                "tp",
                "fp",
                "fn",
                "tn",
                "precision",
                "recall",
                "f1",
                "average_precision",
                "average_precision_defined",
            ]
        )
    output_paths["per_swc_metrics_parquet"].parent.mkdir(parents=True, exist_ok=True)
    per_swc_df.to_parquet(output_paths["per_swc_metrics_parquet"], index=False)

    aggregated_df = _aggregate_metric_rows(metric_df)
    aggregated_records = aggregated_df.to_dict(orient="records")
    test_df = aggregated_df[aggregated_df["split"] == "test"].copy()
    test_leaderboard = (
        test_df.sort_values(["dataset_variant", "macro_f1_mean"], ascending=[True, False]).to_dict(orient="records")
        if not test_df.empty
        else []
    )

    ablation_rows = []
    ablation_rows.extend(
        _ablation_rows(
            aggregated_test=test_df,
            base_variant="clean_default",
            compare_variant="no_proxy",
            label="clean_default_vs_no_proxy",
        )
    )
    ablation_rows.extend(
        _ablation_rows(
            aggregated_test=test_df,
            base_variant="cgt_only",
            compare_variant="combined_posaug",
            label="cgt_only_vs_combined_posaug",
        )
    )

    status_counts: Dict[str, int] = {}
    for run in runs_raw:
        if isinstance(run, dict):
            status = str(run.get("status", "unknown"))
            status_counts[status] = status_counts.get(status, 0) + 1

    expected_runs = int(len([run for run in runs_raw if isinstance(run, dict)]))
    completed_count = int(status_counts.get("completed", 0))
    evidence_complete = bool(completed_count == expected_runs)
    consistency_checks = {
        "all_completed_runs_have_metrics_files": bool(len(missing_metrics_runs) == 0),
        "all_completed_runs_have_train_val_test_metrics": bool(len(missing_split_metrics_runs) == 0),
        "all_runs_completed": bool(completed_count == expected_runs),
    }

    results_summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "phase6_aggregation",
        "config_path": _rel(resolved_config),
        "run_manifest_json": _rel(run_manifest_path),
        "status_counts": status_counts,
        "expected_run_count": expected_runs,
        "completed_run_count": completed_count,
        "failed_run_count": int(status_counts.get("failed", 0)),
        "unavailable_run_count": int(status_counts.get("unavailable", 0)),
        "pending_run_count": int(status_counts.get("pending", 0)) + int(status_counts.get("running", 0)),
        "missing_metrics_run_count": int(len(missing_metrics_runs)),
        "missing_split_metrics_run_count": int(len(missing_split_metrics_runs)),
        "evidence_base_complete": evidence_complete,
        "consistency_checks": consistency_checks,
        "split_metric_aggregates": aggregated_records,
        "test_leaderboard": test_leaderboard,
        "failed_runs": failed_runs,
        "unavailable_runs": unavailable_runs,
        "pending_runs": pending_runs,
        "missing_metrics_runs": missing_metrics_runs,
        "missing_split_metrics_runs": missing_split_metrics_runs,
        "artifacts": {
            "results_summary_json": _rel(output_paths["results_summary_json"]),
            "ablation_summary_json": _rel(output_paths["ablation_summary_json"]),
            "per_swc_metrics_parquet": _rel(output_paths["per_swc_metrics_parquet"]),
        },
    }
    _write_json(results_summary, output_paths["results_summary_json"])

    ablation_summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "phase6_ablation_aggregation",
        "config_path": _rel(resolved_config),
        "run_manifest_json": _rel(run_manifest_path),
        "rows": ablation_rows,
        "row_count": int(len(ablation_rows)),
    }
    _write_json(ablation_summary, output_paths["ablation_summary_json"])

    return {
        "results_summary_json": _rel(output_paths["results_summary_json"]),
        "ablation_summary_json": _rel(output_paths["ablation_summary_json"]),
        "per_swc_metrics_parquet": _rel(output_paths["per_swc_metrics_parquet"]),
        "completed_run_count": completed_count,
        "expected_run_count": expected_runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Phase 6 run metrics into summary artifacts.")
    parser.add_argument("--config", type=Path, default=DEFAULT_PHASE6_CONFIG_PATH)
    parser.add_argument("--manifest", type=Path, default=None, help="Optional override for Phase 6 run manifest path.")
    args = parser.parse_args()

    result = aggregate_phase6_results(
        config_path=args.config.resolve(),
        run_manifest_override=args.manifest.resolve() if args.manifest is not None else None,
    )
    print(f"results_summary: {result['results_summary_json']}")
    print(f"ablation_summary: {result['ablation_summary_json']}")
    print(f"per_swc_metrics: {result['per_swc_metrics_parquet']}")
    print(f"completed_runs: {result['completed_run_count']} / {result['expected_run_count']}")


if __name__ == "__main__":
    main()
