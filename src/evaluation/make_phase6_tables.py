from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping

import matplotlib.pyplot as plt
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


def _load_paths(config_path: Path) -> Dict[str, Path]:
    cfg = _safe_read_mapping(config_path, "Phase 6 config")
    outputs = cfg.get("outputs", {})
    if outputs is None:
        outputs = {}
    if not isinstance(outputs, dict):
        raise ValueError("`outputs` in Phase 6 config must be a mapping when provided.")

    report_dir = _resolve_path(outputs.get("report_dir") or "reports/phase6")
    return {
        "results_summary_json": _resolve_path(outputs.get("results_summary_json") or "reports/phase6/results_summary.json"),
        "ablation_summary_json": _resolve_path(outputs.get("ablation_summary_json") or "reports/phase6/ablation_summary.json"),
        "per_swc_metrics_parquet": _resolve_path(outputs.get("per_swc_metrics_parquet") or "reports/phase6/per_swc_metrics.parquet"),
        "final_tables_md": _resolve_path(outputs.get("final_tables_md") or "reports/phase6/final_tables.md"),
        "figures_dir": _resolve_path(outputs.get("figures_dir") or str(report_dir / "figures")),
    }


def _fmt_mean_std(mean: Any, std: Any, digits: int = 4) -> str:
    if pd.isna(mean):
        return "n/a"
    std_value = 0.0 if pd.isna(std) else float(std)
    return f"{float(mean):.{digits}f} ± {std_value:.{digits}f}"


def _markdown_table(frame: pd.DataFrame, columns: List[str]) -> str:
    if frame.empty:
        header = "| " + " | ".join(columns) + " |"
        sep = "| " + " | ".join(["---"] * len(columns)) + " |"
        return "\n".join([header, sep, "| " + " | ".join(["n/a"] * len(columns)) + " |"])

    safe = frame[columns].copy()
    for col in safe.columns:
        safe[col] = safe[col].map(lambda v: "n/a" if pd.isna(v) else str(v))
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in safe.to_numpy().tolist()]
    return "\n".join([header, sep] + rows)


def _save_empty_figure(path: Path, title: str, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.axis("off")
    plt.title(title)
    plt.text(0.5, 0.5, message, ha="center", va="center")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def make_phase6_tables(config_path: Path = DEFAULT_PHASE6_CONFIG_PATH) -> Dict[str, Any]:
    resolved_config = config_path.resolve()
    paths = _load_paths(resolved_config)
    results_summary = _safe_read_mapping(paths["results_summary_json"], "Phase 6 results summary")
    ablation_summary = _safe_read_mapping(paths["ablation_summary_json"], "Phase 6 ablation summary")
    per_swc = pd.read_parquet(paths["per_swc_metrics_parquet"])

    aggregates = pd.DataFrame(results_summary.get("split_metric_aggregates", []))
    test_aggregates = aggregates[aggregates["split"] == "test"].copy() if "split" in aggregates.columns else pd.DataFrame()
    clean_default = (
        test_aggregates[test_aggregates["dataset_variant"] == "clean_default"].copy()
        if "dataset_variant" in test_aggregates.columns
        else pd.DataFrame()
    )
    clean_default = clean_default.sort_values(["macro_f1_mean"], ascending=False)

    clean_table = pd.DataFrame(
        {
            "model_group": clean_default.get("model_group", pd.Series(dtype=str)),
            "model_variant": clean_default.get("model_variant", pd.Series(dtype=str)),
            "seeds": clean_default.get("seed_count", pd.Series(dtype=int)),
            "macro_f1 (test)": [
                _fmt_mean_std(mean, std)
                for mean, std in zip(
                    clean_default.get("macro_f1_mean", pd.Series(dtype=float)),
                    clean_default.get("macro_f1_std", pd.Series(dtype=float)),
                )
            ],
            "micro_f1 (test)": [
                _fmt_mean_std(mean, std)
                for mean, std in zip(
                    clean_default.get("micro_f1_mean", pd.Series(dtype=float)),
                    clean_default.get("micro_f1_std", pd.Series(dtype=float)),
                )
            ],
            "macro_AP (test)": [
                _fmt_mean_std(mean, std)
                for mean, std in zip(
                    clean_default.get("macro_average_precision_mean", pd.Series(dtype=float)),
                    clean_default.get("macro_average_precision_std", pd.Series(dtype=float)),
                )
            ],
            "loss (test)": [
                _fmt_mean_std(mean, std)
                for mean, std in zip(
                    clean_default.get("multilabel_loss_mean", pd.Series(dtype=float)),
                    clean_default.get("multilabel_loss_std", pd.Series(dtype=float)),
                )
            ],
        }
    )

    ablation_rows = pd.DataFrame(ablation_summary.get("rows", []))
    ablation_display = pd.DataFrame()
    if not ablation_rows.empty:
        ablation_display = ablation_rows[
            [
                "comparison",
                "model_group",
                "model_variant",
                "delta_macro_f1",
                "delta_micro_f1",
                "delta_multilabel_loss",
                "delta_macro_average_precision",
            ]
        ].copy()
        ablation_display["delta_macro_f1"] = ablation_display["delta_macro_f1"].map(lambda x: f"{float(x):+.4f}")
        ablation_display["delta_micro_f1"] = ablation_display["delta_micro_f1"].map(lambda x: f"{float(x):+.4f}")
        ablation_display["delta_multilabel_loss"] = ablation_display["delta_multilabel_loss"].map(lambda x: f"{float(x):+.4f}")
        ablation_display["delta_macro_average_precision"] = ablation_display["delta_macro_average_precision"].map(
            lambda x: "n/a" if pd.isna(x) else f"{float(x):+.4f}"
        )

    per_swc_table = pd.DataFrame()
    if not per_swc.empty:
        fused_clean = per_swc[
            (per_swc["split"] == "test")
            & (per_swc["dataset_variant"] == "clean_default")
            & (per_swc["model_group"] == "opcodegt")
            & (per_swc["model_variant"] == "fused")
        ].copy()
        if not fused_clean.empty:
            grouped = (
                fused_clean.groupby("swc_id", dropna=False)
                .agg(
                    seed_count=("seed", "nunique"),
                    precision_mean=("precision", "mean"),
                    recall_mean=("recall", "mean"),
                    f1_mean=("f1", "mean"),
                    avg_support_assessed=("support_assessed", "mean"),
                    avg_support_positive=("support_positive", "mean"),
                )
                .reset_index()
                .sort_values("swc_id")
            )
            grouped["precision_mean"] = grouped["precision_mean"].map(lambda x: f"{float(x):.4f}")
            grouped["recall_mean"] = grouped["recall_mean"].map(lambda x: f"{float(x):.4f}")
            grouped["f1_mean"] = grouped["f1_mean"].map(lambda x: f"{float(x):.4f}")
            grouped["avg_support_assessed"] = grouped["avg_support_assessed"].map(lambda x: f"{float(x):.2f}")
            grouped["avg_support_positive"] = grouped["avg_support_positive"].map(lambda x: f"{float(x):.2f}")
            per_swc_table = grouped

    figures_dir = paths["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    macro_fig_path = (figures_dir / "macro_f1_clean_default.png").resolve()
    ablation_fig_path = (figures_dir / "ablation_macro_f1_delta.png").resolve()

    if clean_default.empty:
        _save_empty_figure(
            macro_fig_path,
            title="Phase 6: clean_default test macro-F1",
            message="No completed clean_default test runs were available.",
        )
    else:
        plot_df = clean_default.sort_values("macro_f1_mean", ascending=False)
        plt.figure(figsize=(10, 5))
        x_labels = [f"{row.model_group}:{row.model_variant}" for row in plot_df.itertuples(index=False)]
        means = plot_df["macro_f1_mean"].astype(float).tolist()
        stds = plot_df["macro_f1_std"].astype(float).tolist()
        plt.bar(range(len(plot_df)), means, yerr=stds, capsize=4)
        plt.xticks(range(len(plot_df)), x_labels, rotation=35, ha="right")
        plt.ylabel("Macro F1 (test)")
        plt.title("Phase 6 clean_default performance")
        plt.tight_layout()
        plt.savefig(macro_fig_path, dpi=220)
        plt.close()

    if ablation_rows.empty:
        _save_empty_figure(
            ablation_fig_path,
            title="Phase 6: ablation deltas",
            message="No ablation rows were available.",
        )
    else:
        pivot = (
            ablation_rows.pivot_table(
                index="model_variant",
                columns="comparison",
                values="delta_macro_f1",
                aggfunc="mean",
            )
            .fillna(0.0)
            .sort_index()
        )
        plt.figure(figsize=(10, 5))
        width = 0.4
        x = list(range(len(pivot.index)))
        cols = list(pivot.columns)
        for idx, col in enumerate(cols):
            offset = (idx - (len(cols) - 1) / 2.0) * width
            plt.bar([v + offset for v in x], pivot[col].astype(float).tolist(), width=width, label=str(col))
        plt.axhline(0.0, color="black", linewidth=0.8)
        plt.xticks(x, [str(name) for name in pivot.index], rotation=35, ha="right")
        plt.ylabel("Delta macro F1 (test)")
        plt.title("Phase 6 ablation deltas")
        plt.legend()
        plt.tight_layout()
        plt.savefig(ablation_fig_path, dpi=220)
        plt.close()

    failed_rows = pd.DataFrame(results_summary.get("failed_runs", []))
    unavailable_rows = pd.DataFrame(results_summary.get("unavailable_runs", []))

    markdown = (
        "# Phase 6 Final Tables\n\n"
        f"- Generated at (UTC): `{datetime.now(timezone.utc).isoformat()}`\n"
        f"- Results summary source: `{_rel(paths['results_summary_json'])}`\n"
        f"- Ablation summary source: `{_rel(paths['ablation_summary_json'])}`\n"
        f"- Per-SWC parquet source: `{_rel(paths['per_swc_metrics_parquet'])}`\n\n"
        "## Run status\n\n"
        f"- expected runs: **{results_summary.get('expected_run_count', 0)}**\n"
        f"- completed runs: **{results_summary.get('completed_run_count', 0)}**\n"
        f"- failed runs: **{results_summary.get('failed_run_count', 0)}**\n"
        f"- unavailable runs: **{results_summary.get('unavailable_run_count', 0)}**\n"
        f"- pending/running runs: **{results_summary.get('pending_run_count', 0)}**\n"
        f"- evidence base complete: **{results_summary.get('evidence_base_complete', False)}**\n\n"
        "## Main comparison (test split, clean_default)\n\n"
        + _markdown_table(
            clean_table,
            ["model_group", "model_variant", "seeds", "macro_f1 (test)", "micro_f1 (test)", "macro_AP (test)", "loss (test)"],
        )
        + "\n\n## Ablation deltas\n\n"
        + _markdown_table(
            ablation_display,
            [
                "comparison",
                "model_group",
                "model_variant",
                "delta_macro_f1",
                "delta_micro_f1",
                "delta_multilabel_loss",
                "delta_macro_average_precision",
            ],
        )
        + "\n\n## OpcodeGT fused per-SWC (test, clean_default)\n\n"
        + _markdown_table(
            per_swc_table,
            ["swc_id", "seed_count", "precision_mean", "recall_mean", "f1_mean", "avg_support_assessed", "avg_support_positive"],
        )
        + "\n\n## Failed runs\n\n"
        + _markdown_table(failed_rows, ["run_id", "model_variant", "dataset_variant", "seed", "error"])
        + "\n\n## Unavailable runs\n\n"
        + _markdown_table(unavailable_rows, ["run_id", "model_variant", "dataset_variant", "seed", "error"])
        + "\n\n## Figures\n\n"
        f"- `{_rel(macro_fig_path)}`\n"
        f"- `{_rel(ablation_fig_path)}`\n"
    )

    paths["final_tables_md"].parent.mkdir(parents=True, exist_ok=True)
    paths["final_tables_md"].write_text(markdown, encoding="utf-8")

    return {
        "final_tables_md": _rel(paths["final_tables_md"]),
        "figures_dir": _rel(figures_dir),
        "figure_paths": [_rel(macro_fig_path), _rel(ablation_fig_path)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 6 publication tables and figures.")
    parser.add_argument("--config", type=Path, default=DEFAULT_PHASE6_CONFIG_PATH)
    args = parser.parse_args()
    result = make_phase6_tables(config_path=args.config.resolve())
    print(f"final_tables: {result['final_tables_md']}")
    print("figures:")
    for path in result["figure_paths"]:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
