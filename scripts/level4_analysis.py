#!/usr/bin/env python
"""Level 4 — Aggregate experiment results and generate publication tables + figures.

Usage:
    python scripts/level4_analysis.py --manifest reports/phase6_level4/phase6_run_manifest.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.statistical_tests import (
    bonferroni_correction,
    cohens_d,
    pairwise_significance_table,
    save_significance_report,
    wilcoxon_signed_rank_test,
)

# Publication metric keys (order for tables)
PUB_METRICS = [
    "test_macro_f1",
    "test_micro_f1",
    "test_macro_precision",
    "test_macro_recall",
    "test_macro_mcc",
    "test_macro_accuracy",
    "test_subset_accuracy",
    "test_macro_average_precision",
]

PUB_METRIC_LABELS = {
    "test_macro_f1": "Macro-F1",
    "test_micro_f1": "Micro-F1",
    "test_macro_precision": "Macro-P",
    "test_macro_recall": "Macro-R",
    "test_macro_mcc": "Macro-MCC",
    "test_macro_accuracy": "Macro-Acc",
    "test_subset_accuracy": "Subset-Acc",
    "test_macro_average_precision": "Macro-AP",
}

# Model display names for tables
MODEL_DISPLAY_NAMES = {
    "opcodegt_fused": "OpcodeGT v2 (Fused)",
    "opcodegt_graph_only": "OpcodeGT v2 (Graph)",
    "opcodegt_opcode_only": "OpcodeGT v2 (Opcode)",
    "opcodegt_v2_full": "OpcodeGT v2-Full",
    "opcodegt_v2_no_hmpgt": "v2 w/o HMPGT",
    "opcodegt_v2_no_cross_attention": "v2 w/o CrossAttn",
    "opcodegt_v2_no_label_attention": "v2 w/o LabelAttn",
    "opcodegt_v2_cfg_only": "v2 CFG-Only",
    "opcodegt_v1_simple_gin_fusion": "OpcodeGT v1 (GIN)",
    "classical_rf": "Random Forest",
    "classical_xgboost": "XGBoost",
    "classical_lgbm": "LightGBM",
    "classical_graph_lr": "Graph-LR",
    "mlp_opcode": "MLP (Opcode)",
    "mlp_graph": "MLP (Graph)",
    "mlp_fusion_concat": "MLP (Fusion)",
    "codebert_classifier": "CodeBERT",
    "gcn_baseline": "GCN",
    "gat_baseline": "GAT",
    "bilstm_baseline": "BiLSTM",
    "mythril_v0_24_8": "Mythril v0.24.8",
}

FIGURE_DISPLAY_NAMES = {
    **MODEL_DISPLAY_NAMES,
    "opcodegt_fused": "HARDENv2 (Fused)",
    "opcodegt_graph_only": "HARDENv2 (Graph)",
    "opcodegt_opcode_only": "HARDENv2 (Opcode)",
}

MODEL_FIGURE_FAMILY = {
    "opcodegt_fused": "HARDENv2",
    "opcodegt_graph_only": "HARDENv2",
    "opcodegt_opcode_only": "HARDENv2",
    "opcodegt_v2_full": "HARDENv2",
    "opcodegt_v2_no_hmpgt": "HARDENv2",
    "opcodegt_v2_no_cross_attention": "HARDENv2",
    "opcodegt_v2_no_label_attention": "HARDENv2",
    "opcodegt_v2_cfg_only": "HARDENv2",
    "opcodegt_v1_simple_gin_fusion": "HARDENv2",
    "classical_rf": "Classical ML",
    "classical_xgboost": "Classical ML",
    "classical_lgbm": "Classical ML",
    "classical_graph_lr": "Classical ML",
    "mlp_opcode": "Learned Baselines",
    "mlp_graph": "Learned Baselines",
    "mlp_fusion_concat": "Learned Baselines",
    "codebert_classifier": "Learned Baselines",
    "gcn_baseline": "Learned Baselines",
    "gat_baseline": "Learned Baselines",
    "bilstm_baseline": "Learned Baselines",
    "mythril_v0_24_8": "External Tool",
}

MACRO_F1_FAMILY_COLORS = {
    "HARDENv2": "#1f77b4",
    "Classical ML": "#2ca02c",
    "Learned Baselines": "#7f7f7f",
    "External Tool": "#d62728",
}

MACRO_F1_LEGEND_ORDER = [
    "HARDENv2",
    "Classical ML",
    "Learned Baselines",
    "External Tool",
]

HEATMAP_CMAP_NAME = "RdYlGn"

REPRESENTATIVE_HEATMAP_GROUPS = [
    ("ours", ["opcodegt_graph_only", "opcodegt_opcode_only", "opcodegt_fused"]),
    ("classical", ["classical_xgboost", "classical_rf", "classical_lgbm"]),
    ("external", ["mythril_v0_24_8"]),
    ("neural", ["mlp_opcode", "codebert_classifier", "bilstm_baseline"]),
    ("homogeneous_gnn", ["gcn_baseline", "gat_baseline"]),
]

REPRESENTATIVE_HEATMAP_MODEL_IDS = [
    model_id
    for _, group_model_ids in REPRESENTATIVE_HEATMAP_GROUPS
    for model_id in group_model_ids
]


def load_manifest(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def aggregate_by_model(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Group completed runs by model_id, compute mean±std across seeds."""
    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for run in manifest["runs"]:
        if run.get("status") != "completed":
            continue
        model_id = run["model_id"]
        summary = run.get("summary", {})
        by_model[model_id].append(summary)

    results: Dict[str, Dict[str, Any]] = {}
    for model_id, summaries in by_model.items():
        agg: Dict[str, Any] = {"n_seeds": len(summaries), "seeds": []}
        for metric in PUB_METRICS:
            values = []
            for s in summaries:
                v = s.get(metric)
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    values.append(float(v))
            if values:
                agg[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1) if len(values) > 1 else 0.0),
                    "values": values,
                }
            else:
                agg[metric] = {"mean": 0.0, "std": 0.0, "values": []}
        results[model_id] = agg
    return results


def load_per_swc_metrics(manifest: Dict[str, Any], metrics_dir: Path) -> Dict[str, List[Dict]]:
    """Load per-SWC test metrics from individual run JSON files."""
    by_model: Dict[str, List[Dict]] = defaultdict(list)
    for run in manifest["runs"]:
        if run.get("status") != "completed":
            continue
        metrics_json = run.get("metrics_json")
        if not metrics_json:
            continue
        mpath = PROJECT_ROOT / metrics_json
        if not mpath.exists():
            continue
        with open(mpath) as f:
            data = json.load(f)
        # Try split_metrics.test.per_swc first, then legacy test.per_swc_rows
        split_metrics = data.get("split_metrics", {})
        test_data = split_metrics.get("test", data.get("test", {}))
        per_swc = test_data.get("per_swc", test_data.get("per_swc_rows", []))
        by_model[run["model_id"]].append(per_swc)
    return by_model


def inject_external_baseline(
    *,
    agg: Dict[str, Dict[str, Any]],
    per_swc_data: Dict[str, List[Dict]],
    model_id: str,
    summary: Dict[str, Any],
) -> None:
    metrics = summary.get("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError(f"External baseline summary for {model_id} must contain a 'metrics' mapping.")

    metric_key_map = {
        "test_macro_f1": "macro_f1",
        "test_micro_f1": "micro_f1",
        "test_macro_precision": "macro_precision",
        "test_macro_recall": "macro_recall",
        "test_macro_mcc": "macro_mcc",
        "test_macro_accuracy": "macro_accuracy",
        "test_subset_accuracy": "subset_accuracy",
        "test_macro_average_precision": "macro_average_precision",
    }
    entry: Dict[str, Any] = {"n_seeds": 1}
    for agg_key, summary_key in metric_key_map.items():
        value = metrics.get(summary_key)
        if value is None:
            entry[agg_key] = {"mean": 0.0, "std": 0.0, "values": []}
        else:
            scalar = float(value)
            entry[agg_key] = {"mean": scalar, "std": 0.0, "values": [scalar]}
    agg[model_id] = entry

    per_swc_rows = metrics.get("per_swc", [])
    if isinstance(per_swc_rows, list) and per_swc_rows:
        per_swc_data[model_id] = [per_swc_rows]


def load_external_baselines(
    *,
    agg: Dict[str, Dict[str, Any]],
    per_swc_data: Dict[str, List[Dict]],
) -> None:
    mythril_summary_path = PROJECT_ROOT / "reports" / "mythril_v0_24_8" / "evaluation_summary.json"
    if mythril_summary_path.exists():
        with open(mythril_summary_path) as f:
            inject_external_baseline(
                agg=agg,
                per_swc_data=per_swc_data,
                model_id="mythril_v0_24_8",
                summary=json.load(f),
            )


def select_heatmap_model_ids(available_model_ids: Any) -> List[str]:
    available = set(available_model_ids)
    return [model_id for model_id in REPRESENTATIVE_HEATMAP_MODEL_IDS if model_id in available]


def _select_heatmap_group_breaks(selected_model_ids: List[str]) -> List[int]:
    selected = set(selected_model_ids)
    breaks: List[int] = []
    count = 0
    for _, group_model_ids in REPRESENTATIVE_HEATMAP_GROUPS:
        present = [model_id for model_id in group_model_ids if model_id in selected]
        count += len(present)
        if present and count < len(selected_model_ids):
            breaks.append(count)
    return breaks


def _macro_f1_color_for_model(model_id: str) -> str:
    family = MODEL_FIGURE_FAMILY.get(model_id, "Learned Baselines")
    return MACRO_F1_FAMILY_COLORS[family]


def _fmt(mean: float, std: float) -> str:
    """Format mean±std for publication tables."""
    if std > 0:
        return f"{mean:.4f}±{std:.4f}"
    return f"{mean:.4f}"


def generate_main_results_table(agg: Dict[str, Dict[str, Any]]) -> str:
    """Table I: Main results across all models."""
    lines = ["# Table I: Main Performance Results (Mean ± Std across 5 Seeds)", ""]

    # Header
    metric_cols = ["Macro-F1", "Micro-F1", "Macro-P", "Macro-R", "Macro-MCC", "Macro-Acc", "Subset-Acc"]
    header = "| Model | " + " | ".join(metric_cols) + " |"
    sep = "|" + "|".join(["---"] * (len(metric_cols) + 1)) + "|"
    lines.append(header)
    lines.append(sep)

    # Sort by macro_f1 descending
    sorted_models = sorted(
        agg.items(),
        key=lambda x: x[1].get("test_macro_f1", {}).get("mean", 0),
        reverse=True,
    )

    for model_id, data in sorted_models:
        name = MODEL_DISPLAY_NAMES.get(model_id, model_id)
        cols = []
        for mk in PUB_METRICS[:7]:  # Exclude AP for main table
            d = data.get(mk, {"mean": 0, "std": 0})
            cols.append(_fmt(d["mean"], d["std"]))
        lines.append(f"| {name} | " + " | ".join(cols) + " |")

    lines.append("")
    return "\n".join(lines)


def generate_ablation_table(agg: Dict[str, Dict[str, Any]]) -> str:
    """Table II: Ablation study for OpcodeGT v2 components."""
    lines = ["# Table II: Ablation Study — OpcodeGT v2 Components", ""]

    ablation_models = [
        "opcodegt_v2_full",
        "opcodegt_fused",
        "opcodegt_v2_no_hmpgt",
        "opcodegt_v2_no_cross_attention",
        "opcodegt_v2_no_label_attention",
        "opcodegt_v2_cfg_only",
        "opcodegt_v1_simple_gin_fusion",
    ]

    metric_cols = ["Macro-F1", "Micro-F1", "Macro-P", "Macro-R", "Macro-MCC"]
    header = "| Variant | " + " | ".join(metric_cols) + " | Δ Macro-F1 |"
    sep = "|" + "|".join(["---"] * (len(metric_cols) + 2)) + "|"
    lines.append(header)
    lines.append(sep)

    # Baseline: v2_full
    baseline_f1 = agg.get("opcodegt_v2_full", {}).get("test_macro_f1", {}).get("mean", 0)
    if baseline_f1 == 0:
        baseline_f1 = agg.get("opcodegt_fused", {}).get("test_macro_f1", {}).get("mean", 0)

    for model_id in ablation_models:
        if model_id not in agg:
            continue
        data = agg[model_id]
        name = MODEL_DISPLAY_NAMES.get(model_id, model_id)
        cols = []
        for mk in PUB_METRICS[:5]:
            d = data.get(mk, {"mean": 0, "std": 0})
            cols.append(_fmt(d["mean"], d["std"]))
        f1_mean = data.get("test_macro_f1", {}).get("mean", 0)
        delta = f1_mean - baseline_f1
        delta_str = f"{delta:+.4f}" if model_id != "opcodegt_v2_full" else "—"
        lines.append(f"| {name} | " + " | ".join(cols) + f" | {delta_str} |")

    lines.append("")
    return "\n".join(lines)


def generate_per_swc_table(
    per_swc_data: Dict[str, List[Dict]],
    top_n: int = 5,
    agg: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """Table III: Per-SWC F1 breakdown for top N models."""
    lines = ["# Table III: Per-SWC F1 Scores (Top 5 Models)", ""]

    # Select top models by macro F1
    if agg:
        sorted_models = sorted(
            agg.items(),
            key=lambda x: x[1].get("test_macro_f1", {}).get("mean", 0),
            reverse=True,
        )[:top_n]
        top_ids = [m[0] for m in sorted_models]
    else:
        top_ids = list(per_swc_data.keys())[:top_n]

    # Collect all SWC ids
    swc_ids_set = set()
    for model_id in top_ids:
        for seed_rows in per_swc_data.get(model_id, []):
            for row in seed_rows:
                swc_ids_set.add(str(row.get("swc_id", "")))
    swc_ids = sorted(swc_ids_set, key=lambda x: int(x) if x.isdigit() else 0)

    header = "| Model | " + " | ".join([f"SWC-{s}" for s in swc_ids]) + " |"
    sep = "|" + "|".join(["---"] * (len(swc_ids) + 1)) + "|"
    lines.append(header)
    lines.append(sep)

    for model_id in top_ids:
        name = MODEL_DISPLAY_NAMES.get(model_id, model_id)
        seed_rows_list = per_swc_data.get(model_id, [])
        # Average per-SWC F1 across seeds
        swc_f1s: Dict[str, List[float]] = defaultdict(list)
        for seed_rows in seed_rows_list:
            for row in seed_rows:
                sid = str(row.get("swc_id", ""))
                f1 = row.get("f1", 0.0)
                if f1 is not None:
                    swc_f1s[sid].append(float(f1))

        cols = []
        for sid in swc_ids:
            vals = swc_f1s.get(sid, [0.0])
            mean = np.mean(vals) if vals else 0.0
            cols.append(f"{mean:.3f}")
        lines.append(f"| {name} | " + " | ".join(cols) + " |")

    lines.append("")
    return "\n".join(lines)


def run_statistical_analysis(
    agg: Dict[str, Dict[str, Any]],
    output_dir: Path,
    reference_model: str = "opcodegt_graph_only",
) -> str:
    """Run Wilcoxon signed-rank tests and generate significance table."""
    lines = ["# Table VI: Statistical Significance (Wilcoxon Signed-Rank Test)", ""]

    ref_scores = agg.get(reference_model, {}).get("test_macro_f1", {}).get("values", [])
    if not ref_scores or len(ref_scores) < 3:
        lines.append("_Insufficient data for statistical tests (need ≥3 seeds)._")
        return "\n".join(lines)

    ref_name = MODEL_DISPLAY_NAMES.get(reference_model, reference_model)

    header = f"| Model | vs {ref_name} p-value | Cohen's d | Significant (α=0.05) |"
    sep = "|---|---|---|---|"
    lines.append(header)
    lines.append(sep)

    comparisons = []
    for model_id, data in agg.items():
        if model_id == reference_model:
            continue
        scores = data.get("test_macro_f1", {}).get("values", [])
        if len(scores) < 3:
            continue
        comparisons.append((model_id, scores))

    p_values = []
    rows = []
    for model_id, scores in comparisons:
        result = wilcoxon_signed_rank_test(ref_scores, scores)
        p_val = result["p_value"]
        p_values.append(p_val)
        d = cohens_d(ref_scores, scores)
        rows.append((model_id, p_val, d))

    # Bonferroni correction
    if p_values:
        corrected = bonferroni_correction(p_values)
    else:
        corrected = []

    for i, (model_id, p_val, d) in enumerate(rows):
        name = MODEL_DISPLAY_NAMES.get(model_id, model_id)
        p_corr = corrected[i]["corrected_p"] if i < len(corrected) else p_val
        sig = "✓" if p_corr < 0.05 else "✗"
        lines.append(f"| {name} | {p_corr:.4f} | {d['d']:.3f} | {sig} |")

    # Save detailed report
    all_scores = {mid: data.get("test_macro_f1", {}).get("values", []) for mid, data in agg.items() if len(data.get("test_macro_f1", {}).get("values", [])) >= 3}
    report_path = output_dir / "statistical_tests.json"
    sig_table = pairwise_significance_table(all_scores)
    save_significance_report(
        sig_table,
        report_path,
    )

    lines.append("")
    lines.append(f"_Full pairwise results saved to `{report_path.relative_to(PROJECT_ROOT)}`_")
    return "\n".join(lines)


def generate_efficiency_table(manifest: Dict[str, Any]) -> str:
    """Table V: Model efficiency comparison — uses benchmark_efficiency.py output."""
    lines = ["# Table V: Model Efficiency", ""]

    # Try to load pre-computed efficiency benchmarks
    benchmark_path = PROJECT_ROOT / "reports" / "phase6_level4" / "efficiency_benchmarks.json"
    if benchmark_path.exists():
        with open(benchmark_path) as f:
            benchmarks = json.load(f)

        header = "| Model | Total Params | Inference (ms) | GPU Mem (MB) |"
        sep = "|---|---|---|---|"
        lines.append(header)
        lines.append(sep)

        for b in benchmarks:
            name = MODEL_DISPLAY_NAMES.get(b["model_name"], b["model_name"])
            total = b.get("total_params", "N/A")
            inf_time = b.get("inference_time_ms", "N/A")
            gpu_mem = b.get("peak_gpu_memory_mb", "N/A")

            if isinstance(total, int):
                if total > 1e6:
                    total_str = f"{total/1e6:.2f}M"
                elif total > 1e3:
                    total_str = f"{total/1e3:.1f}K"
                else:
                    total_str = str(total)
            else:
                total_str = str(total)

            time_str = f"{inf_time:.2f}" if isinstance(inf_time, (int, float)) else str(inf_time)
            mem_str = f"{gpu_mem:.1f}" if isinstance(gpu_mem, (int, float)) else str(gpu_mem)
            lines.append(f"| {name} | {total_str} | {time_str} | {mem_str} |")
    else:
        lines.append("_Efficiency benchmarks not yet generated. Run `scripts/benchmark_efficiency.py`._")

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Level 4 experiment analysis")
    parser.add_argument("--manifest", type=str, required=True, help="Path to run manifest JSON")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for tables/figures")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path)

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = manifest_path.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    # Count completed
    completed = sum(1 for r in manifest["runs"] if r.get("status") == "completed")
    total = len(manifest["runs"])
    failed = sum(1 for r in manifest["runs"] if r.get("status") == "failed")
    print(f"Runs: {completed}/{total} completed, {failed} failed")

    if completed == 0:
        print("No completed runs — nothing to analyze.")
        return

    # Aggregate
    agg = aggregate_by_model(manifest)

    # Load per-SWC metrics
    metrics_dir = output_dir.parent / "metrics" if "metrics" not in str(output_dir) else output_dir
    per_swc_data = load_per_swc_metrics(manifest, metrics_dir)
    load_external_baselines(agg=agg, per_swc_data=per_swc_data)
    print(f"Models with results: {len(agg)}")

    # Generate all tables
    tables = []
    tables.append(generate_main_results_table(agg))
    tables.append(generate_ablation_table(agg))
    tables.append(generate_per_swc_table(per_swc_data, top_n=5, agg=agg))
    tables.append(generate_efficiency_table(manifest))
    tables.append(run_statistical_analysis(agg, output_dir))

    # Write combined publication tables
    combined = "\n\n".join(tables)
    tables_path = output_dir / "publication_tables.md"
    with open(tables_path, "w") as f:
        f.write(combined)
    print(f"Publication tables saved to: {tables_path}")

    # Write aggregated results JSON
    agg_json_path = output_dir / "aggregated_results.json"
    # Convert numpy to native Python for JSON
    agg_serializable = {}
    for mid, data in agg.items():
        entry = {"n_seeds": data["n_seeds"]}
        for mk in PUB_METRICS:
            d = data.get(mk, {"mean": 0, "std": 0, "values": []})
            entry[mk] = {"mean": d["mean"], "std": d["std"], "values": d["values"]}
        agg_serializable[mid] = entry
    with open(agg_json_path, "w") as f:
        json.dump(agg_serializable, f, indent=2)
    print(f"Aggregated results saved to: {agg_json_path}")

    # Print quick summary
    print("\n--- Quick Summary (Test Macro-F1) ---")
    sorted_models = sorted(
        agg.items(),
        key=lambda x: x[1].get("test_macro_f1", {}).get("mean", 0),
        reverse=True,
    )
    for model_id, data in sorted_models:
        d = data.get("test_macro_f1", {"mean": 0, "std": 0})
        name = MODEL_DISPLAY_NAMES.get(model_id, model_id)
        print(f"  {name:35s}  {d['mean']:.4f} ± {d['std']:.4f}  (n={data['n_seeds']})")

    # Generate figures if matplotlib available
    try:
        _generate_figures(agg, per_swc_data, output_dir)
    except ImportError:
        print("matplotlib not available — skipping figures")
    except Exception as e:
        print(f"Figure generation failed: {e}")


def _generate_figures(
    agg: Dict[str, Dict[str, Any]],
    per_swc_data: Dict[str, List[Dict]],
    output_dir: Path,
):
    """Generate publication-quality figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Macro-F1 bar chart with error bars
    sorted_models = sorted(
        agg.items(),
        key=lambda x: x[1].get("test_macro_f1", {}).get("mean", 0),
        reverse=True,
    )

    names = [FIGURE_DISPLAY_NAMES.get(mid, mid) for mid, _ in sorted_models]
    means = [d.get("test_macro_f1", {}).get("mean", 0) for _, d in sorted_models]
    stds = [d.get("test_macro_f1", {}).get("std", 0) for _, d in sorted_models]
    bar_colors = [_macro_f1_color_for_model(mid) for mid, _ in sorted_models]

    fig_height = max(6.0, 0.38 * len(names) + 1.25)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.barh(range(len(names)), means, xerr=stds, capsize=3, color=bar_colors, edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Test Macro-F1", fontsize=11)
    ax.set_title("Model Comparison — Test Macro-F1", fontsize=13)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    legend_handles = [
        Patch(facecolor=MACRO_F1_FAMILY_COLORS[family], edgecolor="white", label=family)
        for family in MACRO_F1_LEGEND_ORDER
    ]
    ax.legend(handles=legend_handles, title="Model family", loc="lower right", fontsize=9, title_fontsize=9)
    plt.tight_layout()
    fig.savefig(fig_dir / "macro_f1_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_dir / 'macro_f1_comparison.png'}")

    # Figure 2: Per-SWC F1 heatmap for representative models
    top_ids = select_heatmap_model_ids(per_swc_data.keys())
    swc_ids_set = set()
    for mid in top_ids:
        for seed_rows in per_swc_data.get(mid, []):
            for row in seed_rows:
                swc_ids_set.add(str(row.get("swc_id", "")))
    swc_ids = sorted(swc_ids_set, key=lambda x: int(x) if x.isdigit() else 0)

    if swc_ids and top_ids:
        matrix = np.zeros((len(top_ids), len(swc_ids)))
        for i, mid in enumerate(top_ids):
            swc_f1s: Dict[str, List[float]] = defaultdict(list)
            for seed_rows in per_swc_data.get(mid, []):
                for row in seed_rows:
                    sid = str(row.get("swc_id", ""))
                    f1 = row.get("f1", 0.0)
                    if f1 is not None:
                        swc_f1s[sid].append(float(f1))
            for j, sid in enumerate(swc_ids):
                vals = swc_f1s.get(sid, [0.0])
                matrix[i, j] = np.mean(vals) if vals else 0.0

        fig_height = max(6.0, 0.5 * len(top_ids) + 1.4)
        fig, ax = plt.subplots(figsize=(12.5, fig_height))
        im = ax.imshow(matrix, cmap=HEATMAP_CMAP_NAME, aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(swc_ids)))
        ax.set_xticklabels([f"SWC-{s}" for s in swc_ids], fontsize=9)
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(top_ids)))
        ax.set_yticklabels([FIGURE_DISPLAY_NAMES.get(m, m) for m in top_ids], fontsize=9)
        for break_after in _select_heatmap_group_breaks(top_ids):
            ax.axhline(break_after - 0.5, color="white", linewidth=2.0)
        # Annotate cells
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                text_color = "white" if val <= 0.2 or val >= 0.75 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7.5, color=text_color)
        plt.colorbar(im, ax=ax, label="F1 Score")
        ax.set_title("Per-SWC F1 Scores — Representative Models", fontsize=13)
        fig.subplots_adjust(left=0.23, right=0.94, bottom=0.18, top=0.9)
        fig.savefig(fig_dir / "per_swc_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fig_dir / 'per_swc_heatmap.png'}")

    # Figure 3: Multi-metric radar/grouped bar for top 3 models
    top3_ids = [mid for mid, _ in sorted_models[:3]]
    metrics_for_radar = ["test_macro_f1", "test_macro_precision", "test_macro_recall", "test_macro_mcc", "test_macro_accuracy"]
    labels_for_radar = ["F1", "Precision", "Recall", "MCC", "Accuracy"]

    if top3_ids:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(labels_for_radar))
        width = 0.25
        for i, mid in enumerate(top3_ids):
            vals = [agg[mid].get(mk, {}).get("mean", 0) for mk in metrics_for_radar]
            errs = [agg[mid].get(mk, {}).get("std", 0) for mk in metrics_for_radar]
            ax.bar(x + i * width, vals, width, yerr=errs, capsize=3,
                   label=MODEL_DISPLAY_NAMES.get(mid, mid), alpha=0.85)
        ax.set_xticks(x + width)
        ax.set_xticklabels(labels_for_radar, fontsize=10)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title("Multi-Metric Comparison — Top 3 Models", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 1.0)
        plt.tight_layout()
        fig.savefig(fig_dir / "multi_metric_top3.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fig_dir / 'multi_metric_top3.png'}")


if __name__ == "__main__":
    main()
