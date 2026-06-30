import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import nbformat as nbf
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SWC_DECISION_MATRIX_PARQUET = PROJECT_ROOT / "reports/phase1/swc_decision_matrix.parquet"
DEFAULT_LABEL_REPORT_JSON = PROJECT_ROOT / "reports/phase1/label_report.json"
DEFAULT_SPLIT_STATS_JSON = PROJECT_ROOT / "reports/phase1/split_stats.json"
DEFAULT_DEDUP_REPORT_JSON = PROJECT_ROOT / "reports/phase1/dedup_report.json"
DEFAULT_RUNTIME_EXTRACTION_REPORT_JSON = PROJECT_ROOT / "reports/phase1/runtime_extraction_report.json"
DEFAULT_UNIFIED_CONTRACTS_PARQUET = PROJECT_ROOT / "data/curated/unified_contracts.parquet"
DEFAULT_UNIFIED_LABELS_PARQUET = PROJECT_ROOT / "data/curated/unified_labels.parquet"

DEFAULT_AUDIT_SUMMARY_MD = PROJECT_ROOT / "reports/phase1/phase1_audit_summary.md"
DEFAULT_FINAL_RECOMMENDATION_CSV = PROJECT_ROOT / "reports/phase1/final_swc_recommendation.csv"
DEFAULT_SWC_DECISION_MATRIX_CSV = PROJECT_ROOT / "reports/phase1/swc_decision_matrix.csv"
DEFAULT_UNIFIED_CONTRACTS_HEAD_CSV = PROJECT_ROOT / "reports/phase1/unified_contracts_head.csv"
DEFAULT_UNIFIED_LABELS_SAMPLE_CSV = PROJECT_ROOT / "reports/phase1/unified_labels_sample.csv"
DEFAULT_AUDIT_NOTEBOOK = PROJECT_ROOT / "notebooks/01d_phase1_audit.ipynb"

MAIN_MIN_KNOWN = 100
MAIN_MIN_MINORITY_CLASS = 10


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        return str(path)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_notebook(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        nbf.v4.new_markdown_cell(
            "# Phase 1 Audit (01d)\n"
            "This notebook reproduces the SWC-level Phase 1 audit and final recommendations."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import pandas as pd\n\n"
            "PROJECT_ROOT = Path.cwd().resolve()\n"
            "if not (PROJECT_ROOT / 'reports/phase1/swc_decision_matrix.parquet').exists():\n"
            "    PROJECT_ROOT = PROJECT_ROOT.parent\n\n"
            "decision_matrix_path = PROJECT_ROOT / 'reports/phase1/swc_decision_matrix.parquet'\n"
            "label_report_path = PROJECT_ROOT / 'reports/phase1/label_report.json'\n"
            "split_stats_path = PROJECT_ROOT / 'reports/phase1/split_stats.json'\n"
            "dedup_report_path = PROJECT_ROOT / 'reports/phase1/dedup_report.json'\n"
            "runtime_report_path = PROJECT_ROOT / 'reports/phase1/runtime_extraction_report.json'\n"
            "unified_contracts_path = PROJECT_ROOT / 'data/curated/unified_contracts.parquet'\n"
            "unified_labels_path = PROJECT_ROOT / 'data/curated/unified_labels.parquet'\n"
        ),
        nbf.v4.new_code_cell(
            "matrix = pd.read_parquet(decision_matrix_path)\n"
            "unified_contracts = pd.read_parquet(unified_contracts_path)\n"
            "unified_labels = pd.read_parquet(unified_labels_path)\n"
            "label_report = json.loads(label_report_path.read_text(encoding='utf-8'))\n"
            "split_stats = json.loads(split_stats_path.read_text(encoding='utf-8'))\n"
            "dedup_report = json.loads(dedup_report_path.read_text(encoding='utf-8'))\n"
            "runtime_report = json.loads(runtime_report_path.read_text(encoding='utf-8'))\n\n"
            "print('SWCs in matrix:', matrix['swc_id'].nunique())\n"
            "print('Known labels:', label_report['label_distribution']['known_rows'])\n"
            "print('Primary contracts:', split_stats['contracts_with_known_labels'])\n"
            "print('Unified contracts:', dedup_report['counts']['unified_contract_rows_out'])\n"
            "print('Runtime extraction success rate:', runtime_report['overall']['success_rate'])"
        ),
        nbf.v4.new_code_cell(
            "primary_stats = split_stats['primary_split']['label_stats']\n"
            "rows = []\n"
            "for swc in matrix['swc_id'].astype(int).tolist():\n"
            "    key = str(swc)\n"
            "    train = primary_stats['train'].get(key, {'known': 0, 'positive': 0, 'negative': 0})\n"
            "    val = primary_stats['val'].get(key, {'known': 0, 'positive': 0, 'negative': 0})\n"
            "    test = primary_stats['test'].get(key, {'known': 0, 'positive': 0, 'negative': 0})\n"
            "    rows.append({\n"
            "        'swc_id': swc,\n"
            "        'contracts_in_primary_split': train['known'] + val['known'] + test['known'],\n"
            "        'primary_positive': train['positive'] + val['positive'] + test['positive'],\n"
            "        'primary_negative': train['negative'] + val['negative'] + test['negative'],\n"
            "    })\n"
            "primary_df = pd.DataFrame(rows)\n"
            "audit_df = matrix.merge(primary_df, on='swc_id', how='left').fillna(0)\n"
            "audit_df[['contracts_in_primary_split','primary_positive','primary_negative']] = (\n"
            "    audit_df[['contracts_in_primary_split','primary_positive','primary_negative']].astype(int)\n"
            ")\n"
            "audit_df = audit_df.rename(columns={'known_total': 'known_total_after_harmonization'})\n"
            "\n"
            "def recommend(row):\n"
            "    if row['action'] == 'drop_low_count' or int(row['known_total_after_harmonization']) < 20:\n"
            "        return (\n"
            "            'drop_low_support',\n"
            "            'Low known-label support after harmonization (below 20), unstable for benchmarking.'\n"
            "        )\n"
            "    if int(row['contracts_in_primary_split']) >= 100 and min(int(row['primary_positive']), int(row['primary_negative'])) >= 10:\n"
            "        return (\n"
            "            'keep_for_main_benchmark',\n"
            "            f\"Strong primary support (known={int(row['contracts_in_primary_split'])}, positive={int(row['primary_positive'])}, negative={int(row['primary_negative'])}).\"\n"
            "        )\n"
            "    return (\n"
            "        'keep_for_auxiliary_only',\n"
            "        f\"Retained with limited/imbalanced primary evidence (known={int(row['contracts_in_primary_split'])}, positive={int(row['primary_positive'])}, negative={int(row['primary_negative'])}).\"\n"
            "    )\n"
            "\n"
            "audit_df[['recommended_action', 'recommendation_reason']] = audit_df.apply(recommend, axis=1, result_type='expand')\n"
            "audit_df[['swc_id','cgt_positive','cgt_negative','dappscan_positive','known_total_after_harmonization','contracts_in_primary_split','recommended_action']]\n"
            "    .sort_values('swc_id')\n"
            "    .reset_index(drop=True)\n"
        ),
        nbf.v4.new_code_cell(
            "output_dir = PROJECT_ROOT / 'reports/phase1'\n"
            "output_dir.mkdir(parents=True, exist_ok=True)\n"
            "audit_df.to_csv(output_dir / 'final_swc_recommendation.csv', index=False)\n"
            "matrix.to_csv(output_dir / 'swc_decision_matrix.csv', index=False)\n"
            "unified_contracts.head(100).to_csv(output_dir / 'unified_contracts_head.csv', index=False)\n"
            "unified_labels.sample(n=min(500, len(unified_labels)), random_state=42).to_csv(\n"
            "    output_dir / 'unified_labels_sample.csv', index=False\n"
            ")\n"
            "print('Exported CSV artifacts to reports/phase1/')"
        ),
    ]
    nbf.write(notebook, path)


def _primary_split_counts(split_stats: Dict[str, Any]) -> pd.DataFrame:
    label_stats = split_stats.get("primary_split", {}).get("label_stats", {})
    train = label_stats.get("train", {})
    val = label_stats.get("val", {})
    test = label_stats.get("test", {})
    all_swcs = sorted({*train.keys(), *val.keys(), *test.keys()}, key=lambda s: int(s))

    rows: List[Dict[str, int]] = []
    for swc_key in all_swcs:
        t = train.get(swc_key, {"known": 0, "positive": 0, "negative": 0})
        v = val.get(swc_key, {"known": 0, "positive": 0, "negative": 0})
        te = test.get(swc_key, {"known": 0, "positive": 0, "negative": 0})
        rows.append(
            {
                "swc_id": int(swc_key),
                "contracts_in_primary_split": int(t["known"] + v["known"] + te["known"]),
                "primary_positive": int(t["positive"] + v["positive"] + te["positive"]),
                "primary_negative": int(t["negative"] + v["negative"] + te["negative"]),
            }
        )
    return pd.DataFrame(rows)


def _recommend_action(row: pd.Series) -> Tuple[str, str]:
    original = str(row["action"])
    known = int(row["contracts_in_primary_split"])
    pos = int(row["primary_positive"])
    neg = int(row["primary_negative"])
    known_total = int(row["known_total_after_harmonization"])

    if original == "drop_low_count" or known_total < 20:
        return (
            "drop_low_support",
            "Low known-label support after harmonization (below 20), unstable for benchmarking.",
        )

    if known >= MAIN_MIN_KNOWN and min(pos, neg) >= MAIN_MIN_MINORITY_CLASS:
        return (
            "keep_for_main_benchmark",
            f"Strong primary support (known={known}, positive={pos}, negative={neg}).",
        )

    return (
        "keep_for_auxiliary_only",
        f"Retained with limited/imbalanced primary evidence (known={known}, positive={pos}, negative={neg}).",
    )


def _markdown_table(df: pd.DataFrame, columns: List[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, record in df[columns].iterrows():
        rows.append("| " + " | ".join(str(record[col]) for col in columns) + " |")
    return "\n".join([header, sep, *rows])


def _build_summary_markdown(
    audit_df: pd.DataFrame,
    label_report: Dict[str, Any],
    split_stats: Dict[str, Any],
    dedup_report: Dict[str, Any],
    runtime_report: Dict[str, Any],
    source_note: str,
) -> str:
    generated_at = datetime.now(timezone.utc).isoformat()
    semantics = label_report.get("semantics_policy", {})
    label_dist = label_report.get("label_distribution", {})
    dedup_counts = dedup_report.get("counts", {})
    runtime_overall = runtime_report.get("overall", {})
    runtime_lengths = runtime_report.get("runtime_length_stats", {})

    swc_columns = [
        "swc_id",
        "cgt_positive",
        "cgt_negative",
        "dappscan_positive",
        "known_total_after_harmonization",
        "contracts_in_primary_split",
        "recommended_action",
    ]
    swc_table = _markdown_table(audit_df.sort_values("swc_id"), swc_columns)

    main_df = audit_df[audit_df["recommended_action"] == "keep_for_main_benchmark"].sort_values("swc_id")
    aux_df = audit_df[audit_df["recommended_action"] == "keep_for_auxiliary_only"].sort_values("swc_id")
    drop_df = audit_df[audit_df["recommended_action"] == "drop_low_support"].sort_values("swc_id")

    def _swc_bullets(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "- None"
        lines = []
        for _, row in frame.iterrows():
            lines.append(f"- SWC-{int(row['swc_id'])}: {row['recommendation_reason']}")
        return "\n".join(lines)

    main_ids = ", ".join(f"SWC-{int(v)}" for v in main_df["swc_id"].tolist()) or "None"
    aux_ids = ", ".join(f"SWC-{int(v)}" for v in aux_df["swc_id"].tolist()) or "None"

    top_failure_modes = runtime_report.get("top_failure_modes", [])
    top_failure_text = "none"
    if top_failure_modes:
        top_failure_text = ", ".join(
            f"{item.get('failure_mode', '<unknown>')} ({item.get('count', 0)})"
            for item in top_failure_modes[:5]
        )

    return (
        "# Phase 1 Audit Summary\n\n"
        f"Generated at (UTC): `{generated_at}`\n\n"
        f"Instruction source used for this pass: `{source_note}` (no `instructions.md` file was present).\n\n"
        "## Input artifact snapshot\n\n"
        f"- Label harmonization verdict: `{semantics.get('verdict', 'unknown')}` "
        f"(DAppSCAN negatives allowed: `{semantics.get('negatives_allowed_for_dappscan', 'unknown')}`)\n"
        f"- Known labels after harmonization: `{label_dist.get('known_rows', 0)}` "
        f"(positive: `{label_dist.get('positive_rows', 0)}`, negative: `{label_dist.get('negative_rows', 0)}`)\n"
        f"- Unified contracts after dedup: `{dedup_counts.get('unified_contract_rows_out', 0)}` "
        f"(shared fingerprints: `{dedup_counts.get('shared_fp_runtime_unified', 0)}`)\n"
        f"- Contracts in primary split universe: `{split_stats.get('contracts_with_known_labels', 0)}`\n"
        f"- Runtime extraction success: `{runtime_overall.get('success_count', 0)}`/"
        f"`{runtime_overall.get('processed_initcodes', 0)}` "
        f"(rate `{runtime_overall.get('success_rate', 0)}`)\n"
        f"- Runtime byte length median/p95: `{runtime_lengths.get('median', 0)}`/`{runtime_lengths.get('p95', 0)}`\n"
        f"- Top runtime extraction failure modes: {top_failure_text}\n\n"
        "## SWC-level audit table\n\n"
        f"{swc_table}\n\n"
        "## Final recommendations\n\n"
        f"### Main benchmark SWCs\n\n{main_ids}\n\n"
        f"{_swc_bullets(main_df)}\n\n"
        f"### Auxiliary-only SWCs\n\n{aux_ids}\n\n"
        f"{_swc_bullets(aux_df)}\n\n"
        "### Dropped SWCs (low support)\n\n"
        f"{_swc_bullets(drop_df)}\n\n"
        "### DAppSCAN usage policy\n\n"
        "DAppSCAN should be used only as positive augmentation in Phase 1/Phase 2 modeling because "
        "the harmonization policy is `POS_ONLY` and DAppSCAN negatives are not treated as reliable true negatives.\n"
    )


def run_phase1_audit(
    swc_decision_matrix_parquet: Path,
    label_report_json: Path,
    split_stats_json: Path,
    dedup_report_json: Path,
    runtime_extraction_report_json: Path,
    unified_contracts_parquet: Path,
    unified_labels_parquet: Path,
    summary_md_out: Path,
    final_recommendation_csv_out: Path,
    swc_decision_matrix_csv_out: Path,
    unified_contracts_head_csv_out: Path,
    unified_labels_sample_csv_out: Path,
    notebook_out: Path,
    unified_contracts_head_n: int = 100,
    unified_labels_sample_n: int = 500,
    sample_seed: int = 42,
) -> Dict[str, Any]:
    matrix = pd.read_parquet(swc_decision_matrix_parquet).copy()
    label_report = _read_json(label_report_json)
    split_stats = _read_json(split_stats_json)
    dedup_report = _read_json(dedup_report_json)
    runtime_report = _read_json(runtime_extraction_report_json)
    unified_contracts = pd.read_parquet(unified_contracts_parquet)
    unified_labels = pd.read_parquet(unified_labels_parquet)

    matrix["swc_id"] = pd.to_numeric(matrix["swc_id"], errors="coerce").astype("Int64")
    matrix = matrix[matrix["swc_id"].notna()].copy()
    matrix["swc_id"] = matrix["swc_id"].astype(int)

    primary_counts = _primary_split_counts(split_stats)
    audit_df = matrix.merge(primary_counts, on="swc_id", how="left").fillna(0)
    for col in ["contracts_in_primary_split", "primary_positive", "primary_negative"]:
        audit_df[col] = pd.to_numeric(audit_df[col], errors="coerce").fillna(0).astype(int)

    audit_df = audit_df.rename(columns={"known_total": "known_total_after_harmonization"})
    recommendations = audit_df.apply(_recommend_action, axis=1, result_type="expand")
    recommendations.columns = ["recommended_action", "recommendation_reason"]
    audit_df = pd.concat([audit_df, recommendations], axis=1)
    audit_df = audit_df.sort_values("swc_id").reset_index(drop=True)

    summary_text = _build_summary_markdown(
        audit_df=audit_df,
        label_report=label_report,
        split_stats=split_stats,
        dedup_report=dedup_report,
        runtime_report=runtime_report,
        source_note=".github/copilot-instructions.md",
    )
    _write_text(summary_md_out, summary_text)

    final_columns = [
        "swc_id",
        "cgt_positive",
        "cgt_negative",
        "dappscan_positive",
        "dappscan_negative",
        "known_total_after_harmonization",
        "contracts_in_primary_split",
        "primary_positive",
        "primary_negative",
        "action",
        "recommended_action",
        "recommendation_reason",
    ]
    audit_df[final_columns].to_csv(final_recommendation_csv_out, index=False)
    matrix.to_csv(swc_decision_matrix_csv_out, index=False)
    unified_contracts.head(max(1, int(unified_contracts_head_n))).to_csv(
        unified_contracts_head_csv_out, index=False
    )
    sample_n = min(max(1, int(unified_labels_sample_n)), len(unified_labels))
    unified_labels.sample(n=sample_n, random_state=int(sample_seed)).to_csv(
        unified_labels_sample_csv_out, index=False
    )
    _write_notebook(notebook_out)

    return {
        "swc_rows": int(len(audit_df)),
        "main_benchmark_swcs": int(
            (audit_df["recommended_action"] == "keep_for_main_benchmark").sum()
        ),
        "auxiliary_swcs": int((audit_df["recommended_action"] == "keep_for_auxiliary_only").sum()),
        "dropped_swcs": int((audit_df["recommended_action"] == "drop_low_support").sum()),
        "outputs": {
            "summary_md": _rel(summary_md_out),
            "final_recommendation_csv": _rel(final_recommendation_csv_out),
            "swc_decision_matrix_csv": _rel(swc_decision_matrix_csv_out),
            "unified_contracts_head_csv": _rel(unified_contracts_head_csv_out),
            "unified_labels_sample_csv": _rel(unified_labels_sample_csv_out),
            "notebook": _rel(notebook_out),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1 audit pass and export recommendations.")
    parser.add_argument("--swc-decision-matrix-parquet", type=Path, default=DEFAULT_SWC_DECISION_MATRIX_PARQUET)
    parser.add_argument("--label-report-json", type=Path, default=DEFAULT_LABEL_REPORT_JSON)
    parser.add_argument("--split-stats-json", type=Path, default=DEFAULT_SPLIT_STATS_JSON)
    parser.add_argument("--dedup-report-json", type=Path, default=DEFAULT_DEDUP_REPORT_JSON)
    parser.add_argument(
        "--runtime-extraction-report-json",
        type=Path,
        default=DEFAULT_RUNTIME_EXTRACTION_REPORT_JSON,
    )
    parser.add_argument("--unified-contracts-parquet", type=Path, default=DEFAULT_UNIFIED_CONTRACTS_PARQUET)
    parser.add_argument("--unified-labels-parquet", type=Path, default=DEFAULT_UNIFIED_LABELS_PARQUET)
    parser.add_argument("--summary-md-out", type=Path, default=DEFAULT_AUDIT_SUMMARY_MD)
    parser.add_argument("--final-recommendation-csv-out", type=Path, default=DEFAULT_FINAL_RECOMMENDATION_CSV)
    parser.add_argument("--swc-decision-matrix-csv-out", type=Path, default=DEFAULT_SWC_DECISION_MATRIX_CSV)
    parser.add_argument("--unified-contracts-head-csv-out", type=Path, default=DEFAULT_UNIFIED_CONTRACTS_HEAD_CSV)
    parser.add_argument("--unified-labels-sample-csv-out", type=Path, default=DEFAULT_UNIFIED_LABELS_SAMPLE_CSV)
    parser.add_argument("--notebook-out", type=Path, default=DEFAULT_AUDIT_NOTEBOOK)
    parser.add_argument("--unified-contracts-head-n", type=int, default=100)
    parser.add_argument("--unified-labels-sample-n", type=int, default=500)
    parser.add_argument("--sample-seed", type=int, default=42)
    args = parser.parse_args()

    result = run_phase1_audit(
        swc_decision_matrix_parquet=args.swc_decision_matrix_parquet,
        label_report_json=args.label_report_json,
        split_stats_json=args.split_stats_json,
        dedup_report_json=args.dedup_report_json,
        runtime_extraction_report_json=args.runtime_extraction_report_json,
        unified_contracts_parquet=args.unified_contracts_parquet,
        unified_labels_parquet=args.unified_labels_parquet,
        summary_md_out=args.summary_md_out,
        final_recommendation_csv_out=args.final_recommendation_csv_out,
        swc_decision_matrix_csv_out=args.swc_decision_matrix_csv_out,
        unified_contracts_head_csv_out=args.unified_contracts_head_csv_out,
        unified_labels_sample_csv_out=args.unified_labels_sample_csv_out,
        notebook_out=args.notebook_out,
        unified_contracts_head_n=args.unified_contracts_head_n,
        unified_labels_sample_n=args.unified_labels_sample_n,
        sample_seed=args.sample_seed,
    )
    print(f"SWC rows audited: {result['swc_rows']}")
    print(
        "Main/Aux/Drop: "
        f"{result['main_benchmark_swcs']}/{result['auxiliary_swcs']}/{result['dropped_swcs']}"
    )
    print(f"Summary: {args.summary_md_out}")
    print(f"Recommendation CSV: {args.final_recommendation_csv_out}")
    print(f"Notebook: {args.notebook_out}")


if __name__ == "__main__":
    main()
