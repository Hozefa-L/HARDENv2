import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import nbformat as nbf
import pandas as pd

from .cgt_curation import curate_cgt
from .dappscan_eda import run_dappscan_eda
from .dappscan_labels import run_dappscan_label_extraction
from .dappscan_label_semantics import generate_dappscan_label_semantics_report
from .evm_runtime_extract import run_runtime_extraction
from .fingerprint import run_fingerprint_pipeline
from .label_harmonize import run_label_harmonization
from .merge_dedup import run_merge_dedup
from .phase1_audit import run_phase1_audit
from .splits import run_splits
from .swc_select import run_swc_selection


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV_PATH = PROJECT_ROOT / "data/raw/cgt-main/consolidated.csv"
DEFAULT_RUNTIME_DIR = PROJECT_ROOT / "data/raw/cgt-main/runtime"
DEFAULT_NOTEBOOK_PATH = PROJECT_ROOT / "notebooks/01a_cgt_eda.ipynb"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "reports/phase1/cgt_eda_summary.json"
DEFAULT_CONTRACTS_OUT = PROJECT_ROOT / "data/intermediate/cgt_contracts.parquet"
DEFAULT_LABELS_OUT = PROJECT_ROOT / "data/intermediate/cgt_labels.parquet"
DEFAULT_CURATION_REPORT_OUT = PROJECT_ROOT / "reports/phase1/cgt_curation_report.json"
DEFAULT_DAPPSCAN_ROOT = PROJECT_ROOT / "data/raw/dappscan"
DEFAULT_DAPPSCAN_NOTEBOOK_PATH = PROJECT_ROOT / "notebooks/01b_dappscan_eda.ipynb"
DEFAULT_DAPPSCAN_SUMMARY_PATH = PROJECT_ROOT / "reports/phase1/dappscan_eda_summary.json"
DEFAULT_DAPPSCAN_SEMANTICS_PATH = PROJECT_ROOT / "reports/phase1/dappscan_label_semantics.md"
DEFAULT_DAPPSCAN_LABELS_OUT = PROJECT_ROOT / "data/intermediate/dappscan_labels.parquet"
DEFAULT_DAPPSCAN_CONTRACTS_IN = PROJECT_ROOT / "data/intermediate/dappscan_contracts.parquet"
DEFAULT_CGT_CONTRACTS_FP_OUT = PROJECT_ROOT / "data/intermediate/cgt_contracts_fp.parquet"
DEFAULT_DAPPSCAN_CONTRACTS_FP_OUT = PROJECT_ROOT / "data/intermediate/dappscan_contracts_fp.parquet"
DEFAULT_FINGERPRINT_SPEC_OUT = PROJECT_ROOT / "reports/phase1/fingerprint_spec.md"
DEFAULT_FINGERPRINT_REPORT_OUT = PROJECT_ROOT / "reports/phase1/fingerprint_report.json"
DEFAULT_RUNTIME_EXTRACTION_REPORT_PATH = PROJECT_ROOT / "reports/phase1/runtime_extraction_report.json"
DEFAULT_RUNTIME_EXTRACTION_RECORDS_PATH = PROJECT_ROOT / "checkpoints/runtime_extraction_records.jsonl"
DEFAULT_UNIFIED_CONTRACTS_OUT = PROJECT_ROOT / "data/curated/unified_contracts.parquet"
DEFAULT_DEDUP_REPORT_OUT = PROJECT_ROOT / "reports/phase1/dedup_report.json"
DEFAULT_DEDUP_REPORT_OUT_COPY = PROJECT_ROOT / "data/curated/dedup_report.json"
DEFAULT_UNIFIED_LABELS_OUT = PROJECT_ROOT / "data/curated/unified_labels.parquet"
DEFAULT_DISAGREEMENT_CASES_OUT = PROJECT_ROOT / "data/curated/disagreement_cases.parquet"
DEFAULT_LABEL_REPORT_OUT = PROJECT_ROOT / "reports/phase1/label_report.json"
DEFAULT_DISAGREEMENT_SUMMARY_OUT = PROJECT_ROOT / "reports/phase1/disagreement_summary.json"
DEFAULT_SWC_DECISION_MATRIX_OUT = PROJECT_ROOT / "reports/phase1/swc_decision_matrix.parquet"
DEFAULT_SWC_SELECTION_REPORT_OUT = PROJECT_ROOT / "reports/phase1/swc_selection_report.json"
DEFAULT_SPLITS_ROOT = PROJECT_ROOT / "data/splits"
DEFAULT_SPLIT_STATS_OUT = PROJECT_ROOT / "reports/phase1/split_stats.json"
DEFAULT_PHASE1_AUDIT_SUMMARY_OUT = PROJECT_ROOT / "reports/phase1/phase1_audit_summary.md"
DEFAULT_PHASE1_FINAL_RECOMMENDATION_CSV_OUT = (
    PROJECT_ROOT / "reports/phase1/final_swc_recommendation.csv"
)
DEFAULT_SWC_DECISION_MATRIX_CSV_OUT = PROJECT_ROOT / "reports/phase1/swc_decision_matrix.csv"
DEFAULT_UNIFIED_CONTRACTS_HEAD_CSV_OUT = PROJECT_ROOT / "reports/phase1/unified_contracts_head.csv"
DEFAULT_UNIFIED_LABELS_SAMPLE_CSV_OUT = PROJECT_ROOT / "reports/phase1/unified_labels_sample.csv"
DEFAULT_PHASE1_AUDIT_NOTEBOOK_OUT = PROJECT_ROOT / "notebooks/01d_phase1_audit.ipynb"
DEFAULT_CGT_CONTRACTS_FP_IN = PROJECT_ROOT / "data/intermediate/cgt_contracts_fp.parquet"
DEFAULT_DAPPSCAN_CONTRACTS_FP_IN = PROJECT_ROOT / "data/intermediate/dappscan_contracts_fp.parquet"
DEFAULT_CGT_LABELS_IN = PROJECT_ROOT / "data/intermediate/cgt_labels.parquet"
DEFAULT_DAPPSCAN_LABELS_IN = PROJECT_ROOT / "data/intermediate/dappscan_labels.parquet"


def _clean(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _distribution(series: pd.Series, missing_label: str = "<MISSING>") -> Dict[str, int]:
    cleaned = _clean(series).replace("", missing_label)
    counts = cleaned.value_counts(dropna=False)
    return {str(k): int(v) for k, v in counts.items()}


def _runtime_hashes(runtime_dir: Path) -> set:
    hashes = set()
    for file_path in runtime_dir.glob("*"):
        if not file_path.is_file():
            continue
        name = file_path.name
        if name.endswith(".rt.hex"):
            hashes.add(name[:-7])
        else:
            hashes.add(file_path.stem)
    return hashes


def compute_summary(df: pd.DataFrame, runtime_dir: Path, csv_path: Path) -> Dict[str, Any]:
    total_entries = int(len(df))

    fp_runtime = _clean(df["fp_runtime"])
    swc = _clean(df["swc"]).replace("", "<MISSING>")
    dasp = _clean(df["dasp"]).replace("", "<MISSING>")

    non_empty_fp_runtime = fp_runtime != ""
    unique_fp_runtime_values = fp_runtime[non_empty_fp_runtime].unique()
    unique_contracts_by_fp_runtime = int(len(unique_fp_runtime_values))

    runtime_hashes = _runtime_hashes(runtime_dir)
    has_runtime_artifact = fp_runtime.isin(runtime_hashes)

    rows_with_fp_runtime = int(non_empty_fp_runtime.sum())
    rows_without_fp_runtime = int((~non_empty_fp_runtime).sum())
    rows_with_runtime_artifact = int((non_empty_fp_runtime & has_runtime_artifact).sum())
    rows_missing_runtime_artifact = int((non_empty_fp_runtime & ~has_runtime_artifact).sum())

    unique_fp_runtime_with_artifact = int(
        len(fp_runtime[non_empty_fp_runtime & has_runtime_artifact].unique())
    )
    unique_fp_runtime_missing_artifact = int(
        len(fp_runtime[non_empty_fp_runtime & ~has_runtime_artifact].unique())
    )

    swc_distribution = _distribution(df["swc"])
    dasp_distribution = _distribution(df["dasp"])

    swc_balance = []
    for swc_label, count in swc.value_counts().items():
        percentage = (float(count) / total_entries * 100.0) if total_entries else 0.0
        swc_balance.append(
            {
                "swc": str(swc_label),
                "count": int(count),
                "percentage": round(percentage, 4),
            }
        )

    field_parsing = {}
    for column in df.columns:
        col = _clean(df[column])
        non_empty = int((col != "").sum())
        empty = int((col == "").sum())
        field_parsing[column] = {
            "dtype": str(df[column].dtype),
            "non_empty_count": non_empty,
            "empty_count": empty,
            "unique_non_empty": int(col[col != ""].nunique()),
        }

    runtime_availability = {
        "runtime_files_on_disk": int(sum(1 for p in runtime_dir.glob("*") if p.is_file())),
        "rows_with_fp_runtime": rows_with_fp_runtime,
        "rows_without_fp_runtime": rows_without_fp_runtime,
        "rows_with_runtime_artifact": rows_with_runtime_artifact,
        "rows_missing_runtime_artifact": rows_missing_runtime_artifact,
        "row_availability_rate": round(
            rows_with_runtime_artifact / rows_with_fp_runtime, 6
        )
        if rows_with_fp_runtime
        else 0.0,
        "unique_fp_runtime_total": unique_contracts_by_fp_runtime,
        "unique_fp_runtime_with_artifact": unique_fp_runtime_with_artifact,
        "unique_fp_runtime_missing_artifact": unique_fp_runtime_missing_artifact,
    }

    return {
        "dataset": "CGT",
        "source_csv": str(csv_path.relative_to(PROJECT_ROOT)),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_entries": total_entries,
        "unique_contracts_by_fp_runtime": unique_contracts_by_fp_runtime,
        "swc_distribution": swc_distribution,
        "dasp_distribution": dasp_distribution,
        "runtime_artifact_availability": runtime_availability,
        "class_balance_per_swc": swc_balance,
        "parsed_fields": list(df.columns),
        "field_parsing": field_parsing,
    }


def create_notebook(notebook_path: Path) -> None:
    notebook_path.parent.mkdir(parents=True, exist_ok=True)

    md_cell = nbf.v4.new_markdown_cell(
        "# CGT EDA (Phase 1)\n"
        "This notebook reads `data/raw/cgt-main/consolidated.csv`, parses all fields, "
        "computes requested summary metrics, and visualizes SWC/DASP/runtime availability."
    )

    load_cell = nbf.v4.new_code_cell(
        "from pathlib import Path\n"
        "import json\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "CSV_REL = Path('data/raw/cgt-main/consolidated.csv')\n"
        "RUNTIME_REL = Path('data/raw/cgt-main/runtime')\n"
        "SUMMARY_REL = Path('reports/phase1/cgt_eda_summary.json')\n"
        "\n"
        "PROJECT_ROOT = Path.cwd().resolve()\n"
        "if not (PROJECT_ROOT / CSV_REL).exists():\n"
        "    PROJECT_ROOT = PROJECT_ROOT.parent\n"
        "\n"
        "csv_path = PROJECT_ROOT / CSV_REL\n"
        "runtime_dir = PROJECT_ROOT / RUNTIME_REL\n"
        "summary_path = PROJECT_ROOT / SUMMARY_REL\n"
        "\n"
        "df = pd.read_csv(csv_path, sep=';', dtype='string', keep_default_na=False)\n"
        "print(f'Rows: {len(df):,}')\n"
        "print(f'Columns ({len(df.columns)}): {list(df.columns)}')"
    )

    fields_cell = nbf.v4.new_code_cell(
        "def clean(s):\n"
        "    return s.fillna('').astype(str).str.strip()\n"
        "\n"
        "field_profile = []\n"
        "for c in df.columns:\n"
        "    series = clean(df[c])\n"
        "    field_profile.append({\n"
        "        'column': c,\n"
        "        'non_empty_count': int((series != '').sum()),\n"
        "        'empty_count': int((series == '').sum()),\n"
        "        'unique_non_empty': int(series[series != ''].nunique()),\n"
        "    })\n"
        "pd.DataFrame(field_profile)"
    )

    metrics_cell = nbf.v4.new_code_cell(
        "swc_counts = clean(df['swc']).replace('', '<MISSING>').value_counts().rename_axis('swc').reset_index(name='count')\n"
        "dasp_counts = clean(df['dasp']).replace('', '<MISSING>').value_counts().rename_axis('dasp').reset_index(name='count')\n"
        "fp_runtime = clean(df['fp_runtime'])\n"
        "non_empty_fp = fp_runtime != ''\n"
        "\n"
        "runtime_hashes = set()\n"
        "for p in runtime_dir.glob('*'):\n"
        "    if p.is_file() and p.name.endswith('.rt.hex'):\n"
        "        runtime_hashes.add(p.name[:-7])\n"
        "\n"
        "artifact_exists = fp_runtime.isin(runtime_hashes)\n"
        "availability = pd.DataFrame([\n"
        "    {'metric': 'rows_with_fp_runtime', 'value': int(non_empty_fp.sum())},\n"
        "    {'metric': 'rows_without_fp_runtime', 'value': int((~non_empty_fp).sum())},\n"
        "    {'metric': 'rows_with_runtime_artifact', 'value': int((non_empty_fp & artifact_exists).sum())},\n"
        "    {'metric': 'rows_missing_runtime_artifact', 'value': int((non_empty_fp & ~artifact_exists).sum())},\n"
        "    {'metric': 'unique_contracts_by_fp_runtime', 'value': int(fp_runtime[non_empty_fp].nunique())},\n"
        "])\n"
        "availability"
    )

    swc_plot_cell = nbf.v4.new_code_cell(
        "top_swc = swc_counts.sort_values('count', ascending=False).head(25)\n"
        "plt.figure(figsize=(12, 8))\n"
        "plt.barh(top_swc['swc'].astype(str), top_swc['count'])\n"
        "plt.title('SWC Distribution (Top 25)')\n"
        "plt.xlabel('Count')\n"
        "plt.ylabel('SWC')\n"
        "plt.gca().invert_yaxis()\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )

    dasp_plot_cell = nbf.v4.new_code_cell(
        "top_dasp = dasp_counts.sort_values('count', ascending=False)\n"
        "plt.figure(figsize=(10, 6))\n"
        "plt.bar(top_dasp['dasp'].astype(str), top_dasp['count'])\n"
        "plt.title('DASP Distribution')\n"
        "plt.xlabel('DASP')\n"
        "plt.ylabel('Count')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )

    runtime_plot_cell = nbf.v4.new_code_cell(
        "runtime_plot = availability[availability['metric'].isin([\n"
        "    'rows_with_fp_runtime',\n"
        "    'rows_with_runtime_artifact',\n"
        "    'rows_missing_runtime_artifact',\n"
        "])]\n"
        "plt.figure(figsize=(10, 5))\n"
        "plt.bar(runtime_plot['metric'], runtime_plot['value'])\n"
        "plt.title('Runtime Artifact Availability (Row-level)')\n"
        "plt.ylabel('Count')\n"
        "plt.xticks(rotation=20)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )

    class_balance_cell = nbf.v4.new_code_cell(
        "swc_balance = swc_counts.copy()\n"
        "swc_balance['percentage'] = (swc_balance['count'] / len(df) * 100).round(4)\n"
        "swc_balance"
    )

    export_cell = nbf.v4.new_code_cell(
        "summary = {\n"
        "    'total_entries': int(len(df)),\n"
        "    'unique_contracts_by_fp_runtime': int(fp_runtime[non_empty_fp].nunique()),\n"
        "    'swc_distribution': {str(k): int(v) for k, v in swc_counts.set_index('swc')['count'].to_dict().items()},\n"
        "    'dasp_distribution': {str(k): int(v) for k, v in dasp_counts.set_index('dasp')['count'].to_dict().items()},\n"
        "    'runtime_artifact_availability': availability.set_index('metric')['value'].to_dict(),\n"
        "    'class_balance_per_swc': swc_balance.to_dict(orient='records'),\n"
        "}\n"
        "summary_path.parent.mkdir(parents=True, exist_ok=True)\n"
        "with summary_path.open('w', encoding='utf-8') as f:\n"
        "    json.dump(summary, f, indent=2)\n"
        "summary_path"
    )

    notebook = nbf.v4.new_notebook(
        cells=[
            md_cell,
            load_cell,
            fields_cell,
            metrics_cell,
            swc_plot_cell,
            dasp_plot_cell,
            runtime_plot_cell,
            class_balance_cell,
            export_cell,
        ]
    )
    nbf.write(notebook, notebook_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="CGT EDA and curation utilities.")
    parser.add_argument(
        "--task",
        choices=[
            "eda",
            "curate",
            "dappscan-eda",
            "dappscan-label-semantics",
                "dappscan-labels",
                "fingerprint",
                "runtime-extract",
                "merge-dedup",
                "label-harmonize",
                "swc-select",
                "splits",
                "phase1-audit",
            ],
            default="eda",
        )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    parser.add_argument("--notebook-out", type=Path, default=DEFAULT_NOTEBOOK_PATH)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--contracts-out", type=Path, default=DEFAULT_CONTRACTS_OUT)
    parser.add_argument("--labels-out", type=Path, default=DEFAULT_LABELS_OUT)
    parser.add_argument("--curation-report-out", type=Path, default=DEFAULT_CURATION_REPORT_OUT)
    parser.add_argument("--candidate-swc-min", type=int, default=100)
    parser.add_argument("--candidate-swc-max", type=int, default=136)
    parser.add_argument("--dappscan-root", type=Path, default=DEFAULT_DAPPSCAN_ROOT)
    parser.add_argument(
        "--dappscan-notebook-out", type=Path, default=DEFAULT_DAPPSCAN_NOTEBOOK_PATH
    )
    parser.add_argument("--dappscan-summary-out", type=Path, default=DEFAULT_DAPPSCAN_SUMMARY_PATH)
    parser.add_argument(
        "--dappscan-semantics-out", type=Path, default=DEFAULT_DAPPSCAN_SEMANTICS_PATH
    )
    parser.add_argument("--dappscan-labels-out", type=Path, default=DEFAULT_DAPPSCAN_LABELS_OUT)
    parser.add_argument("--dappscan-contracts-in", type=Path, default=DEFAULT_DAPPSCAN_CONTRACTS_IN)
    parser.add_argument("--cgt-contracts-fp-out", type=Path, default=DEFAULT_CGT_CONTRACTS_FP_OUT)
    parser.add_argument(
        "--dappscan-contracts-fp-out", type=Path, default=DEFAULT_DAPPSCAN_CONTRACTS_FP_OUT
    )
    parser.add_argument("--fingerprint-spec-out", type=Path, default=DEFAULT_FINGERPRINT_SPEC_OUT)
    parser.add_argument("--fingerprint-report-out", type=Path, default=DEFAULT_FINGERPRINT_REPORT_OUT)
    parser.add_argument(
        "--runtime-extraction-report-out",
        type=Path,
        default=DEFAULT_RUNTIME_EXTRACTION_REPORT_PATH,
    )
    parser.add_argument(
        "--runtime-extraction-records-out",
        type=Path,
        default=DEFAULT_RUNTIME_EXTRACTION_RECORDS_PATH,
    )
    parser.add_argument("--runtime-min-len", type=int, default=100)
    parser.add_argument("--runtime-max-failure-samples", type=int, default=30)
    parser.add_argument("--fingerprint-metadata-mode", choices=["strip", "zero"], default="strip")
    parser.add_argument("--stub-threshold-bytes", type=int, default=100)
    parser.add_argument("--delegatecall-proxy-threshold", type=float, default=0.02)
    parser.add_argument("--dappscan-min-runtime-len", type=int, default=1)
    parser.add_argument("--sample-dapps", type=int, default=20)
    parser.add_argument("--contracts-per-dapp", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cgt-contracts-fp", type=Path, default=DEFAULT_CGT_CONTRACTS_FP_IN)
    parser.add_argument("--dappscan-contracts-fp", type=Path, default=DEFAULT_DAPPSCAN_CONTRACTS_FP_IN)
    parser.add_argument("--cgt-labels-in", type=Path, default=DEFAULT_CGT_LABELS_IN)
    parser.add_argument("--dappscan-labels-in", type=Path, default=DEFAULT_DAPPSCAN_LABELS_IN)
    parser.add_argument("--unified-contracts-out", type=Path, default=DEFAULT_UNIFIED_CONTRACTS_OUT)
    parser.add_argument("--dedup-report-out", type=Path, default=DEFAULT_DEDUP_REPORT_OUT)
    parser.add_argument("--dedup-report-out-copy", type=Path, default=DEFAULT_DEDUP_REPORT_OUT_COPY)
    parser.add_argument("--unified-labels-out", type=Path, default=DEFAULT_UNIFIED_LABELS_OUT)
    parser.add_argument("--disagreement-cases-out", type=Path, default=DEFAULT_DISAGREEMENT_CASES_OUT)
    parser.add_argument("--label-report-out", type=Path, default=DEFAULT_LABEL_REPORT_OUT)
    parser.add_argument("--disagreement-summary-out", type=Path, default=DEFAULT_DISAGREEMENT_SUMMARY_OUT)
    parser.add_argument("--swc-decision-matrix-out", type=Path, default=DEFAULT_SWC_DECISION_MATRIX_OUT)
    parser.add_argument("--swc-selection-report-out", type=Path, default=DEFAULT_SWC_SELECTION_REPORT_OUT)
    parser.add_argument("--swc-candidates", type=str, default=None)
    parser.add_argument("--swc-min-total-known", type=int, default=20)
    parser.add_argument("--splits-root", type=Path, default=DEFAULT_SPLITS_ROOT)
    parser.add_argument("--split-stats-out", type=Path, default=DEFAULT_SPLIT_STATS_OUT)
    parser.add_argument("--phase1-audit-summary-out", type=Path, default=DEFAULT_PHASE1_AUDIT_SUMMARY_OUT)
    parser.add_argument(
        "--phase1-final-recommendation-csv-out",
        type=Path,
        default=DEFAULT_PHASE1_FINAL_RECOMMENDATION_CSV_OUT,
    )
    parser.add_argument(
        "--swc-decision-matrix-csv-out",
        type=Path,
        default=DEFAULT_SWC_DECISION_MATRIX_CSV_OUT,
    )
    parser.add_argument(
        "--unified-contracts-head-csv-out",
        type=Path,
        default=DEFAULT_UNIFIED_CONTRACTS_HEAD_CSV_OUT,
    )
    parser.add_argument(
        "--unified-labels-sample-csv-out",
        type=Path,
        default=DEFAULT_UNIFIED_LABELS_SAMPLE_CSV_OUT,
    )
    parser.add_argument("--phase1-audit-notebook-out", type=Path, default=DEFAULT_PHASE1_AUDIT_NOTEBOOK_OUT)
    parser.add_argument("--unified-contracts-head-n", type=int, default=100)
    parser.add_argument("--unified-labels-sample-n", type=int, default=500)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--no-cv", action="store_true")
    args = parser.parse_args()

    if args.task == "dappscan-label-semantics":
        result = generate_dappscan_label_semantics_report(
            dappscan_root=args.dappscan_root,
            out_path=args.dappscan_semantics_out,
            sample_dapps=args.sample_dapps,
            contracts_per_dapp=args.contracts_per_dapp,
            seed=args.seed,
        )
        print(f"Verdict: {result['verdict']}")
        print(f"Allow negatives: {result['allow_negative_generation']}")
        print(f"Sampled reports: {result['sampled_reports']}")
        print(f"Report: {args.dappscan_semantics_out}")
        return

    if args.task == "dappscan-eda":
        summary = run_dappscan_eda(
            dappscan_root=args.dappscan_root,
            notebook_out=args.dappscan_notebook_out,
            summary_out=args.dappscan_summary_out,
        )
        print(f"DApp projects (bytecode): {summary['inventory']['dapp_projects_in_bytecode']}")
        print(f".bin files: {summary['inventory']['bin_files_count']}")
        print(f"SWC findings: {summary['swc_findings']['total_findings']}")
        print(f"Notebook: {args.dappscan_notebook_out}")
        print(f"Summary JSON: {args.dappscan_summary_out}")
        return

    if args.task == "dappscan-labels":
        summary = run_dappscan_label_extraction(
            dappscan_root=args.dappscan_root,
            labels_out=args.dappscan_labels_out,
            semantics_report_path=args.dappscan_semantics_out,
            sample_dapps=args.sample_dapps,
            contracts_per_dapp=args.contracts_per_dapp,
            seed=args.seed,
        )
        print(f"Semantics verdict: {summary['semantics']['verdict']}")
        print(f"Negatives allowed: {summary['semantics']['negatives_allowed']}")
        print(
            f"Label rows: {summary['counts']['rows']} "
            f"(+{summary['counts']['positive_rows']}, "
            f"-{summary['counts']['negative_rows']}, "
            f"unlabeled={summary['counts']['unlabeled_rows']})"
        )
        print(f"Labels parquet: {args.dappscan_labels_out}")
        return

    if args.task == "runtime-extract":
        report = run_runtime_extraction(
            dappscan_root=args.dappscan_root,
            report_out=args.runtime_extraction_report_out,
            records_out=args.runtime_extraction_records_out,
            min_runtime_len=args.runtime_min_len,
            max_failure_samples=args.runtime_max_failure_samples,
        )
        print(f"Processed initcodes: {report['overall']['processed_initcodes']}")
        print(f"Success count: {report['overall']['success_count']}")
        print(f"Success rate: {report['overall']['success_rate']}")
        print(f"Runtime report: {args.runtime_extraction_report_out}")
        print(f"Extraction records: {args.runtime_extraction_records_out}")
        return

    if args.task == "merge-dedup":
        report = run_merge_dedup(
            cgt_contracts_fp=args.cgt_contracts_fp,
            dappscan_contracts_fp=args.dappscan_contracts_fp,
            unified_contracts_out=args.unified_contracts_out,
            report_out=args.dedup_report_out,
            report_out_copy=args.dedup_report_out_copy,
        )
        print(f"Unified contracts: {report['counts']['unified_contract_rows_out']}")
        print(f"Shared fingerprints: {report['counts']['shared_fp_runtime_unified']}")
        print(f"Contracts parquet: {args.unified_contracts_out}")
        print(f"Dedup report: {args.dedup_report_out}")
        return

    if args.task == "label-harmonize":
        report = run_label_harmonization(
            cgt_contracts_fp_path=args.cgt_contracts_fp,
            dappscan_contracts_fp_path=args.dappscan_contracts_fp,
            cgt_labels_path=args.cgt_labels_in,
            dappscan_labels_path=args.dappscan_labels_in,
            semantics_report_path=args.dappscan_semantics_out,
            unified_labels_out=args.unified_labels_out,
            disagreement_cases_out=args.disagreement_cases_out,
            label_report_out=args.label_report_out,
            disagreement_summary_out=args.disagreement_summary_out,
        )
        print(f"Unified label rows: {report['label_distribution']['total_rows']}")
        print(f"Known labels: {report['label_distribution']['known_rows']}")
        print(f"Disagreement pairs: {report['disagreements']['pairs_with_conflicts']}")
        print(f"Unified labels parquet: {args.unified_labels_out}")
        return

    if args.task == "swc-select":
        report = run_swc_selection(
            unified_labels_path=args.unified_labels_out,
            decision_matrix_out=args.swc_decision_matrix_out,
            report_out=args.swc_selection_report_out,
            swc_candidates=args.swc_candidates,
            swc_min=args.candidate_swc_min,
            swc_max=args.candidate_swc_max,
            min_total_known=args.swc_min_total_known,
        )
        print(f"SWC candidates: {report['counts']['candidate_swcs']}")
        print(f"Decision matrix: {args.swc_decision_matrix_out}")
        return

    if args.task == "splits":
        report = run_splits(
            unified_labels_path=args.unified_labels_out,
            swc_decision_matrix_path=args.swc_decision_matrix_out,
            unified_contracts_path=args.unified_contracts_out,
            cgt_contracts_fp_path=args.cgt_contracts_fp,
            cgt_csv_path=args.csv,
            splits_root=args.splits_root,
            split_stats_out=args.split_stats_out,
            seed=args.seed,
            cv_folds=args.cv_folds,
            generate_cv=not args.no_cv,
        )
        print(f"Contracts with known labels: {report['contracts_with_known_labels']}")
        print(
            "Primary split train/val/test: "
            f"{report['primary_split']['train_size']}/"
            f"{report['primary_split']['val_size']}/"
            f"{report['primary_split']['test_size']}"
        )
        print(f"Splits root: {args.splits_root}")
        print(f"Split stats: {args.split_stats_out}")
        return

    if args.task == "phase1-audit":
        report = run_phase1_audit(
            swc_decision_matrix_parquet=args.swc_decision_matrix_out,
            label_report_json=args.label_report_out,
            split_stats_json=args.split_stats_out,
            dedup_report_json=args.dedup_report_out,
            runtime_extraction_report_json=args.runtime_extraction_report_out,
            unified_contracts_parquet=args.unified_contracts_out,
            unified_labels_parquet=args.unified_labels_out,
            summary_md_out=args.phase1_audit_summary_out,
            final_recommendation_csv_out=args.phase1_final_recommendation_csv_out,
            swc_decision_matrix_csv_out=args.swc_decision_matrix_csv_out,
            unified_contracts_head_csv_out=args.unified_contracts_head_csv_out,
            unified_labels_sample_csv_out=args.unified_labels_sample_csv_out,
            notebook_out=args.phase1_audit_notebook_out,
            unified_contracts_head_n=args.unified_contracts_head_n,
            unified_labels_sample_n=args.unified_labels_sample_n,
            sample_seed=args.seed,
        )
        print(
            f"Main/Aux/Drop SWCs: {report['main_benchmark_swcs']}/"
            f"{report['auxiliary_swcs']}/{report['dropped_swcs']}"
        )
        print(f"Audit summary: {args.phase1_audit_summary_out}")
        print(f"Recommendation CSV: {args.phase1_final_recommendation_csv_out}")
        print(f"Notebook: {args.phase1_audit_notebook_out}")
        return

    if args.task == "fingerprint":
        report = run_fingerprint_pipeline(
            cgt_contracts_in=args.contracts_out,
            dappscan_contracts_in=args.dappscan_contracts_in,
            cgt_runtime_dir=args.runtime_dir,
            dappscan_root=args.dappscan_root,
            cgt_out=args.cgt_contracts_fp_out,
            dappscan_out=args.dappscan_contracts_fp_out,
            report_out=args.fingerprint_report_out,
            spec_out=args.fingerprint_spec_out,
            metadata_mode=args.fingerprint_metadata_mode,
            stub_threshold_bytes=args.stub_threshold_bytes,
            delegatecall_proxy_threshold=args.delegatecall_proxy_threshold,
            dappscan_min_runtime_len=args.dappscan_min_runtime_len,
        )
        print(f"Hash algorithm: {report['hash_algorithm']}")
        print(
            f"CGT proxy/stub: {report['datasets']['cgt']['flags']['proxy_like_rows']}/"
            f"{report['datasets']['cgt']['flags']['stub_like_rows']}"
        )
        print(
            f"DAppSCAN proxy/stub: {report['datasets']['dappscan']['flags']['proxy_like_rows']}/"
            f"{report['datasets']['dappscan']['flags']['stub_like_rows']}"
        )
        print(f"CGT FP parquet: {args.cgt_contracts_fp_out}")
        print(f"DAppSCAN FP parquet: {args.dappscan_contracts_fp_out}")
        print(f"Fingerprint report: {args.fingerprint_report_out}")
        print(f"Fingerprint spec: {args.fingerprint_spec_out}")
        return

    if args.task == "curate":
        report = curate_cgt(
            csv_path=args.csv,
            runtime_dir=args.runtime_dir,
            contracts_out=args.contracts_out,
            labels_out=args.labels_out,
            report_out=args.curation_report_out,
            swc_min=args.candidate_swc_min,
            swc_max=args.candidate_swc_max,
        )
        print(f"Candidate SWCs: {len(report['candidate_swcs'])}")
        print(f"Contracts: {report['contracts']['total_contracts']}")
        print(f"Labels (long): {report['labels_long_format']['total_rows']}")
        print(f"Contracts parquet: {args.contracts_out}")
        print(f"Labels parquet: {args.labels_out}")
        print(f"Curation report: {args.curation_report_out}")
        return

    df = pd.read_csv(args.csv, sep=";", dtype="string", keep_default_na=False)
    summary = compute_summary(df, args.runtime_dir, args.csv)

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_out.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    create_notebook(args.notebook_out)

    print(f"Total entries: {summary['total_entries']}")
    print(f"Unique contracts by fp_runtime: {summary['unique_contracts_by_fp_runtime']}")
    print(f"Notebook: {args.notebook_out}")
    print(f"Summary JSON: {args.summary_out}")


if __name__ == "__main__":
    main()
