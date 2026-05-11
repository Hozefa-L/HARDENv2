import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import nbformat as nbf
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _counter_to_sorted_dict(counter: Counter, numeric_keys: bool = False) -> Dict[str, int]:
    items = list(counter.items())
    if numeric_keys:
        items.sort(key=lambda kv: (-kv[1], int(kv[0])))
    else:
        items.sort(key=lambda kv: (-kv[1], str(kv[0])))
    return {str(k): int(v) for k, v in items}


def _find_metadata_workbook(dappscan_root: Path) -> Tuple[Optional[Path], str]:
    requested = dappscan_root / "DApp_list.xlsx"
    if requested.exists():
        return requested, "DApp_list.xlsx"

    requested_alt = dappscan_root / "DApp_list.xls"
    if requested_alt.exists():
        return requested_alt, "DApp_list.xls"

    fallback = dappscan_root / "Audit_and_Repository_link.xlsx"
    if fallback.exists():
        return fallback, "Audit_and_Repository_link.xlsx"

    glob_match = next(iter(dappscan_root.glob("**/*DApp*list*.xlsx")), None)
    if glob_match:
        return glob_match, glob_match.name
    return None, ""


def _compute_bytecode_inventory(bytecode_root: Path, dappscan_bytecode_root: Path) -> Dict[str, Any]:
    bytecode_projects = sorted([p.name for p in bytecode_root.iterdir() if p.is_dir()])
    bytecode_projects_set = set(bytecode_projects)

    bin_file_count = int(sum(1 for _ in dappscan_bytecode_root.rglob("*.bin")))
    bytecode_json_files = [p for p in bytecode_root.rglob("*.json") if p.is_file()]

    per_dapp_json_files = Counter()
    per_dapp_contract_entries = Counter()
    per_dapp_nonempty_contract_entries = Counter()
    bytecode_lengths_chars = []
    bytecode_parse_errors = 0

    for json_path in bytecode_json_files:
        rel = json_path.relative_to(bytecode_root)
        dapp_project = rel.parts[0] if rel.parts else "<ROOT>"
        per_dapp_json_files[dapp_project] += 1
        try:
            obj = json.loads(json_path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            bytecode_parse_errors += 1
            continue
        contracts = obj.get("contracts", {})
        if not isinstance(contracts, dict):
            continue

        for entry in contracts.values():
            if not isinstance(entry, dict):
                continue
            per_dapp_contract_entries[dapp_project] += 1
            bin_hex = entry.get("bin", "")
            if not isinstance(bin_hex, str):
                continue
            bin_hex = bin_hex.strip()
            if bin_hex.startswith("0x"):
                bin_hex = bin_hex[2:]
            if bin_hex:
                per_dapp_nonempty_contract_entries[dapp_project] += 1
                bytecode_lengths_chars.append(len(bin_hex))

    lengths_arr = np.array(bytecode_lengths_chars, dtype=np.int64)
    if len(lengths_arr) > 0:
        hist_counts, hist_edges = np.histogram(lengths_arr, bins=40)
        histogram = [
            {
                "left": int(hist_edges[i]),
                "right": int(hist_edges[i + 1]),
                "count": int(hist_counts[i]),
            }
            for i in range(len(hist_counts))
        ]
        length_stats = {
            "count": int(len(lengths_arr)),
            "min_chars": int(lengths_arr.min()),
            "max_chars": int(lengths_arr.max()),
            "mean_chars": float(np.round(lengths_arr.mean(), 4)),
            "median_chars": float(np.round(float(np.median(lengths_arr)), 4)),
            "p95_chars": float(np.round(float(np.percentile(lengths_arr, 95)), 4)),
            "mean_bytes": float(np.round(lengths_arr.mean() / 2.0, 4)),
            "median_bytes": float(np.round(float(np.median(lengths_arr)) / 2.0, 4)),
            "p95_bytes": float(np.round(float(np.percentile(lengths_arr, 95)) / 2.0, 4)),
            "histogram": histogram,
        }
    else:
        length_stats = {
            "count": 0,
            "min_chars": 0,
            "max_chars": 0,
            "mean_chars": 0.0,
            "median_chars": 0.0,
            "p95_chars": 0.0,
            "mean_bytes": 0.0,
            "median_bytes": 0.0,
            "p95_bytes": 0.0,
            "histogram": [],
        }

    return {
        "project_count": int(len(bytecode_projects)),
        "projects": bytecode_projects,
        "project_set": bytecode_projects_set,
        "bin_file_count": bin_file_count,
        "bytecode_json_file_count": int(len(bytecode_json_files)),
        "bytecode_json_parse_errors": int(bytecode_parse_errors),
        "total_contract_entries": int(sum(per_dapp_contract_entries.values())),
        "nonempty_bytecode_entries": int(sum(per_dapp_nonempty_contract_entries.values())),
        "per_dapp_json_file_counts": _counter_to_sorted_dict(per_dapp_json_files),
        "per_dapp_contract_counts": _counter_to_sorted_dict(per_dapp_contract_entries),
        "per_dapp_nonempty_contract_counts": _counter_to_sorted_dict(
            per_dapp_nonempty_contract_entries
        ),
        "bytecode_length_distribution": length_stats,
    }


def _compute_swc_findings(swc_root: Path) -> Dict[str, Any]:
    swc_projects = sorted([p.name for p in swc_root.iterdir() if p.is_dir()])
    swc_report_files = [p for p in swc_root.rglob("*.json") if p.is_file()]

    swc_id_counts = Counter()
    swc_category_counts = Counter()
    findings_per_dapp = Counter()
    report_files_per_dapp = Counter()
    parse_errors = 0
    findings_without_id = 0
    total_findings = 0

    for report_path in swc_report_files:
        rel = report_path.relative_to(swc_root)
        dapp_project = rel.parts[0] if rel.parts else "<ROOT>"
        report_files_per_dapp[dapp_project] += 1
        try:
            obj = json.loads(report_path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            parse_errors += 1
            continue

        findings = obj.get("SWCs", [])
        if not isinstance(findings, list):
            continue
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            total_findings += 1
            findings_per_dapp[dapp_project] += 1
            category = str(finding.get("category", "")).strip()
            if category:
                swc_category_counts[category] += 1
            match = re.search(r"SWC-(\d+)", category)
            if match:
                swc_id_counts[match.group(1)] += 1
            else:
                findings_without_id += 1

    return {
        "project_count": int(len(swc_projects)),
        "projects": swc_projects,
        "swc_report_json_file_count": int(len(swc_report_files)),
        "swc_report_parse_errors": int(parse_errors),
        "total_findings": int(total_findings),
        "findings_without_swc_id": int(findings_without_id),
        "swc_counts": _counter_to_sorted_dict(swc_id_counts, numeric_keys=True),
        "swc_category_counts": _counter_to_sorted_dict(swc_category_counts),
        "findings_per_dapp": _counter_to_sorted_dict(findings_per_dapp),
        "report_files_per_dapp": _counter_to_sorted_dict(report_files_per_dapp),
    }


def _compute_metadata(
    dappscan_root: Path, bytecode_projects: set
) -> Dict[str, Any]:
    workbook_path, selected_name = _find_metadata_workbook(dappscan_root)
    if workbook_path is None:
        return {
            "available": False,
            "requested_file": "DApp_list.xlsx",
            "used_file": None,
            "selection": None,
            "parse_error": "No metadata workbook found.",
            "row_count": 0,
            "columns": [],
            "audit_company_distribution": {},
            "coverage": {
                "metadata_project_count": 0,
                "bytecode_projects_with_metadata": 0,
                "coverage_rate": 0.0,
            },
        }

    try:
        xls = pd.ExcelFile(workbook_path)
        sheet_name = xls.sheet_names[0]
        meta_df = pd.read_excel(workbook_path, sheet_name=sheet_name)
    except Exception as exc:
        return {
            "available": False,
            "requested_file": "DApp_list.xlsx",
            "used_file": str(workbook_path.relative_to(PROJECT_ROOT)),
            "selection": selected_name,
            "parse_error": str(exc),
            "row_count": 0,
            "columns": [],
            "audit_company_distribution": {},
            "coverage": {
                "metadata_project_count": 0,
                "bytecode_projects_with_metadata": 0,
                "coverage_rate": 0.0,
            },
        }

    meta_df = meta_df.copy()
    meta_df.columns = [str(c).strip() for c in meta_df.columns]
    lower_map = {c.lower(): c for c in meta_df.columns}

    file_col = None
    for target in ["file name", "filename", "file_name", "dapp", "dapp name"]:
        if target in lower_map:
            file_col = lower_map[target]
            break
    if file_col is None:
        for col in meta_df.columns:
            if "file" in col.lower() and "name" in col.lower():
                file_col = col
                break

    audit_company_col = None
    for col in meta_df.columns:
        if "audit" in col.lower() and "company" in col.lower():
            audit_company_col = col
            break

    project_col = None
    for col in meta_df.columns:
        if "project" in col.lower() and "name" in col.lower():
            project_col = col
            break

    metadata_projects = set()
    if file_col is not None:
        metadata_projects = set(
            meta_df[file_col]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", np.nan)
            .dropna()
            .tolist()
        )

    bytecode_projects_with_metadata = len(bytecode_projects & metadata_projects)
    metadata_project_count = len(metadata_projects)
    coverage_rate = (
        float(np.round(bytecode_projects_with_metadata / len(bytecode_projects), 6))
        if bytecode_projects
        else 0.0
    )

    audit_dist = {}
    if audit_company_col is not None:
        audit_series = (
            meta_df[audit_company_col]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", "<MISSING>")
        )
        audit_dist = {
            str(k): int(v)
            for k, v in audit_series.value_counts(dropna=False).items()
        }

    project_name_dist = {}
    if project_col is not None:
        proj_series = (
            meta_df[project_col]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", "<MISSING>")
        )
        # Keep top 30 to keep summary compact.
        project_name_dist = {
            str(k): int(v)
            for k, v in proj_series.value_counts(dropna=False).head(30).items()
        }

    return {
        "available": True,
        "requested_file": "DApp_list.xlsx",
        "used_file": str(workbook_path.relative_to(PROJECT_ROOT)),
        "selection": selected_name,
        "sheet_name": sheet_name,
        "row_count": int(len(meta_df)),
        "columns": list(meta_df.columns),
        "audit_company_distribution": audit_dist,
        "project_name_distribution_top30": project_name_dist,
        "coverage": {
            "metadata_project_count": int(metadata_project_count),
            "bytecode_projects_with_metadata": int(bytecode_projects_with_metadata),
            "coverage_rate": coverage_rate,
        },
    }


def create_dappscan_notebook(notebook_path: Path, summary_path: Path) -> None:
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    rel_summary = str(summary_path.relative_to(PROJECT_ROOT))

    cells = [
        nbf.v4.new_markdown_cell(
            "# DAppSCAN-bytecode EDA (Phase 1)\n"
            "This notebook visualizes the inventory summary exported to JSON."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "\n"
            f"SUMMARY_REL = Path('{rel_summary}')\n"
            "PROJECT_ROOT = Path.cwd().resolve()\n"
            "if not (PROJECT_ROOT / SUMMARY_REL).exists():\n"
            "    PROJECT_ROOT = PROJECT_ROOT.parent\n"
            "summary_path = PROJECT_ROOT / SUMMARY_REL\n"
            "with summary_path.open('r', encoding='utf-8') as fp:\n"
            "    summary = json.load(fp)\n"
            "summary_path"
        ),
        nbf.v4.new_code_cell(
            "inventory = summary['inventory']\n"
            "pd.DataFrame([\n"
            "    {'metric': 'dapp_projects_in_bytecode', 'value': inventory['dapp_projects_in_bytecode']},\n"
            "    {'metric': 'dapp_projects_in_swc_reports', 'value': inventory['dapp_projects_in_swc_reports']},\n"
            "    {'metric': 'bin_files_count', 'value': inventory['bin_files_count']},\n"
            "    {'metric': 'bytecode_json_file_count', 'value': inventory['bytecode_json_file_count']},\n"
            "    {'metric': 'total_contract_entries', 'value': inventory['total_contract_entries']},\n"
            "    {'metric': 'nonempty_bytecode_entries', 'value': inventory['nonempty_bytecode_entries']},\n"
            "])"
        ),
        nbf.v4.new_code_cell(
            "swc_counts = pd.Series(summary['swc_findings']['swc_counts'], name='count')\n"
            "swc_df = swc_counts.rename_axis('swc_id').reset_index()\n"
            "swc_df['swc_id'] = swc_df['swc_id'].astype(int)\n"
            "swc_df = swc_df.sort_values('count', ascending=False)\n"
            "swc_df.head(20)"
        ),
        nbf.v4.new_code_cell(
            "top_swc = swc_df.head(20)\n"
            "plt.figure(figsize=(12, 6))\n"
            "plt.bar(top_swc['swc_id'].astype(str), top_swc['count'])\n"
            "plt.title('SWC Findings Counts (Top 20)')\n"
            "plt.xlabel('SWC ID')\n"
            "plt.ylabel('Findings')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        nbf.v4.new_code_cell(
            "per_dapp = pd.Series(summary['bytecode_inventory']['per_dapp_nonempty_contract_counts'], name='contracts')\n"
            "per_dapp_df = per_dapp.rename_axis('dapp').reset_index().sort_values('contracts', ascending=False)\n"
            "per_dapp_df.head(30)"
        ),
        nbf.v4.new_code_cell(
            "top_dapp = per_dapp_df.head(30).iloc[::-1]\n"
            "plt.figure(figsize=(12, 10))\n"
            "plt.barh(top_dapp['dapp'], top_dapp['contracts'])\n"
            "plt.title('Per-DApp Contract Counts (non-empty bytecode, Top 30)')\n"
            "plt.xlabel('Contract count')\n"
            "plt.ylabel('DApp project')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        nbf.v4.new_code_cell(
            "hist = pd.DataFrame(summary['bytecode_inventory']['bytecode_length_distribution']['histogram'])\n"
            "hist.head()"
        ),
        nbf.v4.new_code_cell(
            "if len(hist) > 0:\n"
            "    centers = (hist['left'] + hist['right']) / 2\n"
            "    widths = (hist['right'] - hist['left']).clip(lower=1)\n"
            "    plt.figure(figsize=(12, 6))\n"
            "    plt.bar(centers, hist['count'], width=widths)\n"
            "    plt.title('Bytecode Length Distribution (non-empty bin, hex chars)')\n"
            "    plt.xlabel('Bytecode length (hex chars)')\n"
            "    plt.ylabel('Frequency')\n"
            "    plt.tight_layout()\n"
            "    plt.show()\n"
            "else:\n"
            "    print('No non-empty bytecode lengths available.')"
        ),
        nbf.v4.new_code_cell(
            "metadata = summary['metadata']\n"
            "pd.DataFrame([\n"
            "    {'metric': 'metadata_available', 'value': metadata.get('available')},\n"
            "    {'metric': 'used_file', 'value': metadata.get('used_file')},\n"
            "    {'metric': 'rows', 'value': metadata.get('row_count', 0)},\n"
            "    {'metric': 'metadata_project_count', 'value': metadata.get('coverage', {}).get('metadata_project_count', 0)},\n"
            "    {'metric': 'bytecode_projects_with_metadata', 'value': metadata.get('coverage', {}).get('bytecode_projects_with_metadata', 0)},\n"
            "    {'metric': 'coverage_rate', 'value': metadata.get('coverage', {}).get('coverage_rate', 0.0)},\n"
            "])"
        ),
        nbf.v4.new_code_cell(
            "audit_dist = metadata.get('audit_company_distribution', {})\n"
            "if audit_dist:\n"
            "    audit_df = pd.Series(audit_dist, name='count').rename_axis('audit_company').reset_index()\n"
            "    audit_df = audit_df.sort_values('count', ascending=False).head(20)\n"
            "    plt.figure(figsize=(12, 8))\n"
            "    plt.barh(audit_df['audit_company'].iloc[::-1], audit_df['count'].iloc[::-1])\n"
            "    plt.title('Metadata Distribution: Audit Company (Top 20)')\n"
            "    plt.xlabel('Count')\n"
            "    plt.ylabel('Audit company')\n"
            "    plt.tight_layout()\n"
            "    plt.show()\n"
            "else:\n"
            "    print('No metadata distribution available.')"
        ),
    ]

    notebook = nbf.v4.new_notebook(cells=cells)
    nbf.write(notebook, notebook_path)


def run_dappscan_eda(dappscan_root: Path, notebook_out: Path, summary_out: Path) -> Dict[str, Any]:
    dappscan_bytecode_root = dappscan_root / "DAppSCAN-bytecode"
    bytecode_root = dappscan_bytecode_root / "bytecode"
    swc_root = dappscan_bytecode_root / "SWCbytecode"

    bytecode_inventory = _compute_bytecode_inventory(
        bytecode_root=bytecode_root, dappscan_bytecode_root=dappscan_bytecode_root
    )
    swc_findings = _compute_swc_findings(swc_root=swc_root)
    metadata = _compute_metadata(
        dappscan_root=dappscan_root, bytecode_projects=bytecode_inventory["project_set"]
    )

    summary = {
        "dataset": "DAppSCAN-bytecode",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "paths": {
            "dappscan_root": str(dappscan_root.relative_to(PROJECT_ROOT)),
            "bytecode_root": str(bytecode_root.relative_to(PROJECT_ROOT)),
            "swc_reports_root": str(swc_root.relative_to(PROJECT_ROOT)),
        },
        "inventory": {
            "dapp_projects_in_bytecode": int(bytecode_inventory["project_count"]),
            "dapp_projects_in_swc_reports": int(swc_findings["project_count"]),
            "bin_files_count": int(bytecode_inventory["bin_file_count"]),
            "bytecode_json_file_count": int(bytecode_inventory["bytecode_json_file_count"]),
            "total_contract_entries": int(bytecode_inventory["total_contract_entries"]),
            "nonempty_bytecode_entries": int(bytecode_inventory["nonempty_bytecode_entries"]),
            "note": (
                "DAppSCAN-bytecode stores compiled artifacts in JSON files with `contracts.*.bin`; "
                "physical `.bin` files may be absent."
            ),
        },
        "bytecode_inventory": {
            "per_dapp_json_file_counts": bytecode_inventory["per_dapp_json_file_counts"],
            "per_dapp_contract_counts": bytecode_inventory["per_dapp_contract_counts"],
            "per_dapp_nonempty_contract_counts": bytecode_inventory[
                "per_dapp_nonempty_contract_counts"
            ],
            "bytecode_length_distribution": bytecode_inventory[
                "bytecode_length_distribution"
            ],
            "bytecode_json_parse_errors": int(bytecode_inventory["bytecode_json_parse_errors"]),
        },
        "swc_findings": swc_findings,
        "metadata": metadata,
    }

    # Remove non-serializable helper set.
    bytecode_inventory.pop("project_set", None)

    summary_out.parent.mkdir(parents=True, exist_ok=True)
    with summary_out.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    create_dappscan_notebook(notebook_out, summary_out)
    return summary

