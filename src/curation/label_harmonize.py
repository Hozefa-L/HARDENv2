import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CGT_CONTRACTS_FP = PROJECT_ROOT / "data/intermediate/cgt_contracts_fp.parquet"
DEFAULT_DAPPSCAN_CONTRACTS_FP = PROJECT_ROOT / "data/intermediate/dappscan_contracts_fp.parquet"
DEFAULT_CGT_LABELS = PROJECT_ROOT / "data/intermediate/cgt_labels.parquet"
DEFAULT_DAPPSCAN_LABELS = PROJECT_ROOT / "data/intermediate/dappscan_labels.parquet"
DEFAULT_SEMANTICS_REPORT = PROJECT_ROOT / "reports/phase1/dappscan_label_semantics.md"
DEFAULT_UNIFIED_LABELS_OUT = PROJECT_ROOT / "data/curated/unified_labels.parquet"
DEFAULT_DISAGREEMENT_CASES_OUT = PROJECT_ROOT / "data/curated/disagreement_cases.parquet"
DEFAULT_LABEL_REPORT_OUT = PROJECT_ROOT / "reports/phase1/label_report.json"
DEFAULT_DISAGREEMENT_SUMMARY_OUT = PROJECT_ROOT / "reports/phase1/disagreement_summary.json"


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        return str(path)


def _write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(out_path, index=False, engine="pyarrow")
        return
    except Exception as pyarrow_exc:
        try:
            df.to_parquet(out_path, index=False, engine="fastparquet")
            return
        except Exception:
            raise RuntimeError(
                "Unable to write parquet: neither pyarrow nor fastparquet is available."
            ) from pyarrow_exc


def _write_json(payload: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def _clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _read_semantics_verdict(semantics_report: Path) -> str:
    text = semantics_report.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"Verdict:\s*`([^`]+)`", text)
    if not match:
        raise ValueError(f"Could not parse verdict from semantics report: {semantics_report}")
    verdict = str(match.group(1)).strip()
    if verdict not in {"POS_ONLY", "POS+NEG_EXPLICIT"}:
        raise ValueError(f"Unsupported semantics verdict: {verdict}")
    return verdict


def _normalize_binary_label(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NA, index=series.index, dtype="Int64")
    out.loc[values == 1] = 1
    out.loc[values == 0] = 0
    return out


def _harmonize_cgt(
    cgt_labels: pd.DataFrame,
    cgt_contracts_fp: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if "swc_id" not in cgt_labels.columns:
        raise ValueError("CGT labels input missing required column: swc_id")
    if "fp_runtime" not in cgt_labels.columns:
        raise ValueError("CGT labels input missing required column: fp_runtime")
    if "fp_runtime" not in cgt_contracts_fp.columns or "fp_runtime_final" not in cgt_contracts_fp.columns:
        raise ValueError("CGT contracts FP input must include fp_runtime and fp_runtime_final")

    cgt = cgt_labels.copy()
    cgt["fp_runtime"] = _clean_text(cgt["fp_runtime"])
    cgt["swc_id"] = pd.to_numeric(cgt["swc_id"], errors="coerce").astype("Int64")

    if "property_holds" in cgt.columns:
        property_holds = _clean_text(cgt["property_holds"]).str.lower()
        label = pd.Series(pd.NA, index=cgt.index, dtype="Int64")
        label.loc[property_holds == "t"] = 1
        label.loc[property_holds == "f"] = 0
        policy_source = "property_holds"
    elif "property_holds_clean" in cgt.columns:
        property_holds = _clean_text(cgt["property_holds_clean"]).str.lower()
        label = pd.Series(pd.NA, index=cgt.index, dtype="Int64")
        label.loc[property_holds == "t"] = 1
        label.loc[property_holds == "f"] = 0
        policy_source = "property_holds_clean"
    elif "label" in cgt.columns:
        label = _normalize_binary_label(cgt["label"])
        policy_source = "label"
    else:
        raise ValueError("CGT labels input must include property_holds/property_holds_clean/label")
    cgt["label_harmonized"] = label

    cgt_fp_map = (
        cgt_contracts_fp[["fp_runtime", "fp_runtime_final"]]
        .copy()
        .assign(fp_runtime=lambda d: _clean_text(d["fp_runtime"]))
        .assign(fp_runtime_final=lambda d: _clean_text(d["fp_runtime_final"]))
        .drop_duplicates(subset=["fp_runtime"])
        .rename(columns={"fp_runtime_final": "fp_runtime_unified"})
    )
    cgt = cgt.merge(cgt_fp_map, on="fp_runtime", how="left")
    cgt["fp_runtime_unified"] = _clean_text(cgt["fp_runtime_unified"])

    cgt_out = cgt[["fp_runtime_unified", "swc_id", "label_harmonized", "fp_runtime"]].copy()
    cgt_out = cgt_out.rename(columns={"label_harmonized": "label", "fp_runtime": "source_identifier"})
    cgt_out["source"] = "cgt"
    cgt_out["label_confidence"] = pd.Series(pd.NA, index=cgt_out.index, dtype="object")
    cgt_out.loc[cgt_out["label"].notna(), "label_confidence"] = "hard_cgt"
    cgt_out = cgt_out[cgt_out["swc_id"].notna()].copy()
    cgt_out["swc_id"] = cgt_out["swc_id"].astype(int)
    cgt_out["label"] = cgt_out["label"].astype("Int64")

    summary = {
        "label_policy_source_column": policy_source,
        "rows_in": int(len(cgt_labels)),
        "rows_with_fp_mapping": int((cgt_out["fp_runtime_unified"] != "").sum()),
        "rows_missing_fp_mapping": int((cgt_out["fp_runtime_unified"] == "").sum()),
        "known_labels": int(cgt_out["label"].notna().sum()),
        "positive_labels": int((cgt_out["label"] == 1).sum()),
        "negative_labels": int((cgt_out["label"] == 0).sum()),
        "null_labels": int(cgt_out["label"].isna().sum()),
    }
    return cgt_out, summary


def _build_dappscan_fp_map(dappscan_contracts_fp: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, Any]]:
    if "dapp" not in dappscan_contracts_fp.columns or "contract_name" not in dappscan_contracts_fp.columns:
        raise ValueError("DAppSCAN contracts FP input missing dapp/contract_name columns")
    if "fp_runtime_final" not in dappscan_contracts_fp.columns:
        raise ValueError("DAppSCAN contracts FP input missing fp_runtime_final")

    contracts = dappscan_contracts_fp.copy()
    contracts["dapp"] = _clean_text(contracts["dapp"])
    contracts["contract_name"] = _clean_text(contracts["contract_name"])
    contracts["fp_runtime_final"] = _clean_text(contracts["fp_runtime_final"])
    contracts = contracts[contracts["fp_runtime_final"] != ""].copy()

    contracts["contract_id"] = contracts["dapp"] + "/" + contracts["contract_name"]
    exact = (
        contracts.groupby("contract_id")["fp_runtime_final"]
        .agg(lambda s: sorted(set(s.dropna().astype(str))))
        .to_dict()
    )
    exact_unique = {cid: fps[0] for cid, fps in exact.items() if len(fps) == 1}
    exact_ambiguous = {cid: fps for cid, fps in exact.items() if len(fps) > 1}

    contracts["contract_name_lower"] = contracts["contract_name"].str.lower()
    contracts["contract_id_lower"] = contracts["dapp"] + "/" + contracts["contract_name_lower"]
    lower = (
        contracts.groupby("contract_id_lower")["fp_runtime_final"]
        .agg(lambda s: sorted(set(s.dropna().astype(str))))
        .to_dict()
    )
    lower_unique = {cid: fps[0] for cid, fps in lower.items() if len(fps) == 1}

    mapping = dict(exact_unique)
    fallback_count = 0
    for lower_cid, fp in lower_unique.items():
        if lower_cid in mapping:
            continue
        fallback_count += 1
        mapping[lower_cid] = fp

    summary = {
        "contract_ids_with_unique_exact_fp": int(len(exact_unique)),
        "contract_ids_with_ambiguous_exact_fp": int(len(exact_ambiguous)),
        "contract_ids_with_unique_casefold_fp": int(len(lower_unique)),
        "casefold_entries_added": int(fallback_count),
    }
    return mapping, summary


def _harmonize_dappscan(
    dappscan_labels: pd.DataFrame,
    dappscan_contracts_fp: pd.DataFrame,
    negatives_allowed: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    required_columns = {"contract_id", "dapp", "contract_name", "swc_id"}
    missing = [column for column in sorted(required_columns) if column not in dappscan_labels.columns]
    if missing:
        raise ValueError(f"DAppSCAN labels input missing required columns: {missing}")

    fp_map, fp_map_stats = _build_dappscan_fp_map(dappscan_contracts_fp)

    dapp = dappscan_labels.copy()
    dapp["contract_id"] = _clean_text(dapp["contract_id"])
    dapp["dapp"] = _clean_text(dapp["dapp"])
    dapp["contract_name"] = _clean_text(dapp["contract_name"])
    dapp["swc_id"] = pd.to_numeric(dapp["swc_id"], errors="coerce").astype("Int64")
    dapp["contract_id_lower"] = dapp["dapp"] + "/" + dapp["contract_name"].str.lower()

    dapp["fp_runtime_unified"] = dapp["contract_id"].map(fp_map)
    missing_mask = dapp["fp_runtime_unified"].isna()
    dapp.loc[missing_mask, "fp_runtime_unified"] = dapp.loc[missing_mask, "contract_id_lower"].map(fp_map)
    dapp["fp_runtime_unified"] = _clean_text(dapp["fp_runtime_unified"])

    positive = pd.Series(False, index=dapp.index)
    if "label" in dapp.columns:
        positive = positive | (_normalize_binary_label(dapp["label"]) == 1)
    if "is_positive_evidence" in dapp.columns:
        positive = positive | dapp["is_positive_evidence"].fillna(False).astype(bool)
    if "positive_findings" in dapp.columns:
        positive = positive | (pd.to_numeric(dapp["positive_findings"], errors="coerce").fillna(0) > 0)

    explicit_negative = pd.Series(False, index=dapp.index)
    if "label" in dapp.columns:
        explicit_negative = explicit_negative | (_normalize_binary_label(dapp["label"]) == 0)
    if "is_negative_evidence" in dapp.columns:
        explicit_negative = explicit_negative | dapp["is_negative_evidence"].fillna(False).astype(bool)
    if "negative_findings" in dapp.columns:
        explicit_negative = explicit_negative | (
            pd.to_numeric(dapp["negative_findings"], errors="coerce").fillna(0) > 0
        )

    harmonized = pd.Series(pd.NA, index=dapp.index, dtype="Int64")
    harmonized.loc[positive & explicit_negative] = pd.NA
    harmonized.loc[positive & ~explicit_negative] = 1
    if negatives_allowed:
        harmonized.loc[explicit_negative & ~positive] = 0
    dapp["label"] = harmonized

    dapp_out = dapp[["fp_runtime_unified", "swc_id", "label", "contract_id"]].copy()
    dapp_out = dapp_out.rename(columns={"contract_id": "source_identifier"})
    dapp_out["source"] = "dappscan"
    dapp_out["label_confidence"] = pd.Series(pd.NA, index=dapp_out.index, dtype="object")
    dapp_out.loc[dapp_out["label"] == 1, "label_confidence"] = "pos_dappscan"
    dapp_out.loc[dapp_out["label"] == 0, "label_confidence"] = "explicit_neg_dappscan"
    dapp_out = dapp_out[dapp_out["swc_id"].notna()].copy()
    dapp_out["swc_id"] = dapp_out["swc_id"].astype(int)
    dapp_out["label"] = dapp_out["label"].astype("Int64")

    contract_id_stats = dapp[["contract_id", "contract_id_lower", "fp_runtime_unified"]].drop_duplicates()
    summary = {
        "rows_in": int(len(dappscan_labels)),
        "rows_with_fp_mapping": int((dapp_out["fp_runtime_unified"] != "").sum()),
        "rows_missing_fp_mapping": int((dapp_out["fp_runtime_unified"] == "").sum()),
        "known_labels": int(dapp_out["label"].notna().sum()),
        "positive_labels": int((dapp_out["label"] == 1).sum()),
        "negative_labels": int((dapp_out["label"] == 0).sum()),
        "null_labels": int(dapp_out["label"].isna().sum()),
        "contract_ids_total": int(contract_id_stats["contract_id"].nunique()),
        "contract_ids_with_fp_mapping": int((contract_id_stats["fp_runtime_unified"] != "").sum()),
        "contract_ids_missing_fp_mapping": int((contract_id_stats["fp_runtime_unified"] == "").sum()),
        "fp_mapping_stats": fp_map_stats,
    }
    return dapp_out, summary


def run_label_harmonization(
    cgt_contracts_fp_path: Path,
    dappscan_contracts_fp_path: Path,
    cgt_labels_path: Path,
    dappscan_labels_path: Path,
    semantics_report_path: Path,
    unified_labels_out: Path,
    disagreement_cases_out: Path,
    label_report_out: Path,
    disagreement_summary_out: Optional[Path] = DEFAULT_DISAGREEMENT_SUMMARY_OUT,
) -> Dict[str, Any]:
    verdict = _read_semantics_verdict(semantics_report_path)
    negatives_allowed = verdict == "POS+NEG_EXPLICIT"

    cgt_contracts_fp = pd.read_parquet(cgt_contracts_fp_path)
    dappscan_contracts_fp = pd.read_parquet(dappscan_contracts_fp_path)
    cgt_labels = pd.read_parquet(cgt_labels_path)
    dappscan_labels = pd.read_parquet(dappscan_labels_path)

    cgt_long, cgt_summary = _harmonize_cgt(cgt_labels, cgt_contracts_fp)
    dapp_long, dapp_summary = _harmonize_dappscan(
        dappscan_labels=dappscan_labels,
        dappscan_contracts_fp=dappscan_contracts_fp,
        negatives_allowed=negatives_allowed,
    )

    unified = pd.concat([cgt_long, dapp_long], ignore_index=True)
    unified["fp_runtime_unified"] = _clean_text(unified["fp_runtime_unified"])
    unified = unified[unified["fp_runtime_unified"] != ""].copy()
    unified = unified[["fp_runtime_unified", "swc_id", "label", "source", "label_confidence"]]
    unified["swc_id"] = pd.to_numeric(unified["swc_id"], errors="coerce").astype("Int64")
    unified["label"] = unified["label"].astype("Int64")
    unified = unified.dropna(subset=["swc_id"]).copy()
    unified["swc_id"] = unified["swc_id"].astype(int)
    unified = unified.sort_values(["fp_runtime_unified", "swc_id", "source"]).reset_index(drop=True)

    known = unified[unified["label"].notna()].copy()
    conflicting_keys = (
        known.groupby(["fp_runtime_unified", "swc_id"])
        .agg(label_nunique=("label", "nunique"), source_nunique=("source", "nunique"))
        .reset_index()
    )
    conflicting_keys = conflicting_keys[
        (conflicting_keys["label_nunique"] > 1) & (conflicting_keys["source_nunique"] > 1)
    ][["fp_runtime_unified", "swc_id"]].copy()
    disagreements = unified.merge(conflicting_keys, on=["fp_runtime_unified", "swc_id"], how="inner")
    disagreements = disagreements.sort_values(["fp_runtime_unified", "swc_id", "source"]).reset_index(
        drop=True
    )

    _write_parquet(unified, unified_labels_out)
    _write_parquet(disagreements, disagreement_cases_out)

    disagreement_summary = {
        "pairs_with_conflicts": int(len(conflicting_keys)),
        "rows_in_disagreement_cases": int(len(disagreements)),
        "contracts_with_conflicts": int(disagreements["fp_runtime_unified"].nunique())
        if not disagreements.empty
        else 0,
        "swcs_with_conflicts": int(disagreements["swc_id"].nunique()) if not disagreements.empty else 0,
    }
    if disagreement_summary_out:
        _write_json(disagreement_summary, disagreement_summary_out)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "semantics_policy": {
            "report_path": _rel(semantics_report_path),
            "verdict": verdict,
            "negatives_allowed_for_dappscan": bool(negatives_allowed),
        },
        "inputs": {
            "cgt_contracts_fp": _rel(cgt_contracts_fp_path),
            "dappscan_contracts_fp": _rel(dappscan_contracts_fp_path),
            "cgt_labels": _rel(cgt_labels_path),
            "dappscan_labels": _rel(dappscan_labels_path),
        },
        "source_summaries": {
            "cgt": cgt_summary,
            "dappscan": dapp_summary,
        },
        "outputs": {
            "unified_labels_parquet": _rel(unified_labels_out),
            "disagreement_cases_parquet": _rel(disagreement_cases_out),
            "label_report_json": _rel(label_report_out),
            "disagreement_summary_json": _rel(disagreement_summary_out)
            if disagreement_summary_out
            else None,
        },
        "label_distribution": {
            "total_rows": int(len(unified)),
            "known_rows": int(unified["label"].notna().sum()),
            "positive_rows": int((unified["label"] == 1).sum()),
            "negative_rows": int((unified["label"] == 0).sum()),
            "null_rows": int(unified["label"].isna().sum()),
            "by_source": {
                source: {
                    "rows": int(len(frame)),
                    "known_rows": int(frame["label"].notna().sum()),
                    "positive_rows": int((frame["label"] == 1).sum()),
                    "negative_rows": int((frame["label"] == 0).sum()),
                    "null_rows": int(frame["label"].isna().sum()),
                }
                for source, frame in unified.groupby("source")
            },
        },
        "disagreements": disagreement_summary,
    }
    _write_json(report, label_report_out)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Harmonize CGT and DAppSCAN labels on unified fingerprints.")
    parser.add_argument("--cgt-contracts-fp", type=Path, default=DEFAULT_CGT_CONTRACTS_FP)
    parser.add_argument("--dappscan-contracts-fp", type=Path, default=DEFAULT_DAPPSCAN_CONTRACTS_FP)
    parser.add_argument("--cgt-labels", type=Path, default=DEFAULT_CGT_LABELS)
    parser.add_argument("--dappscan-labels", type=Path, default=DEFAULT_DAPPSCAN_LABELS)
    parser.add_argument("--semantics-report", type=Path, default=DEFAULT_SEMANTICS_REPORT)
    parser.add_argument("--unified-labels-out", type=Path, default=DEFAULT_UNIFIED_LABELS_OUT)
    parser.add_argument("--disagreement-cases-out", type=Path, default=DEFAULT_DISAGREEMENT_CASES_OUT)
    parser.add_argument("--label-report-out", type=Path, default=DEFAULT_LABEL_REPORT_OUT)
    parser.add_argument("--disagreement-summary-out", type=Path, default=DEFAULT_DISAGREEMENT_SUMMARY_OUT)
    args = parser.parse_args()

    report = run_label_harmonization(
        cgt_contracts_fp_path=args.cgt_contracts_fp,
        dappscan_contracts_fp_path=args.dappscan_contracts_fp,
        cgt_labels_path=args.cgt_labels,
        dappscan_labels_path=args.dappscan_labels,
        semantics_report_path=args.semantics_report,
        unified_labels_out=args.unified_labels_out,
        disagreement_cases_out=args.disagreement_cases_out,
        label_report_out=args.label_report_out,
        disagreement_summary_out=args.disagreement_summary_out,
    )
    print(f"Unified label rows: {report['label_distribution']['total_rows']}")
    print(f"Known labels: {report['label_distribution']['known_rows']}")
    print(f"Conflicting pairs: {report['disagreements']['pairs_with_conflicts']}")
    print(f"Unified labels parquet: {args.unified_labels_out}")


if __name__ == "__main__":
    main()
