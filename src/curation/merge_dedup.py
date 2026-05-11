import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CGT_CONTRACTS_FP = PROJECT_ROOT / "data/intermediate/cgt_contracts_fp.parquet"
DEFAULT_DAPPSCAN_CONTRACTS_FP = PROJECT_ROOT / "data/intermediate/dappscan_contracts_fp.parquet"
DEFAULT_UNIFIED_CONTRACTS_OUT = PROJECT_ROOT / "data/curated/unified_contracts.parquet"
DEFAULT_REPORT_OUT = PROJECT_ROOT / "reports/phase1/dedup_report.json"
DEFAULT_REPORT_OUT_COPY = PROJECT_ROOT / "data/curated/dedup_report.json"


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


def _clean_text_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _json_list(values: Iterable[Any]) -> str:
    normalized = sorted(
        {
            str(v).strip()
            for v in values
            if v is not None and not pd.isna(v) and str(v).strip()
        }
    )
    return json.dumps(normalized)


def _first_nonempty(values: Iterable[Any]) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _max_int(values: Iterable[Any]) -> int:
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if series.empty:
        return 0
    return int(series.max())


def _any_bool(values: Iterable[Any]) -> bool:
    series = pd.Series(list(values)).fillna(False).astype(bool)
    return bool(series.any()) if not series.empty else False


def _aggregate_cgt(cgt_df: pd.DataFrame) -> pd.DataFrame:
    cgt = cgt_df.copy()
    cgt["fp_runtime_unified"] = _clean_text_series(cgt["fp_runtime_final"])
    cgt["cgt_fp_runtime"] = _clean_text_series(cgt["fp_runtime"]) if "fp_runtime" in cgt.columns else ""
    cgt = cgt[cgt["fp_runtime_unified"] != ""].copy()

    grouped: List[Dict[str, Any]] = []
    for fp_value, frame in cgt.groupby("fp_runtime_unified", sort=False):
        grouped.append(
            {
                "fp_runtime_unified": str(fp_value),
                "cgt_row_count": int(len(frame)),
                "cgt_fp_runtime_ids": _json_list(frame["cgt_fp_runtime"].tolist()),
                "runtime_bytecode_hex_normalized_cgt": _first_nonempty(
                    frame.get("runtime_bytecode_hex_normalized", pd.Series(dtype="object")).tolist()
                ),
                "runtime_size_bytes_cgt": _max_int(frame.get("runtime_size_bytes", pd.Series(dtype="float"))),
                "is_proxy_like_cgt": _any_bool(frame.get("is_proxy_like", pd.Series(dtype="bool"))),
                "is_stub_like_cgt": _any_bool(frame.get("is_stub_like", pd.Series(dtype="bool"))),
                "delegatecall_count_cgt": _max_int(
                    frame.get("delegatecall_count", pd.Series(dtype="float"))
                ),
                "opcode_count_cgt": _max_int(frame.get("opcode_count", pd.Series(dtype="float"))),
            }
        )
    out = pd.DataFrame(grouped)
    expected_columns = [
        "fp_runtime_unified",
        "cgt_row_count",
        "cgt_fp_runtime_ids",
        "runtime_bytecode_hex_normalized_cgt",
        "runtime_size_bytes_cgt",
        "is_proxy_like_cgt",
        "is_stub_like_cgt",
        "delegatecall_count_cgt",
        "opcode_count_cgt",
    ]
    for column in expected_columns:
        if column not in out.columns:
            out[column] = pd.Series(dtype="object")
    return out[expected_columns]


def _aggregate_dappscan(dappscan_df: pd.DataFrame) -> pd.DataFrame:
    dappscan = dappscan_df.copy()
    dappscan["fp_runtime_unified"] = _clean_text_series(dappscan["fp_runtime_final"])
    dappscan["dapp"] = _clean_text_series(dappscan["dapp"]) if "dapp" in dappscan.columns else ""
    dappscan["contract_name"] = (
        _clean_text_series(dappscan["contract_name"]) if "contract_name" in dappscan.columns else ""
    )
    dappscan["contract_key"] = (
        _clean_text_series(dappscan["contract_key"]) if "contract_key" in dappscan.columns else ""
    )
    dappscan["contract_id"] = (dappscan["dapp"] + "/" + dappscan["contract_name"]).str.strip("/")
    dappscan = dappscan[dappscan["fp_runtime_unified"] != ""].copy()

    grouped: List[Dict[str, Any]] = []
    for fp_value, frame in dappscan.groupby("fp_runtime_unified", sort=False):
        grouped.append(
            {
                "fp_runtime_unified": str(fp_value),
                "dappscan_row_count": int(len(frame)),
                "dappscan_contract_ids": _json_list(frame["contract_id"].tolist()),
                "dappscan_contract_keys": _json_list(frame["contract_key"].tolist()),
                "dappscan_dapps": _json_list(frame["dapp"].tolist()),
                "runtime_bytecode_hex_normalized_dappscan": _first_nonempty(
                    frame.get("runtime_bytecode_hex_normalized", pd.Series(dtype="object")).tolist()
                ),
                "runtime_size_bytes_dappscan": _max_int(
                    frame.get("runtime_size_bytes", pd.Series(dtype="float"))
                ),
                "is_proxy_like_dappscan": _any_bool(frame.get("is_proxy_like", pd.Series(dtype="bool"))),
                "is_stub_like_dappscan": _any_bool(frame.get("is_stub_like", pd.Series(dtype="bool"))),
                "delegatecall_count_dappscan": _max_int(
                    frame.get("delegatecall_count", pd.Series(dtype="float"))
                ),
                "opcode_count_dappscan": _max_int(frame.get("opcode_count", pd.Series(dtype="float"))),
            }
        )
    out = pd.DataFrame(grouped)
    expected_columns = [
        "fp_runtime_unified",
        "dappscan_row_count",
        "dappscan_contract_ids",
        "dappscan_contract_keys",
        "dappscan_dapps",
        "runtime_bytecode_hex_normalized_dappscan",
        "runtime_size_bytes_dappscan",
        "is_proxy_like_dappscan",
        "is_stub_like_dappscan",
        "delegatecall_count_dappscan",
        "opcode_count_dappscan",
    ]
    for column in expected_columns:
        if column not in out.columns:
            out[column] = pd.Series(dtype="object")
    return out[expected_columns]


def run_merge_dedup(
    cgt_contracts_fp: Path,
    dappscan_contracts_fp: Path,
    unified_contracts_out: Path,
    report_out: Path,
    report_out_copy: Optional[Path] = DEFAULT_REPORT_OUT_COPY,
) -> Dict[str, Any]:
    cgt_df = pd.read_parquet(cgt_contracts_fp)
    dappscan_df = pd.read_parquet(dappscan_contracts_fp)

    for required in ["fp_runtime_final"]:
        if required not in cgt_df.columns:
            raise ValueError(f"CGT contracts input missing required column: {required}")
        if required not in dappscan_df.columns:
            raise ValueError(f"DAppSCAN contracts input missing required column: {required}")

    cgt_agg = _aggregate_cgt(cgt_df)
    dappscan_agg = _aggregate_dappscan(dappscan_df)

    merged = cgt_agg.merge(dappscan_agg, on="fp_runtime_unified", how="outer")
    defaults = {
        "cgt_row_count": 0,
        "dappscan_row_count": 0,
        "cgt_fp_runtime_ids": "[]",
        "dappscan_contract_ids": "[]",
        "dappscan_contract_keys": "[]",
        "dappscan_dapps": "[]",
        "runtime_bytecode_hex_normalized_cgt": "",
        "runtime_bytecode_hex_normalized_dappscan": "",
        "runtime_size_bytes_cgt": 0,
        "runtime_size_bytes_dappscan": 0,
        "is_proxy_like_cgt": False,
        "is_proxy_like_dappscan": False,
        "is_stub_like_cgt": False,
        "is_stub_like_dappscan": False,
        "delegatecall_count_cgt": 0,
        "delegatecall_count_dappscan": 0,
        "opcode_count_cgt": 0,
        "opcode_count_dappscan": 0,
    }
    for column, default in defaults.items():
        if column not in merged.columns:
            merged[column] = default

    merged["cgt_row_count"] = merged["cgt_row_count"].fillna(0).astype(int)
    merged["dappscan_row_count"] = merged["dappscan_row_count"].fillna(0).astype(int)
    merged["has_cgt"] = merged["cgt_row_count"] > 0
    merged["has_dappscan"] = merged["dappscan_row_count"] > 0

    merged["sources"] = merged.apply(
        lambda row: json.dumps(
            [src for src, keep in (("cgt", bool(row["has_cgt"])), ("dappscan", bool(row["has_dappscan"]))) if keep]
        ),
        axis=1,
    )
    merged["source_count"] = merged["has_cgt"].astype(int) + merged["has_dappscan"].astype(int)

    merged["cgt_fp_runtime_ids"] = merged["cgt_fp_runtime_ids"].fillna("[]")
    merged["dappscan_contract_ids"] = merged["dappscan_contract_ids"].fillna("[]")
    merged["dappscan_contract_keys"] = merged["dappscan_contract_keys"].fillna("[]")
    merged["dappscan_dapps"] = merged["dappscan_dapps"].fillna("[]")

    def _pick_bytecode(row: pd.Series) -> str:
        """Select best available bytecode, preferring CGT over DAppSCAN.

        Handles NaN values correctly: str(NaN) produces the truthy string
        'nan', so we must check with pd.isna() before converting to str.
        """
        cgt_val = row.get("runtime_bytecode_hex_normalized_cgt")
        if cgt_val is not None and not pd.isna(cgt_val):
            cgt_str = str(cgt_val).strip()
            if cgt_str:
                return cgt_str
        dappscan_val = row.get("runtime_bytecode_hex_normalized_dappscan")
        if dappscan_val is not None and not pd.isna(dappscan_val):
            dappscan_str = str(dappscan_val).strip()
            if dappscan_str:
                return dappscan_str
        return ""

    merged["runtime_bytecode_hex_normalized"] = merged.apply(_pick_bytecode, axis=1)
    merged["runtime_size_bytes"] = (
        pd.concat(
            [
                pd.to_numeric(merged.get("runtime_size_bytes_cgt", 0), errors="coerce"),
                pd.to_numeric(merged.get("runtime_size_bytes_dappscan", 0), errors="coerce"),
            ],
            axis=1,
        )
        .fillna(0)
        .max(axis=1)
        .astype(int)
    )
    merged["delegatecall_count"] = (
        pd.concat(
            [
                pd.to_numeric(merged.get("delegatecall_count_cgt", 0), errors="coerce"),
                pd.to_numeric(merged.get("delegatecall_count_dappscan", 0), errors="coerce"),
            ],
            axis=1,
        )
        .fillna(0)
        .max(axis=1)
        .astype(int)
    )
    merged["opcode_count"] = (
        pd.concat(
            [
                pd.to_numeric(merged.get("opcode_count_cgt", 0), errors="coerce"),
                pd.to_numeric(merged.get("opcode_count_dappscan", 0), errors="coerce"),
            ],
            axis=1,
        )
        .fillna(0)
        .max(axis=1)
        .astype(int)
    )
    merged["is_proxy_like"] = (
        merged["is_proxy_like_cgt"].fillna(False).astype(bool)
        | merged["is_proxy_like_dappscan"].fillna(False).astype(bool)
    )
    merged["is_stub_like"] = (
        merged["is_stub_like_cgt"].fillna(False).astype(bool)
        | merged["is_stub_like_dappscan"].fillna(False).astype(bool)
    )

    unified_contracts = merged[
        [
            "fp_runtime_unified",
            "sources",
            "source_count",
            "has_cgt",
            "has_dappscan",
            "cgt_row_count",
            "dappscan_row_count",
            "cgt_fp_runtime_ids",
            "dappscan_contract_ids",
            "dappscan_contract_keys",
            "dappscan_dapps",
            "runtime_bytecode_hex_normalized",
            "runtime_size_bytes",
            "is_proxy_like",
            "is_stub_like",
            "delegatecall_count",
            "opcode_count",
        ]
    ].sort_values("fp_runtime_unified")
    unified_contracts = unified_contracts.reset_index(drop=True)

    _write_parquet(unified_contracts, unified_contracts_out)

    cgt_fps = set(_clean_text_series(cgt_df["fp_runtime_final"]).replace("", pd.NA).dropna().tolist())
    dappscan_fps = set(
        _clean_text_series(dappscan_df["fp_runtime_final"]).replace("", pd.NA).dropna().tolist()
    )
    overlap = cgt_fps & dappscan_fps

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "cgt_contracts_fp": _rel(cgt_contracts_fp),
            "dappscan_contracts_fp": _rel(dappscan_contracts_fp),
        },
        "counts": {
            "cgt_rows_in": int(len(cgt_df)),
            "dappscan_rows_in": int(len(dappscan_df)),
            "cgt_unique_fp_runtime_unified": int(len(cgt_fps)),
            "dappscan_unique_fp_runtime_unified": int(len(dappscan_fps)),
            "shared_fp_runtime_unified": int(len(overlap)),
            "union_unique_fp_runtime_unified": int(len(cgt_fps | dappscan_fps)),
            "cgt_only_fp_runtime_unified": int(len(cgt_fps - dappscan_fps)),
            "dappscan_only_fp_runtime_unified": int(len(dappscan_fps - cgt_fps)),
            "cgt_duplicates_collapsed": int(len(cgt_df) - len(cgt_fps)),
            "dappscan_duplicates_collapsed": int(len(dappscan_df) - len(dappscan_fps)),
            "unified_contract_rows_out": int(len(unified_contracts)),
        },
        "source_row_distribution": {
            "cgt_only_rows": int((unified_contracts["has_cgt"] & ~unified_contracts["has_dappscan"]).sum()),
            "dappscan_only_rows": int(
                (~unified_contracts["has_cgt"] & unified_contracts["has_dappscan"]).sum()
            ),
            "both_sources_rows": int(
                (unified_contracts["has_cgt"] & unified_contracts["has_dappscan"]).sum()
            ),
        },
        "outputs": {
            "unified_contracts_parquet": _rel(unified_contracts_out),
            "dedup_report_json": _rel(report_out),
            "dedup_report_json_copy": _rel(report_out_copy) if report_out_copy else None,
        },
    }

    _write_json(report, report_out)
    if report_out_copy and report_out_copy.resolve() != report_out.resolve():
        _write_json(report, report_out_copy)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge and deduplicate CGT/DAppSCAN contracts.")
    parser.add_argument("--cgt-contracts-fp", type=Path, default=DEFAULT_CGT_CONTRACTS_FP)
    parser.add_argument("--dappscan-contracts-fp", type=Path, default=DEFAULT_DAPPSCAN_CONTRACTS_FP)
    parser.add_argument("--unified-contracts-out", type=Path, default=DEFAULT_UNIFIED_CONTRACTS_OUT)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--report-out-copy", type=Path, default=DEFAULT_REPORT_OUT_COPY)
    args = parser.parse_args()

    report = run_merge_dedup(
        cgt_contracts_fp=args.cgt_contracts_fp,
        dappscan_contracts_fp=args.dappscan_contracts_fp,
        unified_contracts_out=args.unified_contracts_out,
        report_out=args.report_out,
        report_out_copy=args.report_out_copy,
    )
    print(f"Unified contracts: {report['counts']['unified_contract_rows_out']}")
    print(f"Shared fingerprints: {report['counts']['shared_fp_runtime_unified']}")
    print(f"Contracts parquet: {args.unified_contracts_out}")
    print(f"Dedup report: {args.report_out}")


if __name__ == "__main__":
    main()
