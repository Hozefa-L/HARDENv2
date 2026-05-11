import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .bytecode_flags import compute_bytecode_flags
from .bytecode_normalize import canonicalize_runtime_hex, normalize_hex
from .evm_runtime_extract import extract_runtime_from_initcode

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CGT_CONTRACTS_IN = PROJECT_ROOT / "data/intermediate/cgt_contracts.parquet"
DEFAULT_DAPPSCAN_CONTRACTS_IN = PROJECT_ROOT / "data/intermediate/dappscan_contracts.parquet"
DEFAULT_CGT_RUNTIME_DIR = PROJECT_ROOT / "data/raw/cgt-main/runtime"
DEFAULT_DAPPSCAN_ROOT = PROJECT_ROOT / "data/raw/dappscan"
DEFAULT_CGT_OUT = PROJECT_ROOT / "data/intermediate/cgt_contracts_fp.parquet"
DEFAULT_DAPPSCAN_OUT = PROJECT_ROOT / "data/intermediate/dappscan_contracts_fp.parquet"
DEFAULT_SPEC_OUT = PROJECT_ROOT / "reports/phase1/fingerprint_spec.md"
DEFAULT_REPORT_OUT = PROJECT_ROOT / "reports/phase1/fingerprint_report.json"

HASH_ALGORITHM = "sha256"


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


def _sha256_hex(hex_string: str) -> str:
    return hashlib.sha256(bytes.fromhex(hex_string)).hexdigest()


def _extract_contract_name(contract_key: str, fallback: str) -> str:
    key = str(contract_key).strip()
    if ":" in key:
        name = key.rsplit(":", 1)[-1].strip()
        if name:
            return name
    return fallback


def _length_stats(values: List[int]) -> Dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
        }
    arr = np.array(values, dtype=np.int64)
    return {
        "count": int(len(arr)),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(np.round(arr.mean(), 4)),
        "median": float(np.round(float(np.median(arr)), 4)),
        "p95": float(np.round(float(np.percentile(arr, 95)), 4)),
    }


def _load_cgt_contracts(cgt_contracts_in: Path, cgt_runtime_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    contracts = pd.read_parquet(cgt_contracts_in).copy()
    if "fp_runtime" not in contracts.columns:
        raise ValueError("cgt_contracts.parquet must include fp_runtime.")

    runtime_source = "existing_runtime_column"
    runtime_col = "runtime_bytecode_hex"
    if runtime_col not in contracts.columns:
        runtime_source = "runtime_artifact_files"
        runtime_hexes: List[str] = []
        missing_artifacts = 0
        for fp_runtime in contracts["fp_runtime"].astype(str):
            p1 = cgt_runtime_dir / f"{fp_runtime}.rt.hex"
            p2 = cgt_runtime_dir / fp_runtime
            runtime_path = p1 if p1.exists() else p2
            if runtime_path.exists():
                runtime_hexes.append(runtime_path.read_text(encoding="utf-8", errors="ignore").strip())
            else:
                runtime_hexes.append("")
                missing_artifacts += 1
        contracts[runtime_col] = runtime_hexes
    else:
        missing_artifacts = int((contracts[runtime_col].fillna("").astype(str).str.strip() == "").sum())

    stats = {
        "rows": int(len(contracts)),
        "runtime_source": runtime_source,
        "missing_runtime_rows": int(missing_artifacts),
    }
    return contracts, stats


def _normalize_solc_hex(raw: Any) -> str:
    """Normalize a hex string from solc combined-json output."""
    if raw is None:
        return ""
    text = str(raw).strip()
    if text.startswith("0x") or text.startswith("0X"):
        text = text[2:]
    return text.lower()


def _build_dappscan_contracts(
    dappscan_root: Path,
    min_runtime_len: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    bytecode_root = dappscan_root / "DAppSCAN-bytecode" / "bytecode"
    records: List[Dict[str, Any]] = []
    json_files_scanned = 0
    parse_errors = 0
    contract_entries_total = 0
    nonempty_initcode_entries = 0
    extraction_success = 0
    extraction_failures = 0
    binruntime_direct_success = 0
    binruntime_fallback_success = 0
    both_empty_skipped = 0

    for json_path in sorted(bytecode_root.rglob("*.json")):
        json_files_scanned += 1
        try:
            rel = json_path.relative_to(bytecode_root)
            dapp = rel.parts[0] if rel.parts else json_path.parent.name
        except Exception:
            dapp = json_path.parent.name
        try:
            obj = json.loads(json_path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            parse_errors += 1
            continue
        contracts = obj.get("contracts", {})
        if not isinstance(contracts, dict):
            continue

        for contract_key, contract_obj in contracts.items():
            contract_entries_total += 1
            if not isinstance(contract_obj, dict):
                continue

            initcode_hex = _normalize_solc_hex(contract_obj.get("bin", ""))
            bin_runtime_hex = _normalize_solc_hex(contract_obj.get("bin-runtime", ""))

            if not initcode_hex and not bin_runtime_hex:
                both_empty_skipped += 1
                continue

            # Try extracting runtime from initcode first (preserves original logic)
            runtime_hex = ""
            runtime_source = ""
            runtime_offset = 0
            runtime_length = 0

            if initcode_hex:
                nonempty_initcode_entries += 1
                extraction = extract_runtime_from_initcode(
                    initcode_hex=initcode_hex,
                    min_runtime_len=min_runtime_len,
                )
                if extraction.get("success"):
                    extraction_success += 1
                    runtime_hex = str(extraction.get("runtime_hex", ""))
                    runtime_source = "bin_extracted"
                    runtime_offset = int(extraction.get("runtime_offset", 0))
                    runtime_length = int(extraction.get("runtime_length", 0))
                else:
                    extraction_failures += 1

            # Fallback (or primary when bin is empty): use bin-runtime directly
            if not runtime_hex and bin_runtime_hex:
                if len(bin_runtime_hex) >= min_runtime_len * 2:
                    runtime_hex = bin_runtime_hex
                    runtime_length = len(bin_runtime_hex) // 2
                    runtime_offset = 0
                    if initcode_hex:
                        runtime_source = "bin_runtime_fallback"
                        binruntime_fallback_success += 1
                    else:
                        runtime_source = "bin_runtime_direct"
                        binruntime_direct_success += 1

            if not runtime_hex:
                continue

            contract_name = _extract_contract_name(str(contract_key), fallback=json_path.stem)
            records.append(
                {
                    "dapp": dapp,
                    "contract_key": str(contract_key),
                    "contract_name": contract_name,
                    "source_json_path": _rel(json_path),
                    "runtime_offset": runtime_offset,
                    "runtime_length": runtime_length,
                    "runtime_bytecode_hex": runtime_hex,
                    "runtime_source": runtime_source,
                }
            )

    contracts_df = pd.DataFrame(records)
    if not contracts_df.empty:
        contracts_df = contracts_df.drop_duplicates(subset=["dapp", "contract_key"]).reset_index(drop=True)

    stats = {
        "json_files_scanned": int(json_files_scanned),
        "parse_errors": int(parse_errors),
        "contract_entries_total": int(contract_entries_total),
        "nonempty_initcode_entries": int(nonempty_initcode_entries),
        "extraction_success": int(extraction_success),
        "extraction_failures": int(extraction_failures),
        "binruntime_direct_success": int(binruntime_direct_success),
        "binruntime_fallback_success": int(binruntime_fallback_success),
        "both_empty_skipped": int(both_empty_skipped),
        "rows_output": int(len(contracts_df)),
    }
    return contracts_df, stats


def _load_dappscan_contracts(
    dappscan_contracts_in: Path,
    dappscan_root: Path,
    min_runtime_len: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if dappscan_contracts_in.exists():
        contracts = pd.read_parquet(dappscan_contracts_in).copy()
        if "runtime_bytecode_hex" not in contracts.columns:
            raise ValueError("dappscan_contracts.parquet must include runtime_bytecode_hex.")
        stats = {"rows": int(len(contracts)), "source": "existing_parquet"}
        return contracts, stats

    contracts, build_stats = _build_dappscan_contracts(
        dappscan_root=dappscan_root,
        min_runtime_len=min_runtime_len,
    )
    if contracts.empty:
        raise ValueError(
            "Could not construct dappscan_contracts.parquet: no extracted runtime bytecode rows found."
        )
    _write_parquet(contracts, dappscan_contracts_in)
    stats = {"rows": int(len(contracts)), "source": "constructed_from_bytecode_json"}
    stats.update(build_stats)
    return contracts, stats


def _apply_fingerprints_and_flags(
    df: pd.DataFrame,
    runtime_col: str,
    metadata_mode: str,
    stub_threshold_bytes: int,
    delegatecall_proxy_threshold: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = df.copy()
    raw_values: List[str] = []
    normalized_values: List[str] = []
    fp_values: List[Optional[str]] = []
    metadata_detected_values: List[bool] = []
    metadata_removed_values: List[bool] = []
    metadata_len_values: List[int] = []
    norm_errors: List[Optional[str]] = []

    runtime_size_values: List[int] = []
    proxy_values: List[bool] = []
    eip1167_values: List[bool] = []
    stub_values: List[bool] = []
    delegatecall_ratio_values: List[float] = []
    delegatecall_count_values: List[int] = []
    opcode_count_values: List[int] = []

    raw_len_values: List[int] = []
    normalized_len_values: List[int] = []
    invalid_hex_rows = 0

    for runtime_hex in out[runtime_col]:
        raw_text = "" if runtime_hex is None else str(runtime_hex).strip()
        raw_values.append(raw_text)
        try:
            raw_normalized = normalize_hex(raw_text)
            canonical_hex, info = canonicalize_runtime_hex(raw_normalized, mode=metadata_mode)
            fp = _sha256_hex(canonical_hex) if canonical_hex else None
            flags = compute_bytecode_flags(
                canonical_hex,
                stub_threshold_bytes=stub_threshold_bytes,
                delegatecall_proxy_threshold=delegatecall_proxy_threshold,
            )
            normalized_values.append(canonical_hex)
            fp_values.append(fp)
            metadata_detected_values.append(bool(info["metadata_detected"]))
            metadata_removed_values.append(bool(info["metadata_removed"]))
            metadata_len_values.append(int(info["metadata_len_bytes"]))
            norm_errors.append(None)
            runtime_size_values.append(int(flags["runtime_size_bytes"]))
            proxy_values.append(bool(flags["is_proxy_like"]))
            eip1167_values.append(bool(flags["is_eip1167_proxy"]))
            stub_values.append(bool(flags["is_stub_like"]))
            delegatecall_ratio_values.append(float(flags["delegatecall_ratio"]))
            delegatecall_count_values.append(int(flags["delegatecall_count"]))
            opcode_count_values.append(int(flags["opcode_count"]))
            raw_len_values.append(int(info["raw_length_bytes"]))
            normalized_len_values.append(int(info["canonical_length_bytes"]))
        except ValueError as exc:
            invalid_hex_rows += 1
            normalized_values.append("")
            fp_values.append(None)
            metadata_detected_values.append(False)
            metadata_removed_values.append(False)
            metadata_len_values.append(0)
            norm_errors.append(str(exc))
            runtime_size_values.append(0)
            proxy_values.append(False)
            eip1167_values.append(False)
            stub_values.append(True)
            delegatecall_ratio_values.append(0.0)
            delegatecall_count_values.append(0)
            opcode_count_values.append(0)
            raw_len_values.append(0)
            normalized_len_values.append(0)

    out["runtime_bytecode_hex_raw"] = raw_values
    out["runtime_bytecode_hex_normalized"] = normalized_values
    out["fp_runtime_computed"] = fp_values
    out["fp_runtime_final"] = fp_values
    out["metadata_detected"] = metadata_detected_values
    out["metadata_removed"] = metadata_removed_values
    out["metadata_len_bytes"] = metadata_len_values
    out["normalization_error"] = norm_errors
    out["runtime_size_bytes"] = runtime_size_values
    out["is_proxy_like"] = proxy_values
    out["is_eip1167_proxy"] = eip1167_values
    out["is_stub_like"] = stub_values
    out["delegatecall_ratio"] = delegatecall_ratio_values
    out["delegatecall_count"] = delegatecall_count_values
    out["opcode_count"] = opcode_count_values

    valid_mask = out["fp_runtime_final"].notna()
    duplicate_fp_count = int(out.loc[valid_mask, "fp_runtime_final"].duplicated(keep=False).sum())
    fp_counts = out.loc[valid_mask, "fp_runtime_final"].value_counts()
    fp_multi_rows = int((fp_counts > 1).sum())
    fp_to_normalized = (
        out.loc[valid_mask, ["fp_runtime_final", "runtime_bytecode_hex_normalized"]]
        .drop_duplicates()
        .groupby("fp_runtime_final")["runtime_bytecode_hex_normalized"]
        .nunique()
    )
    fp_with_multiple_normalized = int((fp_to_normalized > 1).sum())

    stats = {
        "rows_total": int(len(out)),
        "rows_with_nonempty_runtime_input": int((pd.Series(raw_values) != "").sum()),
        "rows_normalized": int(valid_mask.sum()),
        "invalid_hex_rows": int(invalid_hex_rows),
        "metadata_detected_rows": int(out["metadata_detected"].sum()),
        "metadata_removed_rows": int(out["metadata_removed"].sum()),
        "metadata_removed_total_bytes": int(out["metadata_len_bytes"].sum()),
        "raw_runtime_length_stats_bytes": _length_stats(raw_len_values),
        "normalized_runtime_length_stats_bytes": _length_stats(normalized_len_values),
        "flags": {
            "proxy_like_rows": int(out["is_proxy_like"].sum()),
            "eip1167_rows": int(out["is_eip1167_proxy"].sum()),
            "stub_like_rows": int(out["is_stub_like"].sum()),
            "delegatecall_ratio_mean": float(np.round(out["delegatecall_ratio"].mean(), 6))
            if len(out)
            else 0.0,
            "delegatecall_ratio_p95": float(
                np.round(np.percentile(out["delegatecall_ratio"], 95), 6)
            )
            if len(out)
            else 0.0,
        },
        "collisions": {
            "unique_fingerprints": int(out.loc[valid_mask, "fp_runtime_final"].nunique()),
            "duplicate_fingerprint_rows": duplicate_fp_count,
            "fingerprints_with_multiple_rows": fp_multi_rows,
            "fingerprints_with_multiple_normalized_bytecodes": fp_with_multiple_normalized,
        },
    }
    return out, stats


def _write_fingerprint_spec(spec_out: Path, metadata_mode: str, hash_algorithm: str) -> None:
    spec_out.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "# Runtime Fingerprint Specification\n\n"
        "## Canonical normalization\n"
        "1. `normalize_hex()` lowercases hex, strips optional `0x`, removes whitespace, and validates even-length hex.\n"
        "2. `strip_solc_metadata()` applies a conservative Solidity CBOR metadata removal on runtime bytecode tails:\n"
        "   - Reads the trailing 2-byte CBOR-length field.\n"
        "   - Verifies in-bounds span and CBOR map-like header.\n"
        "   - Requires known Solidity metadata markers (`ipfs`, `bzzr0`, `bzzr1`, `solc`, `experimental`) in the tail blob.\n"
        f"   - Current policy: `{metadata_mode}` (preferred `strip`; `zero` supported for fixed-length normalization).\n\n"
        "Solidity appends metadata to runtime bytecode by default (CBOR with source/compiler metadata); "
        "removing it improves semantic equivalence of functionally identical contracts compiled with different metadata payloads.\n\n"
        "## Fingerprint\n"
        f"- Hash algorithm: `{hash_algorithm}`.\n"
        "- Computation: `fp_runtime_computed = hash(runtime_bytecode_hex_normalized_bytes)`.\n"
        "- `fp_runtime_final` currently equals `fp_runtime_computed`.\n\n"
        "## Graph-quality flags\n"
        "- `is_eip1167_proxy`: strict EIP-1167 runtime pattern match.\n"
        "- `is_stub_like`: runtime bytecode length `< 100` bytes.\n"
        "- `delegatecall_ratio`: `count(DELEGATECALL opcode 0xF4) / opcode_count`.\n"
        "- `is_proxy_like`: `is_eip1167_proxy` OR (`delegatecall_ratio` above configured threshold and at least one DELEGATECALL).\n"
    )
    spec_out.write_text(content, encoding="utf-8")


def run_fingerprint_pipeline(
    cgt_contracts_in: Path,
    dappscan_contracts_in: Path,
    cgt_runtime_dir: Path,
    dappscan_root: Path,
    cgt_out: Path,
    dappscan_out: Path,
    report_out: Path,
    spec_out: Path,
    metadata_mode: str = "strip",
    stub_threshold_bytes: int = 100,
    delegatecall_proxy_threshold: float = 0.02,
    dappscan_min_runtime_len: int = 1,
) -> Dict[str, Any]:
    cgt_df, cgt_input_stats = _load_cgt_contracts(cgt_contracts_in, cgt_runtime_dir)
    dappscan_df, dappscan_input_stats = _load_dappscan_contracts(
        dappscan_contracts_in=dappscan_contracts_in,
        dappscan_root=dappscan_root,
        min_runtime_len=dappscan_min_runtime_len,
    )

    cgt_fp, cgt_stats = _apply_fingerprints_and_flags(
        cgt_df,
        runtime_col="runtime_bytecode_hex",
        metadata_mode=metadata_mode,
        stub_threshold_bytes=stub_threshold_bytes,
        delegatecall_proxy_threshold=delegatecall_proxy_threshold,
    )
    dappscan_fp, dappscan_stats = _apply_fingerprints_and_flags(
        dappscan_df,
        runtime_col="runtime_bytecode_hex",
        metadata_mode=metadata_mode,
        stub_threshold_bytes=stub_threshold_bytes,
        delegatecall_proxy_threshold=delegatecall_proxy_threshold,
    )

    _write_parquet(cgt_fp, cgt_out)
    _write_parquet(dappscan_fp, dappscan_out)
    _write_fingerprint_spec(spec_out, metadata_mode=metadata_mode, hash_algorithm=HASH_ALGORITHM)

    cgt_fp_set = set(cgt_fp["fp_runtime_final"].dropna().astype(str).tolist())
    dappscan_fp_set = set(dappscan_fp["fp_runtime_final"].dropna().astype(str).tolist())
    shared_fps = cgt_fp_set & dappscan_fp_set

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hash_algorithm": HASH_ALGORITHM,
        "normalization": {
            "metadata_mode": metadata_mode,
            "stub_threshold_bytes": int(stub_threshold_bytes),
            "delegatecall_proxy_threshold": float(delegatecall_proxy_threshold),
            "dappscan_min_runtime_len_for_extraction": int(dappscan_min_runtime_len),
        },
        "inputs": {
            "cgt_contracts": _rel(cgt_contracts_in),
            "dappscan_contracts": _rel(dappscan_contracts_in),
            "cgt_runtime_dir": _rel(cgt_runtime_dir),
            "dappscan_root": _rel(dappscan_root),
            "cgt_input_stats": cgt_input_stats,
            "dappscan_input_stats": dappscan_input_stats,
        },
        "datasets": {
            "cgt": cgt_stats,
            "dappscan": dappscan_stats,
        },
        "cross_dataset_collision_sanity": {
            "cgt_unique_fingerprints": int(len(cgt_fp_set)),
            "dappscan_unique_fingerprints": int(len(dappscan_fp_set)),
            "shared_fingerprint_count": int(len(shared_fps)),
        },
        "outputs": {
            "cgt_contracts_fp_parquet": _rel(cgt_out),
            "dappscan_contracts_fp_parquet": _rel(dappscan_out),
            "fingerprint_spec_md": _rel(spec_out),
            "fingerprint_report_json": _rel(report_out),
        },
    }

    report_out.parent.mkdir(parents=True, exist_ok=True)
    with report_out.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical runtime fingerprinting and proxy/stub flags.")
    parser.add_argument("--cgt-contracts-in", type=Path, default=DEFAULT_CGT_CONTRACTS_IN)
    parser.add_argument("--dappscan-contracts-in", type=Path, default=DEFAULT_DAPPSCAN_CONTRACTS_IN)
    parser.add_argument("--cgt-runtime-dir", type=Path, default=DEFAULT_CGT_RUNTIME_DIR)
    parser.add_argument("--dappscan-root", type=Path, default=DEFAULT_DAPPSCAN_ROOT)
    parser.add_argument("--cgt-out", type=Path, default=DEFAULT_CGT_OUT)
    parser.add_argument("--dappscan-out", type=Path, default=DEFAULT_DAPPSCAN_OUT)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--spec-out", type=Path, default=DEFAULT_SPEC_OUT)
    parser.add_argument("--metadata-mode", choices=["strip", "zero"], default="strip")
    parser.add_argument("--stub-threshold-bytes", type=int, default=100)
    parser.add_argument("--delegatecall-proxy-threshold", type=float, default=0.02)
    parser.add_argument("--dappscan-min-runtime-len", type=int, default=1)
    args = parser.parse_args()

    report = run_fingerprint_pipeline(
        cgt_contracts_in=args.cgt_contracts_in,
        dappscan_contracts_in=args.dappscan_contracts_in,
        cgt_runtime_dir=args.cgt_runtime_dir,
        dappscan_root=args.dappscan_root,
        cgt_out=args.cgt_out,
        dappscan_out=args.dappscan_out,
        report_out=args.report_out,
        spec_out=args.spec_out,
        metadata_mode=args.metadata_mode,
        stub_threshold_bytes=args.stub_threshold_bytes,
        delegatecall_proxy_threshold=args.delegatecall_proxy_threshold,
        dappscan_min_runtime_len=args.dappscan_min_runtime_len,
    )
    print(f"Hash algorithm: {report['hash_algorithm']}")
    print(f"CGT rows: {report['datasets']['cgt']['rows_total']}")
    print(f"DAppSCAN rows: {report['datasets']['dappscan']['rows_total']}")
    print(f"CGT output: {args.cgt_out}")
    print(f"DAppSCAN output: {args.dappscan_out}")
    print(f"Report: {args.report_out}")
    print(f"Spec: {args.spec_out}")


if __name__ == "__main__":
    main()
