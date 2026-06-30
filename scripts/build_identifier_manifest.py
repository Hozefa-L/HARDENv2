#!/usr/bin/env python3
"""Build the CGT-DAppSCAN contract-identifier manifest (all 2,186 contracts).

Layer 2 of the release (see DATA_PROVENANCE.md) references contracts by
*identifier* instead of re-hosting upstream bytecode. This script regenerates
``benchmark/contract_identifiers.csv`` so that ``scripts/reconstruct_bytecode.py``
can resolve the runtime bytecode for **every** contract in the benchmark and the
pipeline can re-derive the Layer-1 artifacts.

The manifest is a faithful export of the *curated benchmark*: it is built from
the fixed split artifacts under ``data/splits/main_benchmark/`` (the same files
the training pipeline consumes), so the manifest contract set is identical to
the released benchmark --- all 2,186 contracts across both sources, not just the
CGT layer. For each contract it emits whichever identifiers are needed to
re-fetch its runtime bytecode:

  * CGT contracts      -> ``runtime_hash`` (resolves from the CGT mirror by
                          ``<hash>.rt.hex``) plus ``chain``/``address`` for the
                          on-chain RPC path, joined from ``consolidated.csv``.
  * DAppSCAN contracts -> ``dappscan_contract_id`` (``<dapp>/<contract>``), which
                          resolves from a local DAppSCAN clone. No DAppSCAN file
                          content is copied here --- only the identifier.
  * contracts in both  -> both sets of identifiers (CGT is preferred at
                          reconstruction time, matching the curation pipeline).

Every row also carries ``fp_runtime_unified`` --- the canonical runtime
fingerprint (SHA-256 of the metadata-stripped runtime). The reconstruction
script re-computes this from the re-fetched bytecode and verifies it matches,
so reconstruction is checkable bit-for-bit without us redistributing bytecode.

Usage:
    python scripts/build_identifier_manifest.py \
        --splits-dir data/splits/main_benchmark \
        --cgt-csv data/raw/cgt-main/consolidated.csv \
        --out benchmark/contract_identifiers.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Ten evaluated SWCs (must match configs/phase7_balanced.yaml). SWC-132 is
# carried in the split files but excluded from the evaluated benchmark.
EVALUATED_SWCS = [101, 103, 104, 107, 113, 114, 115, 120, 128, 135]


def _json_first(value: Any) -> str:
    """Return the first element of a JSON-encoded list field, or ''."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    try:
        items = json.loads(value) if isinstance(value, str) else list(value)
    except (json.JSONDecodeError, TypeError):
        return ""
    items = [str(i).strip() for i in items if str(i).strip()]
    return sorted(items)[0] if items else ""


def _load_cgt_addresses(cgt_csv: Path) -> Dict[str, Dict[str, str]]:
    """Map CGT fp_runtime -> first non-empty {chain, address, bytecode_hash}.

    consolidated.csv is ';'-delimited with one row per (contract, property).
    """
    out: Dict[str, Dict[str, str]] = {}
    if not cgt_csv.exists():
        return out
    with cgt_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter=";"):
            fp = (row.get("fp_runtime") or "").strip()
            if not fp or fp in out:
                continue
            out[fp] = {
                "chain": (row.get("chain") or "").strip(),
                "address": (row.get("addr") or "").strip(),
                "bytecode_hash": (row.get("fp_bytecode") or "").strip(),
            }
    return out


def _read_splits(splits_dir: Path) -> pd.DataFrame:
    frames = []
    for name in ("train", "val", "test"):
        fp = splits_dir / f"{name}.parquet"
        if not fp.exists():
            raise SystemExit(
                f"Split file {fp} not found. Re-derive the benchmark first "
                f"(see README 'Reproducing the benchmark') or point --splits-dir "
                f"at the curated split artifacts."
            )
        frames.append(pd.read_parquet(fp))
    return pd.concat(frames, ignore_index=True)


def _swc_fields(row: pd.Series) -> Dict[str, str]:
    assessed: List[int] = []
    positive: List[int] = []
    for swc in EVALUATED_SWCS:
        col = f"swc_{swc}"
        if col not in row:
            continue
        val = row[col]
        if pd.isna(val):
            continue
        assessed.append(swc)
        if int(val) == 1:
            positive.append(swc)
    return {
        "assessed_swcs": "|".join(str(s) for s in assessed),
        "positive_swcs": "|".join(str(s) for s in positive),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits-dir", type=Path,
                        default=Path("data/splits/main_benchmark"),
                        help="Curated split artifacts (train/val/test.parquet).")
    parser.add_argument("--cgt-csv", type=Path,
                        default=Path("data/raw/cgt-main/consolidated.csv"),
                        help="CGT consolidated.csv for chain/address/bytecode_hash.")
    parser.add_argument("--out", type=Path,
                        default=Path("benchmark/contract_identifiers.csv"))
    args = parser.parse_args()

    df = _read_splits(args.splits_dir)
    cgt_addr = _load_cgt_addresses(args.cgt_csv)

    fields = ["contract_id", "source", "chain", "address", "runtime_hash",
              "bytecode_hash", "dappscan_contract_id", "fp_runtime_unified",
              "assessed_swcs", "positive_swcs"]

    counts = defaultdict(int)
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        has_cgt = bool(r.get("has_cgt"))
        has_dappscan = bool(r.get("has_dappscan"))
        source = "both" if (has_cgt and has_dappscan) else ("cgt" if has_cgt else "dappscan")
        counts[source] += 1

        runtime_hash = _json_first(r.get("cgt_fp_runtime_ids")) if has_cgt else ""
        addr = cgt_addr.get(runtime_hash, {}) if runtime_hash else {}
        rec = {
            "contract_id": str(r["fp_runtime_unified"]),
            "source": source,
            "chain": addr.get("chain", ""),
            "address": addr.get("address", ""),
            "runtime_hash": runtime_hash,
            "bytecode_hash": addr.get("bytecode_hash", ""),
            "dappscan_contract_id": _json_first(r.get("dappscan_contract_ids")) if has_dappscan else "",
            "fp_runtime_unified": str(r["fp_runtime_unified"]),
        }
        rec.update(_swc_fields(r))
        rows.append(rec)

    rows.sort(key=lambda x: (x["source"], x["contract_id"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} contract identifiers to {args.out}")
    print(f"  source breakdown: cgt={counts['cgt']} dappscan={counts['dappscan']} "
          f"both={counts['both']}")
    if not cgt_addr:
        print("  note: --cgt-csv not found; chain/address columns left empty "
              "(CGT mirror path via runtime_hash still works).")


if __name__ == "__main__":
    main()
