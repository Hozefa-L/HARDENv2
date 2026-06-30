#!/usr/bin/env python3
"""Reconstruct runtime bytecode for the CGT-DAppSCAN benchmark (all 2,186).

Layer 2 of the release (see DATA_PROVENANCE.md) ships contract *identifiers*
instead of re-hosted bytecode. This script resolves the runtime bytecode for
each identifier in ``benchmark/contract_identifiers.csv`` into
``data/reconstructed/<contract_id>.rt.hex``. After reconstruction, the standard
pipeline re-derives the Layer-1 graph/feature/label artifacts.

The benchmark draws on two corpora, so there are two reconstruction sources:

  * CGT contracts ``source in {cgt, both}``
      1. Local mirror  — copy the runtime hex from
         ``<cgt-root>/runtime/<runtime_hash>.rt.hex`` (deterministic, no network).
      2. On-chain RPC  — otherwise call ``eth_getCode`` for the contract address
         (``--rpc-url``); use an archive node for historical code.

  * DAppSCAN contracts ``source in {dappscan, both}``
      Resolved from a local DAppSCAN clone (``--dappscan-root``). DAppSCAN ships
      no license, so we never re-host its files; the user supplies their own
      clone. Runtime is recovered exactly as the curation pipeline does (extract
      from ``bin`` initcode, falling back to ``bin-runtime``) and keyed by the
      canonical runtime fingerprint, so it resolves regardless of file naming.

For ``both`` contracts the CGT mirror path is preferred (it is the deterministic,
bit-for-bit path). Every resolved contract is verified by re-computing its
canonical fingerprint (SHA-256 of the metadata-stripped runtime) and checking it
against ``fp_runtime_unified`` from the manifest, so reconstruction is provably
faithful without us redistributing any upstream bytecode.

Usage:
    # deterministic CGT mirror + local DAppSCAN clone
    python scripts/reconstruct_bytecode.py \
        --manifest benchmark/contract_identifiers.csv \
        --cgt-root data/raw/cgt-main \
        --dappscan-root data/raw/dappscan \
        --out data/reconstructed

    # CGT from chain instead of the mirror (requires an archive node)
    python scripts/reconstruct_bytecode.py \
        --manifest benchmark/contract_identifiers.csv \
        --rpc-url https://your-eth-endpoint \
        --dappscan-root data/raw/dappscan \
        --out data/reconstructed
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

# Allow ``import src.curation.*`` when run as ``python scripts/...`` from root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _from_mirror(cgt_root: Path, runtime_hash: str) -> str | None:
    if not runtime_hash:
        return None
    candidate = cgt_root / "runtime" / f"{runtime_hash}.rt.hex"
    if candidate.exists():
        return candidate.read_text(encoding="utf-8").strip()
    return None


def _from_rpc(rpc_url: str, address: str, block: str, retries: int = 3) -> str | None:
    if not address:
        return None
    payload = json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": "eth_getCode",
        "params": [address, block],
    }).encode()
    req = urllib.request.Request(
        rpc_url, data=payload, headers={"Content-Type": "application/json"}
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read()).get("result")
            if result and result != "0x":
                return result[2:] if result.startswith("0x") else result
            return None
        except Exception as exc:  # network hiccup; back off and retry
            if attempt == retries - 1:
                print(f"  RPC failed for {address}: {exc}", file=sys.stderr)
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


def _canonical_fp(runtime_hex: str) -> str | None:
    """Canonical runtime fingerprint: SHA-256 of metadata-stripped runtime."""
    from src.curation.bytecode_normalize import canonicalize_runtime_hex, normalize_hex
    try:
        canonical, _ = canonicalize_runtime_hex(normalize_hex(runtime_hex), mode="strip")
    except ValueError:
        return None
    return hashlib.sha256(bytes.fromhex(canonical)).hexdigest() if canonical else None


def _build_dappscan_index(dappscan_root: Path, min_runtime_len: int) -> dict[str, str]:
    """Map canonical fp_runtime_unified -> raw runtime hex over a DAppSCAN clone.

    Mirrors the curation pipeline's runtime recovery (extract from ``bin``
    initcode, fall back to ``bin-runtime``) so fingerprints match exactly.
    """
    from src.curation.evm_runtime_extract import extract_runtime_from_initcode
    from src.curation.fingerprint import _normalize_solc_hex

    bytecode_root = dappscan_root / "DAppSCAN-bytecode" / "bytecode"
    if not bytecode_root.is_dir():
        raise SystemExit(
            f"DAppSCAN bytecode dir {bytecode_root} not found. Clone "
            f"InPlusLab/DAppSCAN into {dappscan_root} (see DATA_PROVENANCE.md)."
        )
    index: dict[str, str] = {}
    for json_path in sorted(bytecode_root.rglob("*.json")):
        try:
            obj = json.loads(json_path.read_text(encoding="utf-8", errors="ignore"))
        except (json.JSONDecodeError, OSError):
            continue
        contracts = obj.get("contracts", {})
        if not isinstance(contracts, dict):
            continue
        for contract_obj in contracts.values():
            if not isinstance(contract_obj, dict):
                continue
            bin_hex = _normalize_solc_hex(contract_obj.get("bin", ""))
            binrt_hex = _normalize_solc_hex(contract_obj.get("bin-runtime", ""))
            runtime = ""
            if bin_hex:
                ex = extract_runtime_from_initcode(initcode_hex=bin_hex,
                                                   min_runtime_len=min_runtime_len)
                if ex.get("success"):
                    runtime = str(ex.get("runtime_hex", ""))
            if not runtime and binrt_hex:
                runtime = binrt_hex
            if not runtime:
                continue
            fp = _canonical_fp(runtime)
            if fp and fp not in index:
                index[fp] = runtime
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=Path("benchmark/contract_identifiers.csv"))
    parser.add_argument("--cgt-root", type=Path, default=Path("data/raw/cgt-main"),
                        help="Local CGT checkout for the deterministic mirror path.")
    parser.add_argument("--dappscan-root", type=Path, default=Path("data/raw/dappscan"),
                        help="Local DAppSCAN clone for DAppSCAN-sourced contracts.")
    parser.add_argument("--rpc-url", type=str, default=None,
                        help="Ethereum JSON-RPC endpoint for the CGT on-chain path.")
    parser.add_argument("--block", type=str, default="latest",
                        help="Block tag/number for eth_getCode (use an archive node).")
    parser.add_argument("--dappscan-min-runtime-len", type=int, default=1,
                        help="Min runtime length (bytes) for DAppSCAN initcode "
                             "extraction; 1 keeps tiny library contracts.")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip canonical-fingerprint verification.")
    parser.add_argument("--out", type=Path, default=Path("data/reconstructed"))
    args = parser.parse_args()

    if not args.manifest.exists():
        raise SystemExit(
            f"Manifest {args.manifest} not found. Run "
            f"scripts/build_identifier_manifest.py first."
        )

    with args.manifest.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    needs_dappscan = any(r.get("source") in {"dappscan", "both"} for r in rows)
    dappscan_index: dict[str, str] = {}
    if needs_dappscan:
        print(f"Indexing DAppSCAN clone at {args.dappscan_root} ...")
        dappscan_index = _build_dappscan_index(args.dappscan_root,
                                               args.dappscan_min_runtime_len)
        print(f"  indexed {len(dappscan_index)} unique DAppSCAN runtime fingerprints")

    args.out.mkdir(parents=True, exist_ok=True)
    n_mirror = n_rpc = n_dappscan = n_miss = 0
    n_verified = n_mismatch = 0

    for row in rows:
        cid = row["contract_id"]
        source = row.get("source", "cgt")
        fp_expected = (row.get("fp_runtime_unified") or "").strip()
        target = args.out / f"{cid.replace(':', '_').replace('/', '_')}.rt.hex"
        if target.exists():
            continue

        code = None
        origin = ""
        if source in {"cgt", "both"}:
            code = _from_mirror(args.cgt_root, row.get("runtime_hash", ""))
            if code is not None:
                origin = "mirror"
            elif args.rpc_url:
                code = _from_rpc(args.rpc_url, row.get("address", ""), args.block)
                if code is not None:
                    origin = "rpc"
        if code is None and source in {"dappscan", "both"} and fp_expected:
            code = dappscan_index.get(fp_expected)
            if code is not None:
                origin = "dappscan"

        if code is None:
            n_miss += 1
            continue

        if not args.no_verify and fp_expected:
            if _canonical_fp(code) == fp_expected:
                n_verified += 1
            else:
                n_mismatch += 1
                print(f"  fingerprint mismatch for {cid} (source={source})",
                      file=sys.stderr)

        if origin == "mirror":
            n_mirror += 1
        elif origin == "rpc":
            n_rpc += 1
        elif origin == "dappscan":
            n_dappscan += 1
        target.write_text(code, encoding="utf-8")

    total = n_mirror + n_rpc + n_dappscan
    print(f"Reconstructed {total}/{len(rows)} contracts "
          f"(cgt_mirror={n_mirror}, cgt_rpc={n_rpc}, dappscan={n_dappscan}, "
          f"unresolved={n_miss}) into {args.out}")
    if not args.no_verify:
        print(f"Fingerprint verification: {n_verified} verified, "
              f"{n_mismatch} mismatched.")
    if n_miss:
        print("Unresolved CGT contracts need a CGT mirror or --rpc-url (archive "
              "node); unresolved DAppSCAN contracts need a local --dappscan-root "
              "clone. See DATA_PROVENANCE.md.")


if __name__ == "__main__":
    main()
