#!/usr/bin/env python3
"""Integrity checks for the committed CGT-DAppSCAN benchmark data.

Run from the repository root:

    python scripts/verify_data.py

The script needs only pandas and pyarrow. If torch is installed it also
spot-loads a few graph tensors. Every check prints PASS or FAIL with the
observed numbers; the exit code is 0 only if all checks pass.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
GRAPH_DIR = REPO / "data" / "curated" / "graphs" / "phase2_task5_20260309T161122Z"
SPLIT_DIR = REPO / "data" / "splits" / "main_benchmark"
FEATURE_DIR = REPO / "data" / "features" / "main_benchmark"
SUMMARY = REPO / "benchmark" / "clean_default_summary.json"
IDENTIFIERS = REPO / "benchmark" / "contract_identifiers.csv"

HEX64 = re.compile(r"^[0-9a-f]{64}$")

_failures: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: {detail}")
    if not ok:
        _failures.append(name)


def main() -> int:
    summary = json.loads(SUMMARY.read_text())
    n_expected = summary["contracts"]
    swc_expected = [f"swc_{i}" for i in summary["evaluated_swc_ids"]]
    split_expected = summary["split_counts"]
    source_expected = summary["source_breakdown"]

    # 1. Graph tensor directory.
    graph_files = sorted(p.name for p in GRAPH_DIR.glob("*.pt"))
    fingerprints = {name[:-3] for name in graph_files}
    check(
        "graph count",
        len(graph_files) == n_expected and len(fingerprints) == n_expected,
        f"{len(graph_files)} .pt files, {len(fingerprints)} unique (expected {n_expected})",
    )
    bad_names = [n for n in graph_files if not HEX64.match(n[:-3])]
    check("graph filenames", not bad_names, f"{len(bad_names)} non-fingerprint names")

    # 2. Split files.
    splits = {name: pd.read_parquet(SPLIT_DIR / f"{name}.parquet") for name in ("train", "val", "test")}
    sizes = {name: len(df) for name, df in splits.items()}
    check(
        "split sizes",
        sizes == dict(split_expected),
        f"{sizes} (expected {dict(split_expected)})",
    )

    fp_sets = {name: set(df["fp_runtime_unified"]) for name, df in splits.items()}
    overlap = (
        (fp_sets["train"] & fp_sets["val"])
        | (fp_sets["train"] & fp_sets["test"])
        | (fp_sets["val"] & fp_sets["test"])
    )
    check("split disjointness", not overlap, f"{len(overlap)} fingerprints shared between splits")

    union = fp_sets["train"] | fp_sets["val"] | fp_sets["test"]
    check(
        "split coverage",
        union == fingerprints,
        f"union of splits has {len(union)} fingerprints; "
        f"{len(union - fingerprints)} without a graph, {len(fingerprints - union)} graphs unsplit",
    )

    for name, df in splits.items():
        check(
            f"{name} carries no raw bytecode",
            "runtime_bytecode_hex_normalized" not in df.columns,
            "column absent" if "runtime_bytecode_hex_normalized" not in df.columns else "column present",
        )
        label_cols = [c for c in df.columns if c.startswith("swc_")]
        values = set()
        for c in label_cols:
            values.update(df[c].dropna().unique().tolist())
        check(
            f"{name} label values",
            values <= {0, 1},
            f"observed values {sorted(values)} across {len(label_cols)} swc columns "
            "(NA marks an unassessed contract-label pair)",
        )

    # 3. Contract identifier manifest.
    with IDENTIFIERS.open() as fh:
        rows = list(csv.DictReader(fh))
    id_fps = {r["fp_runtime_unified"] for r in rows}
    check(
        "identifier manifest size",
        len(rows) == n_expected and id_fps == fingerprints,
        f"{len(rows)} rows, {len(id_fps & fingerprints)} matching graph fingerprints",
    )
    source_counts = {"cgt_only": 0, "dappscan_only": 0, "both": 0}
    for r in rows:
        src = r["source"]
        key = "both" if src == "both" else f"{src}_only"
        if key in source_counts:
            source_counts[key] += 1
    check(
        "source breakdown",
        source_counts == dict(source_expected),
        f"{source_counts} (expected {dict(source_expected)})",
    )

    # 4. Feature index and feature tables.
    fi = pd.read_parquet(FEATURE_DIR / "phase3_feature_index_10swc.parquet")
    fi_swc = [c for c in fi.columns if c.startswith("swc_") and not c.endswith("_assessed")]
    check(
        "feature index labels",
        sorted(fi_swc) == sorted(swc_expected),
        f"label columns {sorted(fi_swc)}",
    )
    check(
        "feature index rows",
        len(fi) == n_expected and set(fi["fp_runtime_unified"]) == fingerprints,
        f"{len(fi)} rows",
    )
    for table in ("tfidf_features", "pattern_features", "opcode_text_corpus"):
        df = pd.read_parquet(FEATURE_DIR / f"{table}.parquet", columns=["fp_runtime_unified"])
        check(
            f"{table} rows",
            len(df) == n_expected and set(df["fp_runtime_unified"]) == fingerprints,
            f"{len(df)} rows",
        )

    # 5. Optional tensor spot-check.
    try:
        import torch

        sample = graph_files[:: max(1, len(graph_files) // 3)][:3]
        loaded = 0
        for name in sample:
            torch.load(GRAPH_DIR / name, map_location="cpu", weights_only=False)
            loaded += 1
        check("graph tensors load", loaded == len(sample), f"{loaded}/{len(sample)} sampled tensors loaded")
    except ImportError:
        print("[SKIP] graph tensors load: torch not installed")

    if _failures:
        print(f"\n{len(_failures)} check(s) failed: {', '.join(_failures)}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
