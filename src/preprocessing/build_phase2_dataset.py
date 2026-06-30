import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs/phase2.yaml"
DEFAULT_MAIN_BENCHMARK_SWCS = [101, 103, 104, 107, 113, 114, 115, 120, 128, 132, 135]
REQUIRED_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Phase2Config:
    main_swc_ids: List[int]
    unified_contracts_path: Path
    unified_labels_path: Path
    final_swc_recommendation_csv_path: Path
    phase1_audit_summary_md_path: Path
    primary_splits_dir: Path
    contracts_out_path: Path
    labels_out_path: Path
    split_root: Path
    dataset_card_md_path: Path
    summary_json_path: Path
    run_manifest_json_path: Path


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _write_json(payload: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def _write_text(text: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")


def _resolve_path(path_value: str) -> Path:
    raw = Path(path_value)
    if raw.is_absolute():
        return raw
    return (PROJECT_ROOT / raw).resolve()


def _clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _require_key(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing required key `{key}` in `{context}`.")
    return mapping[key]


def _load_config(config_path: Path) -> Phase2Config:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp)
    if not isinstance(raw, dict):
        raise ValueError("Phase 2 config must be a mapping.")

    inputs = _require_key(raw, "inputs", "config")
    outputs = _require_key(raw, "outputs", "config")
    if not isinstance(inputs, dict) or not isinstance(outputs, dict):
        raise ValueError("`inputs` and `outputs` must both be mappings.")

    benchmark = raw.get("main_benchmark", {})
    if benchmark is None:
        benchmark = {}
    if not isinstance(benchmark, dict):
        raise ValueError("`main_benchmark` must be a mapping when provided.")
    configured_swcs = benchmark.get("swc_ids", DEFAULT_MAIN_BENCHMARK_SWCS)
    if not isinstance(configured_swcs, list) or not configured_swcs:
        raise ValueError("`main_benchmark.swc_ids` must be a non-empty list of integers.")

    swc_ids: List[int] = []
    seen = set()
    for value in configured_swcs:
        swc_id = int(value)
        if swc_id not in seen:
            swc_ids.append(swc_id)
            seen.add(swc_id)

    return Phase2Config(
        main_swc_ids=swc_ids,
        unified_contracts_path=_resolve_path(str(_require_key(inputs, "unified_contracts", "inputs"))),
        unified_labels_path=_resolve_path(str(_require_key(inputs, "unified_labels", "inputs"))),
        final_swc_recommendation_csv_path=_resolve_path(
            str(_require_key(inputs, "final_swc_recommendation_csv", "inputs"))
        ),
        phase1_audit_summary_md_path=_resolve_path(str(_require_key(inputs, "phase1_audit_summary_md", "inputs"))),
        primary_splits_dir=_resolve_path(str(_require_key(inputs, "primary_splits_dir", "inputs"))),
        contracts_out_path=_resolve_path(str(_require_key(outputs, "contracts_parquet", "outputs"))),
        labels_out_path=_resolve_path(str(_require_key(outputs, "labels_parquet", "outputs"))),
        split_root=_resolve_path(str(_require_key(outputs, "split_root", "outputs"))),
        dataset_card_md_path=_resolve_path(str(_require_key(outputs, "dataset_card_md", "outputs"))),
        summary_json_path=_resolve_path(str(_require_key(outputs, "summary_json", "outputs"))),
        run_manifest_json_path=_resolve_path(str(_require_key(outputs, "run_manifest_json", "outputs"))),
    )


def _validate_inputs(config: Phase2Config) -> None:
    required_paths = [
        config.unified_contracts_path,
        config.unified_labels_path,
        config.final_swc_recommendation_csv_path,
        config.phase1_audit_summary_md_path,
        config.primary_splits_dir,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required input artifact(s): {missing}")

    for split in REQUIRED_SPLITS:
        ids_path = config.primary_splits_dir / f"{split}_ids.txt"
        if not ids_path.exists():
            raise FileNotFoundError(f"Primary split IDs file not found: {ids_path}")


def _load_split_ids(primary_splits_dir: Path) -> Dict[str, List[str]]:
    split_ids: Dict[str, List[str]] = {}
    assigned: Dict[str, str] = {}
    for split in REQUIRED_SPLITS:
        ids_path = primary_splits_dir / f"{split}_ids.txt"
        ids = [line.strip() for line in ids_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        duplicated_ids = pd.Series(ids).duplicated()
        if bool(duplicated_ids.any()):
            dup_values = sorted(set(pd.Series(ids)[duplicated_ids].tolist()))
            raise ValueError(f"Duplicate contract IDs found in {ids_path}: {dup_values[:10]}")

        for contract_id in ids:
            if contract_id in assigned:
                raise ValueError(
                    f"Primary split assignment overlap: `{contract_id}` is in `{assigned[contract_id]}` and `{split}`."
                )
            assigned[contract_id] = split
        split_ids[split] = ids
    return split_ids


def _validate_main_swcs(configured_swcs: Sequence[int], recommendation_csv_path: Path) -> List[int]:
    recommendation = pd.read_csv(recommendation_csv_path)
    required_columns = {"swc_id", "recommended_action"}
    missing = required_columns - set(recommendation.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in `{recommendation_csv_path}`: {sorted(missing)}"
        )
    recommendation["swc_id"] = pd.to_numeric(recommendation["swc_id"], errors="coerce").astype("Int64")
    recommendation = recommendation[recommendation["swc_id"].notna()].copy()
    recommendation["swc_id"] = recommendation["swc_id"].astype(int)

    recommended_main = sorted(
        recommendation.loc[
            recommendation["recommended_action"].astype(str).str.strip() == "keep_for_main_benchmark",
            "swc_id",
        ].tolist()
    )
    configured = sorted(set(int(v) for v in configured_swcs))
    if configured != recommended_main:
        raise ValueError(
            "Configured main benchmark SWCs do not match Phase 1 final recommendation. "
            f"configured={configured}, recommended={recommended_main}"
        )
    return configured


def _aggregate_assessed_pair_labels(labels: pd.DataFrame) -> pd.DataFrame:
    if labels.empty:
        return pd.DataFrame(
            columns=[
                "fp_runtime_unified",
                "swc_id",
                "has_positive",
                "has_negative",
                "has_conflict",
                "label",
                "label_sources",
                "label_source_count",
                "has_cgt_assessment",
                "has_dappscan_assessment",
                "has_dappscan_positive",
            ]
        )

    working = labels.copy()
    working["source"] = _clean_text(working["source"]).str.lower()
    working["is_positive"] = working["label"] == 1
    working["is_negative"] = working["label"] == 0
    working["is_cgt"] = working["source"] == "cgt"
    working["is_dappscan"] = working["source"] == "dappscan"
    working["is_dappscan_positive"] = working["is_dappscan"] & working["is_positive"]

    grouped = working.groupby(["fp_runtime_unified", "swc_id"], sort=True)
    pair = grouped.agg(
        has_positive=("is_positive", "any"),
        has_negative=("is_negative", "any"),
        has_cgt_assessment=("is_cgt", "any"),
        has_dappscan_assessment=("is_dappscan", "any"),
        has_dappscan_positive=("is_dappscan_positive", "any"),
    )
    source_values = grouped["source"].agg(
        lambda s: json.dumps(sorted({value for value in s.tolist() if value}))
    )
    source_count = grouped["source"].agg(lambda s: len({value for value in s.tolist() if value}))

    pair = pair.join(source_values.rename("label_sources")).join(source_count.rename("label_source_count")).reset_index()
    pair["has_conflict"] = pair["has_positive"] & pair["has_negative"]
    pair["label"] = pd.Series(pd.NA, index=pair.index, dtype="Int64")
    pair.loc[pair["has_positive"] & ~pair["has_negative"], "label"] = 1
    pair.loc[pair["has_negative"] & ~pair["has_positive"], "label"] = 0
    pair["label"] = pair["label"].astype("Int64")
    pair["label_source_count"] = pair["label_source_count"].astype(int)
    return pair


def _source_breakdown(contracts: pd.DataFrame) -> Dict[str, int]:
    has_cgt = contracts["has_cgt"].fillna(False).astype(bool) if "has_cgt" in contracts.columns else pd.Series(
        False, index=contracts.index
    )
    has_dappscan = (
        contracts["has_dappscan"].fillna(False).astype(bool)
        if "has_dappscan" in contracts.columns
        else pd.Series(False, index=contracts.index)
    )

    source = pd.Series("unknown", index=contracts.index)
    source.loc[has_cgt & ~has_dappscan] = "cgt_only"
    source.loc[~has_cgt & has_dappscan] = "dappscan_only"
    source.loc[has_cgt & has_dappscan] = "both"

    counts = source.value_counts()
    return {str(key): int(value) for key, value in counts.items()}


def _markdown_table(rows: List[Mapping[str, Any]], columns: Sequence[Tuple[str, str]]) -> str:
    headers = [header for _, header in columns]
    keys = [key for key, _ in columns]
    table_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        table_lines.append("| " + " | ".join(str(row.get(key, "")) for key in keys) + " |")
    return "\n".join(table_lines)


def _build_dataset_card(
    config: Phase2Config,
    summary: Dict[str, Any],
    swc_rows: List[Dict[str, Any]],
    split_counts: Dict[str, int],
    source_breakdown: Dict[str, int],
    dropped_reason_counts: Dict[str, int],
) -> str:
    swc_table_rows: List[Dict[str, Any]] = []
    for row in swc_rows:
        known = int(row["assessed_contracts"])
        pos = int(row["positive"])
        neg = int(row["negative"])
        swc_table_rows.append(
            {
                "swc_id": row["swc_id"],
                "assessed_contracts": known,
                "positive": pos,
                "negative": neg,
                "positive_rate": round((pos / known), 4) if known else 0.0,
            }
        )

    split_table_rows = [{"split": split, "contracts": split_counts.get(split, 0)} for split in REQUIRED_SPLITS]
    source_table_rows = [{"source_group": key, "contracts": value} for key, value in sorted(source_breakdown.items())]
    drop_rows = [{"reason": key, "count": value} for key, value in sorted(dropped_reason_counts.items())]

    audit_heading = config.phase1_audit_summary_md_path.read_text(encoding="utf-8").splitlines()[0].strip()

    return (
        "# Main Benchmark Dataset Card (Phase 2 Task 1)\n\n"
        f"Generated at (UTC): `{summary['generated_at_utc']}`\n\n"
        "## Scope\n\n"
        "This package freezes the **main benchmark** from Phase 1 curated artifacts only, preserving "
        "existing assessed labels and primary split assignments.\n\n"
        f"- Main benchmark SWCs: `{summary['main_benchmark_swcs']}`\n"
        f"- Phase 1 audit reference: `{_rel(config.phase1_audit_summary_md_path)}` (`{audit_heading}`)\n"
        "- DAppSCAN policy: **positive-only augmentation** (no DAppSCAN negatives are created or inferred).\n\n"
        "## Contracts per split\n\n"
        + _markdown_table(split_table_rows, [("split", "split"), ("contracts", "contracts")])
        + "\n\n## SWC class distribution (assessed labels)\n\n"
        + _markdown_table(
            swc_table_rows,
            [
                ("swc_id", "swc_id"),
                ("assessed_contracts", "assessed_contracts"),
                ("positive", "positive"),
                ("negative", "negative"),
                ("positive_rate", "positive_rate"),
            ],
        )
        + "\n\n## Source breakdown\n\n"
        + _markdown_table(source_table_rows, [("source_group", "source_group"), ("contracts", "contracts")])
        + "\n\n## Dropped contracts\n\n"
        + _markdown_table(drop_rows, [("reason", "reason"), ("count", "count")])
        + "\n\n## Output artifacts\n\n"
        f"- `{_rel(config.contracts_out_path)}`\n"
        f"- `{_rel(config.labels_out_path)}`\n"
        f"- `{_rel(config.split_root / 'train.parquet')}`\n"
        f"- `{_rel(config.split_root / 'val.parquet')}`\n"
        f"- `{_rel(config.split_root / 'test.parquet')}`\n"
        f"- `{_rel(config.summary_json_path)}`\n"
        f"- `{_rel(config.run_manifest_json_path)}`\n"
    )


def _register_drops(dropped: Dict[str, str], contract_ids: Iterable[str], reason: str) -> None:
    for contract_id in contract_ids:
        if contract_id not in dropped:
            dropped[contract_id] = reason


def build_phase2_dataset(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    run_started = datetime.now(timezone.utc)
    config = _load_config(config_path)
    _validate_inputs(config)
    main_swcs = _validate_main_swcs(config.main_swc_ids, config.final_swc_recommendation_csv_path)
    split_ids = _load_split_ids(config.primary_splits_dir)

    split_lookup: Dict[str, str] = {}
    for split, ids in split_ids.items():
        for contract_id in ids:
            split_lookup[contract_id] = split
    primary_ids = set(split_lookup.keys())

    contracts = pd.read_parquet(config.unified_contracts_path).copy()
    labels = pd.read_parquet(config.unified_labels_path).copy()

    required_contract_cols = {"fp_runtime_unified"}
    required_label_cols = {"fp_runtime_unified", "swc_id", "label", "source"}
    missing_contract_cols = sorted(required_contract_cols - set(contracts.columns))
    missing_label_cols = sorted(required_label_cols - set(labels.columns))
    if missing_contract_cols:
        raise ValueError(f"Unified contracts missing required columns: {missing_contract_cols}")
    if missing_label_cols:
        raise ValueError(f"Unified labels missing required columns: {missing_label_cols}")

    contracts["fp_runtime_unified"] = _clean_text(contracts["fp_runtime_unified"])
    contracts = contracts[contracts["fp_runtime_unified"] != ""].drop_duplicates(
        subset=["fp_runtime_unified"], keep="first"
    )

    labels["fp_runtime_unified"] = _clean_text(labels["fp_runtime_unified"])
    labels["source"] = _clean_text(labels["source"]).str.lower()
    labels["swc_id"] = pd.to_numeric(labels["swc_id"], errors="coerce").astype("Int64")
    labels["label"] = pd.to_numeric(labels["label"], errors="coerce").astype("Int64")
    labels = labels[(labels["fp_runtime_unified"] != "") & labels["swc_id"].notna()].copy()
    labels["swc_id"] = labels["swc_id"].astype(int)

    dappscan_negative_rows = labels[(labels["source"] == "dappscan") & (labels["label"] == 0)]
    if not dappscan_negative_rows.empty:
        raise ValueError(
            "Detected DAppSCAN negative labels in unified_labels; POS_ONLY policy must be preserved."
        )

    labels_main = labels[labels["swc_id"].isin(main_swcs)].copy()
    labels_main_known = labels_main[labels_main["label"].notna()].copy()
    pair_labels = _aggregate_assessed_pair_labels(labels_main_known)
    assessed_pairs = pair_labels[pair_labels["label"].notna()].copy()
    known_main_contract_ids = set(assessed_pairs["fp_runtime_unified"].tolist())

    dropped_contracts: Dict[str, str] = {}
    _register_drops(
        dropped_contracts,
        sorted(primary_ids - known_main_contract_ids),
        reason="no_assessed_main_swc_label",
    )
    _register_drops(
        dropped_contracts,
        sorted(known_main_contract_ids - primary_ids),
        reason="missing_primary_split_assignment",
    )

    candidate_ids = sorted(primary_ids & known_main_contract_ids)
    available_contract_ids = set(contracts["fp_runtime_unified"].tolist())
    missing_metadata_ids = sorted(set(candidate_ids) - available_contract_ids)
    _register_drops(dropped_contracts, missing_metadata_ids, reason="missing_contract_metadata")
    benchmark_contract_ids = sorted(set(candidate_ids) - set(missing_metadata_ids))

    if not benchmark_contract_ids:
        raise ValueError("No contracts left for main benchmark after filtering.")

    contracts_out = contracts[contracts["fp_runtime_unified"].isin(benchmark_contract_ids)].copy()
    contracts_out["split"] = contracts_out["fp_runtime_unified"].map(split_lookup)
    if bool(contracts_out["split"].isna().any()):
        raise ValueError("Found benchmark contracts without a primary split assignment.")
    split_order = pd.CategoricalDtype(categories=list(REQUIRED_SPLITS), ordered=True)
    contracts_out["split"] = contracts_out["split"].astype(split_order)
    contracts_out = contracts_out.sort_values(["split", "fp_runtime_unified"]).reset_index(drop=True)
    contracts_out["split"] = contracts_out["split"].astype(str)

    pair_index = pd.MultiIndex.from_product(
        [contracts_out["fp_runtime_unified"].tolist(), main_swcs],
        names=["fp_runtime_unified", "swc_id"],
    )
    labels_out = pair_index.to_frame(index=False)
    labels_out = labels_out.merge(
        pair_labels[
            [
                "fp_runtime_unified",
                "swc_id",
                "label",
                "has_conflict",
                "label_sources",
                "label_source_count",
                "has_cgt_assessment",
                "has_dappscan_assessment",
                "has_dappscan_positive",
            ]
        ],
        on=["fp_runtime_unified", "swc_id"],
        how="left",
    )
    labels_out["label"] = labels_out["label"].astype("Int64")
    labels_out["has_conflict"] = labels_out["has_conflict"].fillna(False).astype(bool)
    labels_out["label_sources"] = labels_out["label_sources"].fillna("[]")
    labels_out["label_source_count"] = labels_out["label_source_count"].fillna(0).astype(int)
    labels_out["has_cgt_assessment"] = labels_out["has_cgt_assessment"].fillna(False).astype(bool)
    labels_out["has_dappscan_assessment"] = labels_out["has_dappscan_assessment"].fillna(False).astype(bool)
    labels_out["has_dappscan_positive"] = labels_out["has_dappscan_positive"].fillna(False).astype(bool)
    labels_out["split"] = labels_out["fp_runtime_unified"].map(split_lookup)
    labels_out["is_assessed"] = labels_out["label"].notna()
    labels_out = labels_out.sort_values(["split", "fp_runtime_unified", "swc_id"]).reset_index(drop=True)

    label_matrix = labels_out.pivot(index="fp_runtime_unified", columns="swc_id", values="label").reindex(
        index=contracts_out["fp_runtime_unified"].tolist(),
        columns=main_swcs,
    )
    label_matrix.columns = [f"swc_{swc_id}" for swc_id in main_swcs]
    split_dataset = contracts_out.set_index("fp_runtime_unified").join(label_matrix).reset_index()

    config.contracts_out_path.parent.mkdir(parents=True, exist_ok=True)
    config.labels_out_path.parent.mkdir(parents=True, exist_ok=True)
    config.split_root.mkdir(parents=True, exist_ok=True)
    contracts_out.to_parquet(config.contracts_out_path, index=False)
    labels_out.to_parquet(config.labels_out_path, index=False)

    split_counts: Dict[str, int] = {}
    for split in REQUIRED_SPLITS:
        split_frame = split_dataset[split_dataset["split"] == split].copy()
        split_counts[split] = int(len(split_frame))
        split_frame.to_parquet(config.split_root / f"{split}.parquet", index=False)

    swc_rows: List[Dict[str, Any]] = []
    for swc_id in main_swcs:
        swc_labels = labels_out.loc[labels_out["swc_id"] == swc_id, "label"]
        swc_rows.append(
            {
                "swc_id": int(swc_id),
                "assessed_contracts": int(swc_labels.notna().sum()),
                "positive": int((swc_labels == 1).sum()),
                "negative": int((swc_labels == 0).sum()),
            }
        )

    source_breakdown = _source_breakdown(contracts_out)
    dropped_reason_counts = pd.Series(list(dropped_contracts.values())).value_counts().to_dict()
    dropped_reason_counts = {str(key): int(value) for key, value in dropped_reason_counts.items()}
    dropped_examples: Dict[str, List[str]] = {}
    for reason in sorted(dropped_reason_counts):
        dropped_examples[reason] = sorted(
            [contract_id for contract_id, why in dropped_contracts.items() if why == reason]
        )[:20]

    summary: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "phase2_task1_main_benchmark_freeze",
        "main_benchmark_swcs": main_swcs,
        "inputs": {
            "unified_contracts": _rel(config.unified_contracts_path),
            "unified_labels": _rel(config.unified_labels_path),
            "final_swc_recommendation_csv": _rel(config.final_swc_recommendation_csv_path),
            "phase1_audit_summary_md": _rel(config.phase1_audit_summary_md_path),
            "primary_splits_dir": _rel(config.primary_splits_dir),
        },
        "outputs": {
            "contracts_parquet": _rel(config.contracts_out_path),
            "labels_parquet": _rel(config.labels_out_path),
            "split_train": _rel(config.split_root / "train.parquet"),
            "split_val": _rel(config.split_root / "val.parquet"),
            "split_test": _rel(config.split_root / "test.parquet"),
            "dataset_card_md": _rel(config.dataset_card_md_path),
            "summary_json": _rel(config.summary_json_path),
            "run_manifest_json": _rel(config.run_manifest_json_path),
        },
        "contracts_total": int(len(contracts_out)),
        "labels_total_rows": int(len(labels_out)),
        "split_counts": split_counts,
        "swc_class_distribution": swc_rows,
        "source_breakdown": source_breakdown,
        "dropped_contracts": {
            "total": int(len(dropped_contracts)),
            "by_reason": dropped_reason_counts,
            "examples": dropped_examples,
        },
        "pair_label_conflicts": int(pair_labels["has_conflict"].sum()) if not pair_labels.empty else 0,
    }
    _write_json(summary, config.summary_json_path)

    run_manifest = {
        "run_started_utc": run_started.isoformat(),
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "entrypoint": "python -m src.preprocessing.build_phase2_dataset --config configs/phase2.yaml",
        "config_path": _rel(config_path.resolve()),
        "inputs": summary["inputs"],
        "outputs": summary["outputs"],
        "contracts_total": summary["contracts_total"],
        "split_counts": summary["split_counts"],
        "dropped_contracts_total": summary["dropped_contracts"]["total"],
    }
    _write_json(run_manifest, config.run_manifest_json_path)

    dataset_card = _build_dataset_card(
        config=config,
        summary=summary,
        swc_rows=swc_rows,
        split_counts=split_counts,
        source_breakdown=source_breakdown,
        dropped_reason_counts=dropped_reason_counts,
    )
    _write_text(dataset_card, config.dataset_card_md_path)
    return summary


def _print_required_summary(summary: Dict[str, Any]) -> None:
    print(f"total contracts in main benchmark: {summary['contracts_total']}")
    print("per-split counts:")
    for split in REQUIRED_SPLITS:
        print(f"- {split}: {summary['split_counts'].get(split, 0)}")

    print("per-SWC class distribution:")
    for row in summary["swc_class_distribution"]:
        print(
            f"- SWC-{row['swc_id']}: assessed={row['assessed_contracts']}, "
            f"positive={row['positive']}, negative={row['negative']}"
        )

    dropped = summary["dropped_contracts"]
    if int(dropped["total"]) == 0:
        print("any contracts dropped and why: none")
        return

    print("any contracts dropped and why:")
    for reason, count in dropped["by_reason"].items():
        examples = dropped["examples"].get(reason, [])
        print(f"- {reason}: {count} (examples: {examples[:5]})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze Phase 2 main benchmark dataset from Phase 1 artifacts.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to configs/phase2.yaml.")
    args = parser.parse_args()
    summary = build_phase2_dataset(args.config.resolve())
    _print_required_summary(summary)


if __name__ == "__main__":
    main()

