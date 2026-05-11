import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import pandas as pd
import yaml

from src.features.codebert_features import CodeBertExtractionConfig, extract_codebert_features
from src.features.graph_features import build_graph_level_features
from src.features.opcode_text import build_failure_mode_index, build_opcode_text_corpus
from src.features.pattern_features import build_pattern_features
from src.features.tfidf_features import build_tfidf_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs/phase3.yaml"
DEFAULT_PHASE2_RUN_MANIFEST = PROJECT_ROOT / "reports/phase2/graph_builder_run_manifest.json"
DEFAULT_MAIN_BENCHMARK_SWCS = [101, 103, 104, 107, 113, 114, 115, 120, 128, 132, 135]
CONTRACT_ID_COLUMN = "fp_runtime_unified"
REQUIRED_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Phase3Config:
    main_swc_ids: List[int]
    phase2_run_manifest_path: Path
    contracts_path: Path
    labels_path: Path
    graph_index_path: Path
    graphs_dir: Path
    instructions_path: Path
    split_root: Path
    feature_index_out_path: Path
    opcode_text_out_path: Path
    codebert_out_path: Path
    graph_features_out_path: Path
    tfidf_out_path: Path
    pattern_out_path: Path
    report_json_path: Path
    dataset_card_md_path: Path
    run_manifest_json_path: Path
    codebert: CodeBertExtractionConfig


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _resolve_path(path_value: Any) -> Path:
    raw = Path(str(path_value))
    if raw.is_absolute():
        return raw
    return (PROJECT_ROOT / raw).resolve()


def _safe_read_mapping(path: Path, context: str) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {context}: {path}")
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) if path.suffix in {".yaml", ".yml"} else json.load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"{context} must be a mapping: {path}")
    return data


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def _write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _resolve_output_path(configured_path: Path, notes: List[Dict[str, str]]) -> Path:
    if not configured_path.exists():
        return configured_path

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = configured_path.with_name(f"{configured_path.stem}_{timestamp}{configured_path.suffix}")
    suffix = 1
    while candidate.exists():
        candidate = configured_path.with_name(f"{configured_path.stem}_{timestamp}_{suffix}{configured_path.suffix}")
        suffix += 1

    notes.append(
        {
            "configured_path": _rel(configured_path),
            "resolved_path": _rel(candidate),
            "reason": "configured_output_already_exists",
        }
    )
    return candidate


def _normalize_swc_ids(raw_values: Sequence[Any]) -> List[int]:
    if not raw_values:
        raise ValueError("`main_benchmark.swc_ids` must be a non-empty list.")
    values: List[int] = []
    seen = set()
    for raw in raw_values:
        swc = int(raw)
        if swc not in seen:
            values.append(swc)
            seen.add(swc)
    return values


def _load_config(config_path: Path) -> Phase3Config:
    raw = _safe_read_mapping(config_path, "Phase 3 config")

    benchmark = raw.get("main_benchmark", {})
    if benchmark is None:
        benchmark = {}
    if not isinstance(benchmark, dict):
        raise ValueError("`main_benchmark` must be a mapping when provided.")
    main_swc_ids = _normalize_swc_ids(benchmark.get("swc_ids", DEFAULT_MAIN_BENCHMARK_SWCS))

    inputs = raw.get("inputs", {})
    if inputs is None:
        inputs = {}
    if not isinstance(inputs, dict):
        raise ValueError("`inputs` must be a mapping when provided.")

    outputs = raw.get("outputs", {})
    if outputs is None:
        outputs = {}
    if not isinstance(outputs, dict):
        raise ValueError("`outputs` must be a mapping when provided.")

    codebert_cfg = raw.get("codebert", {})
    if codebert_cfg is None:
        codebert_cfg = {}
    if not isinstance(codebert_cfg, dict):
        raise ValueError("`codebert` must be a mapping when provided.")

    phase2_manifest_path = _resolve_path(
        inputs.get("phase2_graph_builder_run_manifest_json") or str(DEFAULT_PHASE2_RUN_MANIFEST)
    )
    phase2_manifest = _safe_read_mapping(phase2_manifest_path, "Phase 2 graph builder run manifest")
    phase2_inputs = phase2_manifest.get("inputs", {})
    phase2_outputs = phase2_manifest.get("outputs", {})
    if not isinstance(phase2_inputs, dict) or not isinstance(phase2_outputs, dict):
        raise ValueError("Phase 2 graph builder run manifest must contain mapping `inputs` and `outputs`.")

    contracts_path = _resolve_path(
        inputs.get("contracts_parquet")
        or phase2_inputs.get("contracts_parquet")
        or "data/curated/main_benchmark_contracts.parquet"
    )
    labels_path = _resolve_path(
        inputs.get("labels_parquet")
        or phase2_inputs.get("labels_parquet")
        or "data/curated/main_benchmark_labels.parquet"
    )
    graph_index_path = _resolve_path(
        inputs.get("graph_index_parquet")
        or phase2_outputs.get("graph_index_parquet")
        or "data/curated/main_benchmark_graph_index.parquet"
    )
    graphs_dir = _resolve_path(
        inputs.get("graphs_dir")
        or phase2_outputs.get("graphs_dir")
        or "data/curated/graphs"
    )
    instructions_path = _resolve_path(
        inputs.get("instructions_parquet")
        or phase2_inputs.get("instructions_parquet")
        or "data/curated/phase2_instructions.parquet"
    )
    split_root = _resolve_path(inputs.get("split_root") or "data/splits/main_benchmark")

    feature_index_out_path = _resolve_path(
        outputs.get("feature_index_parquet") or "data/features/main_benchmark/phase3_feature_index.parquet"
    )
    opcode_text_out_path = _resolve_path(
        outputs.get("opcode_text_corpus_parquet") or "data/features/main_benchmark/opcode_text_corpus.parquet"
    )
    codebert_out_path = _resolve_path(
        outputs.get("codebert_features_parquet") or "data/features/main_benchmark/codebert_features.parquet"
    )
    graph_features_out_path = _resolve_path(
        outputs.get("graph_level_features_parquet") or "data/features/main_benchmark/graph_level_features.parquet"
    )
    tfidf_out_path = _resolve_path(
        outputs.get("tfidf_features_parquet") or "data/features/main_benchmark/tfidf_features.parquet"
    )
    pattern_out_path = _resolve_path(
        outputs.get("pattern_features_parquet") or "data/features/main_benchmark/pattern_features.parquet"
    )
    report_json_path = _resolve_path(
        outputs.get("report_json") or "reports/phase3/feature_extraction_report.json"
    )
    dataset_card_md_path = _resolve_path(
        outputs.get("dataset_card_md") or "reports/phase3/feature_dataset_card.md"
    )
    run_manifest_json_path = _resolve_path(
        outputs.get("run_manifest_json") or "reports/phase3/phase3_run_manifest.json"
    )

    codebert = CodeBertExtractionConfig(
        model_name=str(codebert_cfg.get("model_name", "microsoft/codebert-base")).strip(),
        pooling=str(codebert_cfg.get("pooling", "cls")).strip().lower(),
        max_length=int(codebert_cfg.get("max_length", 256)),
        batch_size=int(codebert_cfg.get("batch_size", 16)),
        device=str(codebert_cfg.get("device", "cpu")).strip().lower(),
        local_files_only=bool(codebert_cfg.get("local_files_only", True)),
        sliding_window=bool(codebert_cfg.get("sliding_window", False)),
        stride=int(codebert_cfg.get("stride", 256)),
    )

    return Phase3Config(
        main_swc_ids=main_swc_ids,
        phase2_run_manifest_path=phase2_manifest_path,
        contracts_path=contracts_path,
        labels_path=labels_path,
        graph_index_path=graph_index_path,
        graphs_dir=graphs_dir,
        instructions_path=instructions_path,
        split_root=split_root,
        feature_index_out_path=feature_index_out_path,
        opcode_text_out_path=opcode_text_out_path,
        codebert_out_path=codebert_out_path,
        graph_features_out_path=graph_features_out_path,
        tfidf_out_path=tfidf_out_path,
        pattern_out_path=pattern_out_path,
        report_json_path=report_json_path,
        dataset_card_md_path=dataset_card_md_path,
        run_manifest_json_path=run_manifest_json_path,
        codebert=codebert,
    )


def _validate_inputs(config: Phase3Config) -> None:
    required_paths = [
        config.phase2_run_manifest_path,
        config.contracts_path,
        config.labels_path,
        config.graph_index_path,
        config.graphs_dir,
        config.instructions_path,
        config.split_root / "train.parquet",
        config.split_root / "val.parquet",
        config.split_root / "test.parquet",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required input artifact(s): {missing}")


def _as_bool(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return bool(value)


def _load_split_mapping(split_root: Path) -> Dict[str, str]:
    split_map: Dict[str, str] = {}
    for split in REQUIRED_SPLITS:
        path = split_root / f"{split}.parquet"
        frame = pd.read_parquet(path).copy()
        if CONTRACT_ID_COLUMN not in frame.columns:
            raise ValueError(f"Split file missing `{CONTRACT_ID_COLUMN}`: {path}")
        frame[CONTRACT_ID_COLUMN] = frame[CONTRACT_ID_COLUMN].fillna("").astype(str).str.strip()
        frame = frame[frame[CONTRACT_ID_COLUMN] != ""]
        duplicated = frame[CONTRACT_ID_COLUMN].duplicated()
        if bool(duplicated.any()):
            dup_ids = sorted(frame.loc[duplicated, CONTRACT_ID_COLUMN].tolist())
            raise ValueError(f"Duplicate contract IDs in split file {path}: {dup_ids[:10]}")
        for contract_id in frame[CONTRACT_ID_COLUMN].tolist():
            if contract_id in split_map:
                raise ValueError(f"Split overlap detected for contract `{contract_id}`.")
            split_map[contract_id] = split
    return split_map


def _validate_split_alignment(
    contracts: pd.DataFrame,
    graph_index: pd.DataFrame,
    split_map: Mapping[str, str],
) -> Dict[str, Any]:
    contracts_ids = set(contracts[CONTRACT_ID_COLUMN].tolist())
    graph_ids = set(graph_index[CONTRACT_ID_COLUMN].tolist())
    split_ids = set(split_map.keys())

    if contracts_ids != graph_ids:
        only_contracts = sorted(contracts_ids - graph_ids)
        only_graphs = sorted(graph_ids - contracts_ids)
        raise ValueError(
            "Mismatch between contracts parquet and graph index contract IDs. "
            f"only_contracts={only_contracts[:10]}, only_graphs={only_graphs[:10]}"
        )
    if contracts_ids != split_ids:
        only_contracts = sorted(contracts_ids - split_ids)
        only_splits = sorted(split_ids - contracts_ids)
        raise ValueError(
            "Mismatch between contracts parquet and split files contract IDs. "
            f"only_contracts={only_contracts[:10]}, only_splits={only_splits[:10]}"
        )

    split_mismatch = []
    for _, row in graph_index[[CONTRACT_ID_COLUMN, "split"]].iterrows():
        contract_id = str(row[CONTRACT_ID_COLUMN])
        observed = str(row["split"]).strip()
        expected = split_map.get(contract_id, "")
        if observed != expected:
            split_mismatch.append((contract_id, observed, expected))
            if len(split_mismatch) >= 10:
                break
    if split_mismatch:
        raise ValueError(f"Graph index split mismatch against split files: {split_mismatch}")

    split_counts = graph_index["split"].value_counts().to_dict()
    split_counts = {str(key): int(value) for key, value in split_counts.items()}
    return {
        "split_alignment_preserved": True,
        "split_counts": split_counts,
        "contracts_total": int(len(graph_index)),
    }


def _validate_swc_columns(graph_index: pd.DataFrame, swc_ids: Sequence[int]) -> Dict[str, Any]:
    expected_label_cols = [f"swc_{swc_id}" for swc_id in swc_ids]
    expected_mask_cols = [f"swc_{swc_id}_assessed" for swc_id in swc_ids]

    missing = [col for col in expected_label_cols + expected_mask_cols if col not in graph_index.columns]
    if missing:
        raise ValueError(f"Graph index missing SWC columns required for Phase 3: {missing}")

    observed_label_cols = [col for col in graph_index.columns if col in set(expected_label_cols)]
    observed_mask_cols = [col for col in graph_index.columns if col in set(expected_mask_cols)]
    if observed_label_cols != expected_label_cols:
        raise ValueError(
            "SWC label column order mismatch in graph index. "
            f"expected={expected_label_cols}, observed={observed_label_cols}"
        )
    if observed_mask_cols != expected_mask_cols:
        raise ValueError(
            "SWC assessed column order mismatch in graph index. "
            f"expected={expected_mask_cols}, observed={observed_mask_cols}"
        )
    return {
        "swc_order_preserved": True,
        "label_columns": expected_label_cols,
        "mask_columns": expected_mask_cols,
    }


def _first_non_empty(values: Iterable[Any]) -> str:
    for value in values:
        text = "" if value is None or pd.isna(value) else str(value).strip()
        if text:
            return text
    return ""


def _validate_labels(labels: pd.DataFrame, swc_ids: Sequence[int]) -> Dict[str, Any]:
    required = {CONTRACT_ID_COLUMN, "swc_id", "label"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"Labels parquet missing required columns: {missing}")
    working = labels.copy()
    working["swc_id"] = pd.to_numeric(working["swc_id"], errors="coerce").astype("Int64")
    working = working[working["swc_id"].notna()].copy()
    observed_swcs = sorted(working["swc_id"].astype(int).unique().tolist())
    expected_swcs_sorted = sorted(int(value) for value in swc_ids)
    if observed_swcs != expected_swcs_sorted:
        raise ValueError(
            "Label parquet SWCs mismatch main benchmark SWCs. "
            f"observed={observed_swcs}, expected={expected_swcs_sorted}"
        )
    return {"labels_rows": int(len(working)), "labels_swcs_verified": observed_swcs}


def _build_feature_index(
    graph_index: pd.DataFrame,
    opcode_corpus: pd.DataFrame,
    codebert_features: pd.DataFrame,
    graph_features: pd.DataFrame,
    tfidf_features: pd.DataFrame,
    pattern_features: pd.DataFrame,
    swc_ids: Sequence[int],
    failure_mode_index: Mapping[str, str],
) -> pd.DataFrame:
    required_opcode = {CONTRACT_ID_COLUMN, "opcode_text_available", "unavailable_cause"}
    required_codebert = {CONTRACT_ID_COLUMN, "codebert_feature_available"}
    required_graph = {CONTRACT_ID_COLUMN, "graph_feature_available", "graph_structure_available"}
    for frame_name, frame, required in [
        ("opcode corpus", opcode_corpus, required_opcode),
        ("codebert features", codebert_features, required_codebert),
        ("graph features", graph_features, required_graph),
    ]:
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{frame_name} missing required columns for feature index merge: {missing}")

    base_columns = [
        "graph_id",
        CONTRACT_ID_COLUMN,
        "artifact_path",
        "artifact_format",
        "split",
        "source_group",
        "has_cgt",
        "has_dappscan",
        "is_proxy_like",
        "is_stub_like",
        "graph_build_success",
        "instruction_parse_success",
        "cfg_success",
        "dfg_success",
        "node_count",
        "basic_block_count",
        "cfg_edge_count",
        "dfg_edge_count",
        "edge_count_total",
        "label_positive_count",
        "label_assessed_count",
        "label_vector",
        "label_mask",
    ]
    swc_label_columns = [f"swc_{swc_id}" for swc_id in swc_ids]
    swc_mask_columns = [f"swc_{swc_id}_assessed" for swc_id in swc_ids]
    selected = graph_index[base_columns + swc_label_columns + swc_mask_columns].copy()

    opcode_min = opcode_corpus[[CONTRACT_ID_COLUMN, "opcode_text_available", "unavailable_cause"]].copy()
    codebert_min = codebert_features[[CONTRACT_ID_COLUMN, "codebert_feature_available"]].copy()
    graph_min = graph_features[[CONTRACT_ID_COLUMN, "graph_feature_available", "graph_structure_available"]].copy()

    merged = selected.merge(opcode_min, on=CONTRACT_ID_COLUMN, how="left", validate="one_to_one")
    merged = merged.merge(codebert_min, on=CONTRACT_ID_COLUMN, how="left", validate="one_to_one")
    merged = merged.merge(graph_min, on=CONTRACT_ID_COLUMN, how="left", validate="one_to_one")

    # Track TF-IDF and pattern feature availability
    tfidf_ids = set(tfidf_features[CONTRACT_ID_COLUMN].astype(str).str.strip().tolist()) if not tfidf_features.empty else set()
    pattern_ids = set(pattern_features[CONTRACT_ID_COLUMN].astype(str).str.strip().tolist()) if not pattern_features.empty else set()
    merged["tfidf_feature_available"] = merged[CONTRACT_ID_COLUMN].isin(tfidf_ids)
    merged["pattern_feature_available"] = merged[CONTRACT_ID_COLUMN].isin(pattern_ids)

    merged["opcode_text_available"] = merged["opcode_text_available"].fillna(False).astype(bool)
    merged["codebert_feature_available"] = merged["codebert_feature_available"].fillna(False).astype(bool)
    merged["graph_feature_available"] = merged["graph_feature_available"].fillna(False).astype(bool)
    merged["graph_structure_available"] = merged["graph_structure_available"].fillna(False).astype(bool)

    merged["graph_unavailable"] = ~merged["graph_build_success"].fillna(False).astype(bool)
    merged["unavailable_cause"] = merged["unavailable_cause"].fillna("").astype(str).str.strip()

    for idx, row in merged.loc[merged["graph_unavailable"] & (merged["unavailable_cause"] == "")].iterrows():
        contract_id = str(row[CONTRACT_ID_COLUMN])
        failure_mode = str(failure_mode_index.get(contract_id, "")).strip()
        merged.at[idx, "unavailable_cause"] = failure_mode or "graph_unavailable"

    merged = merged.sort_values(["split", CONTRACT_ID_COLUMN]).reset_index(drop=True)
    return merged


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


def _build_dataset_card(report: Mapping[str, Any]) -> str:
    split_rows = [{"split": split, "contracts": report["split_counts"].get(split, 0)} for split in REQUIRED_SPLITS]
    row_count_rows = [
        {"artifact": key, "rows": value}
        for key, value in report.get("feature_row_counts", {}).items()
    ]
    unavailable_rows = [
        {"cause": key, "contracts": value}
        for key, value in sorted(report.get("unavailable_by_cause", {}).items())
    ]
    if not unavailable_rows:
        unavailable_rows = [{"cause": "none", "contracts": 0}]

    return (
        "# Phase 3 Feature Dataset Card\n\n"
        f"Generated at (UTC): `{report['generated_at_utc']}`\n\n"
        "## Scope\n\n"
        "Phase 3 packages train-ready features from Phase 2 artifacts only. "
        "No model training, baselines, or final experiments are included.\n\n"
        f"- Main benchmark SWCs (fixed order): `{report['main_benchmark_swcs']}`\n"
        f"- Contracts accounted for: **{report['contracts_total']}**\n"
        f"- Graph-unavailable contracts: **{report['graph_unavailable_count']}**\n\n"
        "## Split counts\n\n"
        + _markdown_table(split_rows, [("split", "split"), ("contracts", "contracts")])
        + "\n\n## Feature artifact row counts\n\n"
        + _markdown_table(row_count_rows, [("artifact", "artifact"), ("rows", "rows")])
        + "\n\n## Unavailable causes\n\n"
        + _markdown_table(unavailable_rows, [("cause", "cause"), ("contracts", "contracts")])
        + "\n\n## Outputs\n\n"
        + f"- `{report['outputs']['feature_index_parquet']}`\n"
        + f"- `{report['outputs']['opcode_text_corpus_parquet']}`\n"
        + f"- `{report['outputs']['codebert_features_parquet']}`\n"
        + f"- `{report['outputs']['graph_level_features_parquet']}`\n"
        + f"- `{report['outputs']['report_json']}`\n"
        + f"- `{report['outputs']['dataset_card_md']}`\n"
        + f"- `{report['outputs']['run_manifest_json']}`\n"
    )


def build_phase3_features(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    run_started = datetime.now(timezone.utc)
    config = _load_config(config_path.resolve())
    _validate_inputs(config)

    contracts = pd.read_parquet(config.contracts_path).copy()
    labels = pd.read_parquet(config.labels_path).copy()
    graph_index = pd.read_parquet(config.graph_index_path).copy()
    instructions = pd.read_parquet(config.instructions_path).copy()

    for frame in [contracts, labels, graph_index, instructions]:
        if CONTRACT_ID_COLUMN in frame.columns:
            frame[CONTRACT_ID_COLUMN] = frame[CONTRACT_ID_COLUMN].fillna("").astype(str).str.strip()

    required_contract_cols = {
        CONTRACT_ID_COLUMN,
        "split",
        "has_cgt",
        "has_dappscan",
        "is_proxy_like",
        "is_stub_like",
    }
    missing_contract_cols = sorted(required_contract_cols - set(contracts.columns))
    if missing_contract_cols:
        raise ValueError(f"Contracts parquet missing required columns: {missing_contract_cols}")

    required_graph_cols = {
        CONTRACT_ID_COLUMN,
        "graph_id",
        "artifact_path",
        "split",
        "source_group",
        "has_cgt",
        "has_dappscan",
        "is_proxy_like",
        "is_stub_like",
        "graph_build_success",
        "instruction_parse_success",
        "cfg_success",
        "dfg_success",
        "node_count",
    }
    missing_graph_cols = sorted(required_graph_cols - set(graph_index.columns))
    if missing_graph_cols:
        raise ValueError(f"Graph index missing required columns: {missing_graph_cols}")

    contracts = contracts[contracts[CONTRACT_ID_COLUMN] != ""].drop_duplicates(subset=[CONTRACT_ID_COLUMN], keep="first")
    graph_index = graph_index[graph_index[CONTRACT_ID_COLUMN] != ""].drop_duplicates(
        subset=[CONTRACT_ID_COLUMN], keep="first"
    )
    if contracts.empty or graph_index.empty:
        raise ValueError("Contracts or graph index is empty after cleaning contract IDs.")

    labels_summary = _validate_labels(labels, config.main_swc_ids)
    swc_validation = _validate_swc_columns(graph_index, config.main_swc_ids)
    split_map = _load_split_mapping(config.split_root)
    split_validation = _validate_split_alignment(contracts, graph_index, split_map)
    failure_mode_index = build_failure_mode_index(instructions)

    opcode_corpus, opcode_summary = build_opcode_text_corpus(
        graph_index=graph_index,
        project_root=PROJECT_ROOT,
        failure_mode_by_contract=failure_mode_index,
    )
    codebert_features, codebert_summary = extract_codebert_features(opcode_corpus, config.codebert)
    graph_features, graph_feature_summary = build_graph_level_features(graph_index, opcode_corpus)

    # Save opcode corpus first so TF-IDF / pattern can read from disk
    output_notes: List[Dict[str, str]] = []
    resolved_opcode_text = _resolve_output_path(config.opcode_text_out_path, output_notes)
    resolved_opcode_text.parent.mkdir(parents=True, exist_ok=True)
    opcode_corpus.to_parquet(resolved_opcode_text, index=False)

    # Build TF-IDF features (fits on train split, transforms all)
    resolved_tfidf = _resolve_output_path(config.tfidf_out_path, output_notes)
    resolved_tfidf.parent.mkdir(parents=True, exist_ok=True)
    tfidf_features = build_tfidf_features(
        corpus_path=resolved_opcode_text,
        output_path=resolved_tfidf,
    )
    tfidf_summary = {
        "rows": int(len(tfidf_features)),
        "feature_columns": int(len([c for c in tfidf_features.columns if c.startswith("tfidf_")])),
    }

    # Build expert pattern features
    resolved_pattern = _resolve_output_path(config.pattern_out_path, output_notes)
    resolved_pattern.parent.mkdir(parents=True, exist_ok=True)
    pattern_features = build_pattern_features(
        corpus_path=resolved_opcode_text,
        output_path=resolved_pattern,
    )
    pattern_summary = {
        "rows": int(len(pattern_features)),
        "feature_columns": int(len([c for c in pattern_features.columns if c.startswith("pat_")])),
    }

    feature_index = _build_feature_index(
        graph_index=graph_index,
        opcode_corpus=opcode_corpus,
        codebert_features=codebert_features,
        graph_features=graph_features,
        tfidf_features=tfidf_features,
        pattern_features=pattern_features,
        swc_ids=config.main_swc_ids,
        failure_mode_index=failure_mode_index,
    )

    if len(feature_index) != len(graph_index):
        raise ValueError("Feature index row count mismatch after assembly.")
    if feature_index[CONTRACT_ID_COLUMN].nunique() != len(feature_index):
        raise ValueError("Feature index contains duplicate contract IDs.")

    resolved_feature_index = _resolve_output_path(config.feature_index_out_path, output_notes)
    resolved_codebert = _resolve_output_path(config.codebert_out_path, output_notes)
    resolved_graph_features = _resolve_output_path(config.graph_features_out_path, output_notes)
    resolved_report = _resolve_output_path(config.report_json_path, output_notes)
    resolved_dataset_card = _resolve_output_path(config.dataset_card_md_path, output_notes)
    resolved_run_manifest = _resolve_output_path(config.run_manifest_json_path, output_notes)

    resolved_feature_index.parent.mkdir(parents=True, exist_ok=True)
    resolved_codebert.parent.mkdir(parents=True, exist_ok=True)
    resolved_graph_features.parent.mkdir(parents=True, exist_ok=True)

    feature_index.to_parquet(resolved_feature_index, index=False)
    codebert_features.to_parquet(resolved_codebert, index=False)
    graph_features.to_parquet(resolved_graph_features, index=False)

    unavailable_rows = feature_index[feature_index["graph_unavailable"]].copy()
    unavailable_by_cause = unavailable_rows["unavailable_cause"].fillna("").astype(str).str.strip().replace("", "unknown")
    unavailable_by_cause = unavailable_by_cause.value_counts().to_dict()
    unavailable_by_cause = {str(key): int(value) for key, value in unavailable_by_cause.items()}

    split_counts = feature_index["split"].value_counts().to_dict()
    split_counts = {str(key): int(value) for key, value in split_counts.items()}

    report: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "phase3_feature_extraction",
        "main_benchmark_swcs": list(config.main_swc_ids),
        "inputs": {
            "phase2_graph_builder_run_manifest_json": _rel(config.phase2_run_manifest_path),
            "contracts_parquet": _rel(config.contracts_path),
            "labels_parquet": _rel(config.labels_path),
            "graph_index_parquet": _rel(config.graph_index_path),
            "graphs_dir": _rel(config.graphs_dir),
            "instructions_parquet": _rel(config.instructions_path),
            "split_root": _rel(config.split_root),
        },
        "outputs": {
            "feature_index_parquet": _rel(resolved_feature_index),
            "opcode_text_corpus_parquet": _rel(resolved_opcode_text),
            "codebert_features_parquet": _rel(resolved_codebert),
            "graph_level_features_parquet": _rel(resolved_graph_features),
            "tfidf_features_parquet": _rel(resolved_tfidf),
            "pattern_features_parquet": _rel(resolved_pattern),
            "report_json": _rel(resolved_report),
            "dataset_card_md": _rel(resolved_dataset_card),
            "run_manifest_json": _rel(resolved_run_manifest),
        },
        "configured_outputs": {
            "feature_index_parquet": _rel(config.feature_index_out_path),
            "opcode_text_corpus_parquet": _rel(config.opcode_text_out_path),
            "codebert_features_parquet": _rel(config.codebert_out_path),
            "graph_level_features_parquet": _rel(config.graph_features_out_path),
            "tfidf_features_parquet": _rel(config.tfidf_out_path),
            "pattern_features_parquet": _rel(config.pattern_out_path),
            "report_json": _rel(config.report_json_path),
            "dataset_card_md": _rel(config.dataset_card_md_path),
            "run_manifest_json": _rel(config.run_manifest_json_path),
        },
        "output_resolution_notes": output_notes,
        "contracts_total": int(len(feature_index)),
        "split_counts": split_counts,
        "feature_row_counts": {
            "feature_index": int(len(feature_index)),
            "opcode_text_corpus": int(len(opcode_corpus)),
            "codebert_features": int(len(codebert_features)),
            "graph_level_features": int(len(graph_features)),
            "tfidf_features": int(len(tfidf_features)),
            "pattern_features": int(len(pattern_features)),
        },
        "graph_unavailable_count": int(feature_index["graph_unavailable"].sum()),
        "unavailable_by_cause": unavailable_by_cause,
        "preservation_checks": {
            "split_alignment_preserved": bool(split_validation["split_alignment_preserved"]),
            "swc_order_preserved": bool(swc_validation["swc_order_preserved"]),
            "provenance_columns_preserved": bool(
                {"source_group", "has_cgt", "has_dappscan"}.issubset(set(feature_index.columns))
            ),
            "proxy_stub_columns_preserved": bool(
                {"is_proxy_like", "is_stub_like"}.issubset(set(feature_index.columns))
            ),
        },
        "codebert": codebert_summary,
        "opcode_text": opcode_summary,
        "graph_features": graph_feature_summary,
        "tfidf": tfidf_summary,
        "pattern": pattern_summary,
        "labels_validation": labels_summary,
    }

    dataset_card = _build_dataset_card(report)
    _write_json(report, resolved_report)
    _write_text(dataset_card, resolved_dataset_card)

    run_manifest = {
        "run_started_utc": run_started.isoformat(),
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "entrypoint": "python -m src.features.build_phase3_features --config configs/phase3.yaml",
        "config_path": _rel(config_path.resolve()),
        "inputs": report["inputs"],
        "outputs": report["outputs"],
        "summary": {
            "contracts_total": report["contracts_total"],
            "graph_unavailable_count": report["graph_unavailable_count"],
            "feature_row_counts": report["feature_row_counts"],
        },
    }
    _write_json(run_manifest, resolved_run_manifest)
    return report


def _print_summary(report: Mapping[str, Any]) -> None:
    print(f"contracts total: {report['contracts_total']}")
    print(f"graph unavailable: {report['graph_unavailable_count']}")
    print(f"feature index: {report['outputs']['feature_index_parquet']}")
    print(f"opcode corpus: {report['outputs']['opcode_text_corpus_parquet']}")
    print(f"codebert features: {report['outputs']['codebert_features_parquet']}")
    print(f"graph features: {report['outputs']['graph_level_features_parquet']}")
    print(f"tfidf features: {report['outputs']['tfidf_features_parquet']}")
    print(f"pattern features: {report['outputs']['pattern_features_parquet']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 3 feature artifacts from Phase 2 graph outputs.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    report = build_phase3_features(args.config.resolve())
    _print_summary(report)


if __name__ == "__main__":
    main()
