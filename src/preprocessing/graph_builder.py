import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs/phase2.yaml"
DEFAULT_MAIN_BENCHMARK_SWCS = [101, 103, 104, 107, 113, 114, 115, 120, 128, 132, 135]
DEFAULT_INPUT_CONTRACTS = PROJECT_ROOT / "data/curated/main_benchmark_contracts.parquet"
DEFAULT_INPUT_LABELS = PROJECT_ROOT / "data/curated/main_benchmark_labels.parquet"
DEFAULT_INPUT_INSTRUCTIONS = PROJECT_ROOT / "data/curated/phase2_instructions.parquet"
DEFAULT_INPUT_BASIC_BLOCKS = PROJECT_ROOT / "data/curated/phase2_basic_blocks.parquet"
DEFAULT_INPUT_CFG_NODES = PROJECT_ROOT / "data/curated/phase2_cfg_nodes.parquet"
DEFAULT_INPUT_CFG_EDGES = PROJECT_ROOT / "data/curated/phase2_cfg_edges.parquet"
DEFAULT_INPUT_DFG_EDGES = PROJECT_ROOT / "data/curated/phase2_dfg_edges.parquet"
DEFAULT_OUTPUT_GRAPH_INDEX = PROJECT_ROOT / "data/curated/main_benchmark_graph_index.parquet"
DEFAULT_OUTPUT_GRAPHS_DIR = PROJECT_ROOT / "data/curated/graphs"
DEFAULT_OUTPUT_QUALITY_REPORT = PROJECT_ROOT / "reports/phase2/graph_quality_report.json"
DEFAULT_OUTPUT_PREPROCESSING_SUMMARY = PROJECT_ROOT / "reports/phase2/preprocessing_summary.md"
DEFAULT_OUTPUT_VARIANT_MANIFESTS_DIR = PROJECT_ROOT / "data/splits/main_benchmark/manifests"
DEFAULT_OUTPUT_RUN_MANIFEST = PROJECT_ROOT / "reports/phase2/graph_builder_run_manifest.json"

CONTRACT_ID_COLUMN = "fp_runtime_unified"
VALID_GRAPH_FORMATS = {"pt"}
CFG_EDGE_TYPES = {"fallthrough", "jump", "jumpi_true", "jumpi_false", "terminal"}
DFG_EDGE_TYPES = {"stack_flow", "storage_flow", "memory_flow", "control_dependency"}
EDGE_TYPE_VOCAB = [
    "cfg_fallthrough",
    "cfg_jump",
    "cfg_jumpi_true",
    "cfg_jumpi_false",
    "cfg_terminal",
    "dfg_stack_flow",
    "dfg_storage_flow",
    "dfg_memory_flow",
    "dfg_control_dependency",
]
EDGE_TYPE_TO_ID = {name: idx for idx, name in enumerate(EDGE_TYPE_VOCAB)}
REQUIRED_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class GraphConfig:
    main_swc_ids: List[int]
    input_contracts_path: Path
    input_labels_path: Path
    input_instructions_path: Path
    input_basic_blocks_path: Path
    input_cfg_nodes_path: Path
    input_cfg_edges_path: Path
    input_dfg_edges_path: Path
    graph_index_out_path: Path
    graphs_dir: Path
    quality_report_json_path: Path
    preprocessing_summary_md_path: Path
    variant_manifests_dir: Path
    run_manifest_json_path: Path
    graph_artifact_format: str


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


def _write_json(payload: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def _write_text(text: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")


def _safe_read_mapping(config_path: Path) -> Mapping[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp)
    if not isinstance(data, dict):
        raise ValueError("Phase 2 config must be a mapping.")
    return data


def _load_config(config_path: Path) -> GraphConfig:
    raw = _safe_read_mapping(config_path)
    benchmark = raw.get("main_benchmark", {})
    if benchmark is None:
        benchmark = {}
    if not isinstance(benchmark, dict):
        raise ValueError("`main_benchmark` must be a mapping when provided.")
    configured_swcs = benchmark.get("swc_ids", DEFAULT_MAIN_BENCHMARK_SWCS)
    if not isinstance(configured_swcs, list) or not configured_swcs:
        raise ValueError("`main_benchmark.swc_ids` must be a non-empty list.")
    swc_ids: List[int] = []
    seen_swcs = set()
    for value in configured_swcs:
        swc_id = int(value)
        if swc_id not in seen_swcs:
            swc_ids.append(swc_id)
            seen_swcs.add(swc_id)

    graph_cfg = raw.get("graph", {})
    if graph_cfg is None:
        graph_cfg = {}
    if not isinstance(graph_cfg, dict):
        raise ValueError("`graph` must be a mapping when provided.")

    graph_format = str(graph_cfg.get("graph_artifact_format", "pt")).strip().lower()
    if graph_format not in VALID_GRAPH_FORMATS:
        raise ValueError(f"Unsupported graph format `{graph_format}`. Supported: {sorted(VALID_GRAPH_FORMATS)}")

    return GraphConfig(
        main_swc_ids=swc_ids,
        input_contracts_path=_resolve_path(graph_cfg.get("input_contracts") or DEFAULT_INPUT_CONTRACTS),
        input_labels_path=_resolve_path(graph_cfg.get("input_labels") or DEFAULT_INPUT_LABELS),
        input_instructions_path=_resolve_path(graph_cfg.get("input_instructions") or DEFAULT_INPUT_INSTRUCTIONS),
        input_basic_blocks_path=_resolve_path(graph_cfg.get("input_basic_blocks") or DEFAULT_INPUT_BASIC_BLOCKS),
        input_cfg_nodes_path=_resolve_path(graph_cfg.get("input_cfg_nodes") or DEFAULT_INPUT_CFG_NODES),
        input_cfg_edges_path=_resolve_path(graph_cfg.get("input_cfg_edges") or DEFAULT_INPUT_CFG_EDGES),
        input_dfg_edges_path=_resolve_path(graph_cfg.get("input_dfg_edges") or DEFAULT_INPUT_DFG_EDGES),
        graph_index_out_path=_resolve_path(graph_cfg.get("graph_index_parquet") or DEFAULT_OUTPUT_GRAPH_INDEX),
        graphs_dir=_resolve_path(graph_cfg.get("graphs_dir") or DEFAULT_OUTPUT_GRAPHS_DIR),
        quality_report_json_path=_resolve_path(graph_cfg.get("quality_report_json") or DEFAULT_OUTPUT_QUALITY_REPORT),
        preprocessing_summary_md_path=_resolve_path(
            graph_cfg.get("preprocessing_summary_md") or DEFAULT_OUTPUT_PREPROCESSING_SUMMARY
        ),
        variant_manifests_dir=_resolve_path(
            graph_cfg.get("variant_manifests_dir") or DEFAULT_OUTPUT_VARIANT_MANIFESTS_DIR
        ),
        run_manifest_json_path=_resolve_path(graph_cfg.get("run_manifest_json") or DEFAULT_OUTPUT_RUN_MANIFEST),
        graph_artifact_format=graph_format,
    )


def _resolve_output_path(configured_path: Path, output_notes: List[Dict[str, str]]) -> Path:
    if not configured_path.exists():
        return configured_path

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = configured_path.with_name(f"{configured_path.stem}_{timestamp}{configured_path.suffix}")
    suffix = 1
    while candidate.exists():
        candidate = configured_path.with_name(
            f"{configured_path.stem}_{timestamp}_{suffix}{configured_path.suffix}"
        )
        suffix += 1

    output_notes.append(
        {
            "configured_path": _rel(configured_path),
            "resolved_path": _rel(candidate),
            "reason": "configured_output_already_exists",
        }
    )
    return candidate


def _as_nullable_bool(value: Any) -> Optional[bool]:
    if pd.isna(value):
        return None
    return bool(value)


def _as_int(value: Any, default: int = 0) -> int:
    if value is None or pd.isna(value):
        return int(default)
    return int(value)


def _safe_json_list(raw: Any) -> List[str]:
    if raw is None or pd.isna(raw):
        return []
    text = str(raw).strip()
    if not text:
        return []
    if not (text.startswith("[") and text.endswith("]")):
        return [text]
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def _quantiles(values: Iterable[int]) -> Dict[str, Optional[float]]:
    series = pd.Series(list(values), dtype="float64")
    if series.empty:
        return {"median": None, "p95": None}
    return {"median": float(series.quantile(0.5)), "p95": float(series.quantile(0.95))}


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


def _aggregate_label_value(values: pd.Series) -> Any:
    known_values = sorted({int(v) for v in values.dropna().tolist()})
    if not known_values:
        return pd.NA
    if len(known_values) == 1:
        return int(known_values[0])
    return pd.NA


def _load_label_matrix(labels: pd.DataFrame, swc_ids: Sequence[int]) -> Tuple[pd.DataFrame, int]:
    required_cols = {CONTRACT_ID_COLUMN, "swc_id", "label"}
    missing_cols = sorted(required_cols - set(labels.columns))
    if missing_cols:
        raise ValueError(f"Input labels missing required columns: {missing_cols}")

    working = labels.copy()
    working[CONTRACT_ID_COLUMN] = working[CONTRACT_ID_COLUMN].fillna("").astype(str).str.strip()
    working = working[working[CONTRACT_ID_COLUMN] != ""]
    working["swc_id"] = pd.to_numeric(working["swc_id"], errors="coerce").astype("Int64")
    working["label"] = pd.to_numeric(working["label"], errors="coerce").astype("Int64")
    working = working[working["swc_id"].notna()].copy()
    working["swc_id"] = working["swc_id"].astype(int)
    working = working[working["swc_id"].isin(swc_ids)].copy()

    grouped = (
        working.groupby([CONTRACT_ID_COLUMN, "swc_id"], sort=True)["label"].agg(list).reset_index(name="label_values")
    )
    conflict_count = int(grouped["label_values"].apply(lambda vals: len(sorted({int(v) for v in vals if pd.notna(v)})) > 1).sum())
    grouped["label"] = grouped["label_values"].apply(lambda vals: _aggregate_label_value(pd.Series(vals, dtype="Int64")))
    grouped = grouped.drop(columns=["label_values"])

    pivot = grouped.pivot(index=CONTRACT_ID_COLUMN, columns="swc_id", values="label")
    for swc_id in swc_ids:
        if swc_id not in pivot.columns:
            pivot[swc_id] = pd.NA
    pivot = pivot.reindex(columns=list(swc_ids)).sort_index()
    return pivot, conflict_count


def _build_node_table(contract_instructions: pd.DataFrame) -> Tuple[Dict[str, List[Any]], bool, str]:
    if contract_instructions.empty:
        return {
            "node_id": [],
            "pc": [],
            "opcode": [],
            "opcode_id": [],
            "push_data": [],
            "size": [],
            "basic_block_id": [],
        }, False, "missing_instruction_rows"

    parse_success = contract_instructions["parse_success"].fillna(False).astype(bool)
    success_rows = contract_instructions[
        parse_success & contract_instructions["pc"].notna() & contract_instructions["opcode_id"].notna()
    ].copy()
    if success_rows.empty:
        failure_mode = ""
        if "failure_mode" in contract_instructions.columns:
            for value in contract_instructions["failure_mode"].tolist():
                text = "" if value is None or pd.isna(value) else str(value).strip()
                if text:
                    failure_mode = text
                    break
        return {
            "node_id": [],
            "pc": [],
            "opcode": [],
            "opcode_id": [],
            "push_data": [],
            "size": [],
            "basic_block_id": [],
        }, False, failure_mode or "instruction_parse_failed"

    success_rows["pc"] = pd.to_numeric(success_rows["pc"], errors="coerce").astype("Int64")
    success_rows["opcode_id"] = pd.to_numeric(success_rows["opcode_id"], errors="coerce").astype("Int64")
    success_rows["size"] = pd.to_numeric(success_rows["size"], errors="coerce").astype("Int64")
    success_rows["basic_block_id"] = pd.to_numeric(success_rows["basic_block_id"], errors="coerce").astype("Int64")
    success_rows = success_rows[success_rows["pc"].notna() & success_rows["opcode_id"].notna()].copy()
    success_rows["pc"] = success_rows["pc"].astype(int)
    success_rows["opcode_id"] = success_rows["opcode_id"].astype(int)
    success_rows = success_rows.sort_values("pc").drop_duplicates(subset=["pc"], keep="first").reset_index(drop=True)

    node_table = {
        "node_id": success_rows["pc"].astype(int).tolist(),
        "pc": success_rows["pc"].astype(int).tolist(),
        "opcode": success_rows["opcode"].fillna("").astype(str).tolist(),
        "opcode_id": success_rows["opcode_id"].astype(int).tolist(),
        "push_data": success_rows["push_data"].fillna("").astype(str).tolist(),
        "size": success_rows["size"].fillna(0).astype(int).tolist(),
        "basic_block_id": success_rows["basic_block_id"].fillna(-1).astype(int).tolist(),
    }
    return node_table, True, ""


def _build_block_end_map(contract_basic_blocks: pd.DataFrame) -> Dict[int, int]:
    if contract_basic_blocks.empty:
        return {}
    required = {"parse_success", "start_pc", "end_pc"}
    if not required.issubset(set(contract_basic_blocks.columns)):
        return {}
    rows = contract_basic_blocks[
        contract_basic_blocks["parse_success"].fillna(False).astype(bool)
        & contract_basic_blocks["start_pc"].notna()
        & contract_basic_blocks["end_pc"].notna()
    ].copy()
    if rows.empty:
        return {}
    rows["start_pc"] = pd.to_numeric(rows["start_pc"], errors="coerce").astype("Int64")
    rows["end_pc"] = pd.to_numeric(rows["end_pc"], errors="coerce").astype("Int64")
    rows = rows[rows["start_pc"].notna() & rows["end_pc"].notna()]
    mapping: Dict[int, int] = {}
    for _, row in rows.iterrows():
        mapping[int(row["start_pc"])] = int(row["end_pc"])
    return mapping


def _build_cfg_instruction_edges(
    contract_cfg_edges: pd.DataFrame,
    node_id_set: Set[int],
    block_end_map: Dict[int, int],
) -> List[Dict[str, Any]]:
    if contract_cfg_edges.empty:
        return []
    required = {"cfg_success", "src_node_id", "dst_node_id", "edge_type"}
    if not required.issubset(set(contract_cfg_edges.columns)):
        return []

    rows = contract_cfg_edges[
        contract_cfg_edges["cfg_success"].fillna(False).astype(bool)
        & contract_cfg_edges["src_node_id"].notna()
        & contract_cfg_edges["dst_node_id"].notna()
    ].copy()
    if rows.empty:
        return []
    rows["src_node_id"] = pd.to_numeric(rows["src_node_id"], errors="coerce").astype("Int64")
    rows["dst_node_id"] = pd.to_numeric(rows["dst_node_id"], errors="coerce").astype("Int64")
    rows = rows[rows["src_node_id"].notna() & rows["dst_node_id"].notna()]

    edges: List[Dict[str, Any]] = []
    seen = set()
    for _, row in rows.iterrows():
        src_block_start = int(row["src_node_id"])
        dst_block_start = int(row["dst_node_id"])
        edge_type = str(row["edge_type"]).strip()
        if edge_type not in CFG_EDGE_TYPES:
            continue

        src_pc = block_end_map.get(src_block_start, src_block_start)
        dst_pc = dst_block_start
        if src_pc not in node_id_set or dst_pc not in node_id_set:
            continue

        key = (src_pc, dst_pc, edge_type)
        if key in seen:
            continue
        seen.add(key)
        edges.append(
            {
                "src_node_id": int(src_pc),
                "dst_node_id": int(dst_pc),
                "edge_family": "cfg",
                "edge_type": edge_type,
                "storage_slot": None,
                "memory_offset": None,
            }
        )
    return edges


def _build_dfg_edges(contract_dfg_edges: pd.DataFrame, node_id_set: Set[int]) -> List[Dict[str, Any]]:
    if contract_dfg_edges.empty:
        return []
    required = {"dfg_success", "src_instruction_pc", "dst_instruction_pc", "edge_type"}
    if not required.issubset(set(contract_dfg_edges.columns)):
        return []

    rows = contract_dfg_edges[
        contract_dfg_edges["dfg_success"].fillna(False).astype(bool)
        & contract_dfg_edges["src_instruction_pc"].notna()
        & contract_dfg_edges["dst_instruction_pc"].notna()
    ].copy()
    if rows.empty:
        return []
    rows["src_instruction_pc"] = pd.to_numeric(rows["src_instruction_pc"], errors="coerce").astype("Int64")
    rows["dst_instruction_pc"] = pd.to_numeric(rows["dst_instruction_pc"], errors="coerce").astype("Int64")
    if "storage_slot" in rows.columns:
        rows["storage_slot"] = pd.to_numeric(rows["storage_slot"], errors="coerce").astype("Int64")
    else:
        rows["storage_slot"] = pd.Series([pd.NA] * len(rows), index=rows.index, dtype="Int64")
    if "memory_offset" in rows.columns:
        rows["memory_offset"] = pd.to_numeric(rows["memory_offset"], errors="coerce").astype("Int64")
    else:
        rows["memory_offset"] = pd.Series([pd.NA] * len(rows), index=rows.index, dtype="Int64")
    rows = rows[rows["src_instruction_pc"].notna() & rows["dst_instruction_pc"].notna()]

    edges: List[Dict[str, Any]] = []
    seen = set()
    for _, row in rows.iterrows():
        src_pc = int(row["src_instruction_pc"])
        dst_pc = int(row["dst_instruction_pc"])
        edge_type = str(row["edge_type"]).strip()
        if edge_type not in DFG_EDGE_TYPES:
            continue
        if src_pc not in node_id_set or dst_pc not in node_id_set:
            continue

        storage_slot = None if pd.isna(row["storage_slot"]) else int(row["storage_slot"])
        memory_offset = None if pd.isna(row["memory_offset"]) else int(row["memory_offset"])
        key = (src_pc, dst_pc, edge_type, storage_slot, memory_offset)
        if key in seen:
            continue
        seen.add(key)
        edges.append(
            {
                "src_node_id": src_pc,
                "dst_node_id": dst_pc,
                "edge_family": "dfg",
                "edge_type": edge_type,
                "storage_slot": storage_slot,
                "memory_offset": memory_offset,
            }
        )
    return edges


def _edge_table_to_columns(edges: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    return {
        "src_node_id": [int(edge["src_node_id"]) for edge in edges],
        "dst_node_id": [int(edge["dst_node_id"]) for edge in edges],
        "edge_family": [str(edge["edge_family"]) for edge in edges],
        "edge_type": [str(edge["edge_type"]) for edge in edges],
        "storage_slot": [edge["storage_slot"] for edge in edges],
        "memory_offset": [edge["memory_offset"] for edge in edges],
    }


def _build_pyg_payload(
    node_table: Dict[str, List[Any]],
    edges: List[Dict[str, Any]],
    label_vector: List[int],
    label_mask: List[bool],
) -> Dict[str, Any]:
    pcs = [int(pc) for pc in node_table["pc"]]
    opcode_ids = [int(value) for value in node_table["opcode_id"]]
    basic_block_ids = [int(value) for value in node_table["basic_block_id"]]
    node_index = {pc: idx for idx, pc in enumerate(pcs)}

    x = torch.tensor(list(zip(opcode_ids, pcs, basic_block_ids)), dtype=torch.long) if pcs else torch.zeros((0, 3), dtype=torch.long)
    if edges:
        src_indices = []
        dst_indices = []
        edge_type_ids = []
        cfg_src = []
        cfg_dst = []
        dfg_src = []
        dfg_dst = []
        for edge in edges:
            src_idx = node_index[int(edge["src_node_id"])]
            dst_idx = node_index[int(edge["dst_node_id"])]
            token = f"{edge['edge_family']}_{edge['edge_type']}"
            if token not in EDGE_TYPE_TO_ID:
                continue
            src_indices.append(src_idx)
            dst_indices.append(dst_idx)
            edge_type_ids.append(EDGE_TYPE_TO_ID[token])
            if edge["edge_family"] == "cfg":
                cfg_src.append(src_idx)
                cfg_dst.append(dst_idx)
            else:
                dfg_src.append(src_idx)
                dfg_dst.append(dst_idx)
        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long) if src_indices else torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.tensor(edge_type_ids, dtype=torch.long) if edge_type_ids else torch.zeros((0,), dtype=torch.long)
        cfg_edge_index = torch.tensor([cfg_src, cfg_dst], dtype=torch.long) if cfg_src else torch.zeros((2, 0), dtype=torch.long)
        dfg_edge_index = torch.tensor([dfg_src, dfg_dst], dtype=torch.long) if dfg_src else torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros((0,), dtype=torch.long)
        cfg_edge_index = torch.zeros((2, 0), dtype=torch.long)
        dfg_edge_index = torch.zeros((2, 0), dtype=torch.long)

    y = torch.tensor(label_vector, dtype=torch.float32)
    y_mask = torch.tensor(label_mask, dtype=torch.bool)

    return {
        "x": x,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "cfg_edge_index": cfg_edge_index,
        "dfg_edge_index": dfg_edge_index,
        "y": y,
        "y_mask": y_mask,
        "num_nodes": int(len(pcs)),
    }


def _label_vector_for_contract(label_matrix: pd.DataFrame, contract_id: str, swc_ids: Sequence[int]) -> Tuple[List[int], List[bool]]:
    if contract_id not in label_matrix.index:
        return [-1 for _ in swc_ids], [False for _ in swc_ids]
    row = label_matrix.loc[contract_id]
    vector: List[int] = []
    mask: List[bool] = []
    for swc_id in swc_ids:
        value = row.get(swc_id, pd.NA)
        if pd.isna(value):
            vector.append(-1)
            mask.append(False)
        else:
            vector.append(int(value))
            mask.append(True)
    return vector, mask


def load_graph_artifact(path: Path) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Graph artifact must deserialize to a dict: {path}")
    return payload


def _validate_graph_index_mapping(index_df: pd.DataFrame, graphs_run_dir: Path) -> Dict[str, Any]:
    required_cols = {"graph_id", CONTRACT_ID_COLUMN, "artifact_path", "artifact_format"}
    missing = sorted(required_cols - set(index_df.columns))
    if missing:
        raise ValueError(f"Graph index missing required columns: {missing}")
    if index_df.empty:
        raise ValueError("Graph index is empty; expected at least one graph artifact.")

    graph_id_dups = index_df["graph_id"].duplicated()
    if bool(graph_id_dups.any()):
        duplicates = sorted(index_df.loc[graph_id_dups, "graph_id"].tolist())
        raise ValueError(f"Graph index has duplicate graph_id entries: {duplicates[:10]}")

    path_dups = index_df["artifact_path"].duplicated()
    if bool(path_dups.any()):
        duplicates = sorted(index_df.loc[path_dups, "artifact_path"].tolist())
        raise ValueError(f"Graph index has duplicate artifact_path entries: {duplicates[:10]}")

    missing_files = []
    for rel_path in index_df["artifact_path"].tolist():
        artifact_path = _resolve_path(rel_path)
        if not artifact_path.exists():
            missing_files.append(rel_path)
    if missing_files:
        raise ValueError(f"Graph index references missing artifacts: {missing_files[:10]}")

    expected_paths = {_resolve_path(path) for path in index_df["artifact_path"].tolist()}
    actual_paths = set(graphs_run_dir.glob("*.pt"))
    if expected_paths != actual_paths:
        missing_in_index = sorted(_rel(path) for path in (actual_paths - expected_paths))
        missing_on_disk = sorted(_rel(path) for path in (expected_paths - actual_paths))
        raise ValueError(
            "Graph index to artifact mismatch. "
            f"on_disk_not_indexed={missing_in_index[:10]}, indexed_not_on_disk={missing_on_disk[:10]}"
        )

    return {
        "mapping_valid": True,
        "rows": int(len(index_df)),
        "artifact_files": int(len(actual_paths)),
    }


def _source_group(has_cgt: Optional[bool], has_dappscan: Optional[bool]) -> str:
    cgt = bool(has_cgt) if has_cgt is not None else False
    dappscan = bool(has_dappscan) if has_dappscan is not None else False
    if cgt and dappscan:
        return "both"
    if cgt:
        return "cgt_only"
    if dappscan:
        return "dappscan_only"
    return "unknown"


def _create_variant_manifests(
    index_df: pd.DataFrame,
    manifests_dir: Path,
    graph_index_path: Path,
) -> Dict[str, Dict[str, Any]]:
    manifests_dir.mkdir(parents=True, exist_ok=True)

    clean_default = index_df[index_df["graph_build_success"]].copy()
    no_proxy = clean_default[
        ~(clean_default["is_proxy_like"].fillna(False).astype(bool) | clean_default["is_stub_like"].fillna(False).astype(bool))
    ].copy()
    cgt_only = clean_default[
        clean_default["has_cgt"].fillna(False).astype(bool) & ~clean_default["has_dappscan"].fillna(False).astype(bool)
    ].copy()
    combined_posaug = clean_default[
        clean_default["has_cgt"].fillna(False).astype(bool) | clean_default["has_dappscan"].fillna(False).astype(bool)
    ].copy()

    variant_frames = {
        "clean_default": (
            clean_default,
            "All build-success graphs for main benchmark with default quality filters from upstream artifacts.",
        ),
        "no_proxy": (
            no_proxy,
            "Subset excluding proxy-like or stub-like contracts for ablation against clean_default.",
        ),
        "cgt_only": (
            cgt_only,
            "Subset restricted to contracts with CGT provenance only (no DAppSCAN provenance).",
        ),
        "combined_posaug": (
            combined_posaug,
            "Subset including CGT and DAppSCAN-positive augmentation candidates for combined training.",
        ),
    }

    results: Dict[str, Dict[str, Any]] = {}
    for variant_name, (frame, description) in variant_frames.items():
        split_counts = {split: int((frame["split"] == split).sum()) for split in REQUIRED_SPLITS}
        source_breakdown = frame["source_group"].value_counts().to_dict()
        source_breakdown = {str(key): int(value) for key, value in source_breakdown.items()}

        payload = {
            "variant_name": variant_name,
            "description": description,
            "graph_index_parquet": _rel(graph_index_path),
            "graph_count": int(len(frame)),
            "split_counts": split_counts,
            "source_breakdown": source_breakdown,
            "graph_ids": frame["graph_id"].tolist(),
            "artifact_paths": frame["artifact_path"].tolist(),
        }
        path = manifests_dir / f"{variant_name}.json"
        _write_json(payload, path)
        results[variant_name] = {
            "path": _rel(path),
            "graph_count": int(len(frame)),
            "split_counts": split_counts,
        }
    return results


def _build_preprocessing_summary(
    report: Mapping[str, Any],
    variant_manifests: Mapping[str, Mapping[str, Any]],
) -> str:
    split_rows = [
        {"split": split, "graphs": report["split_counts"].get(split, 0)}
        for split in REQUIRED_SPLITS
    ]
    variant_rows = [
        {"variant": name, "graphs": info["graph_count"], "manifest": info["path"]}
        for name, info in variant_manifests.items()
    ]
    edge_type_rows = [
        {"edge_type": key, "count": value}
        for key, value in sorted(report["edge_type_counts"].items())
    ]

    return (
        "# Phase 2 Preprocessing Summary (Task 5)\n\n"
        f"Generated at (UTC): `{report['generated_at_utc']}`\n\n"
        "## Scope\n\n"
        "- Assembled final graph artifacts for the Phase 2 main benchmark.\n"
        "- Graph artifact format: `.pt` (PyTorch tensors + metadata, directly usable for PyTorch Geometric input construction).\n"
        f"- Main benchmark SWCs: `{report['main_benchmark_swcs']}`\n\n"
        "## Graph coverage\n\n"
        f"- contracts processed: **{report['contracts_processed']}**\n"
        f"- graph-build success: **{report['graph_build_success_count']}**\n"
        f"- empty graphs: **{report['empty_graph_count']}**\n"
        f"- CFG coverage (contracts): **{report['cfg_success_count']}**\n"
        f"- DFG coverage (contracts): **{report['dfg_success_count']}**\n\n"
        "## Split counts\n\n"
        + _markdown_table(split_rows, [("split", "split"), ("graphs", "graphs")])
        + "\n\n## Edge-type counts\n\n"
        + _markdown_table(edge_type_rows, [("edge_type", "edge_type"), ("count", "count")])
        + "\n\n## Variant manifests\n\n"
        + _markdown_table(variant_rows, [("variant", "variant"), ("graphs", "graphs"), ("manifest", "manifest")])
        + "\n\n## Core outputs\n\n"
        f"- graph index parquet: `{report['outputs']['graph_index_parquet']}`\n"
        f"- graph artifacts root: `{report['outputs']['graphs_dir']}`\n"
        f"- graph quality report: `{report['outputs']['graph_quality_report_json']}`\n"
        f"- preprocessing summary: `{report['outputs']['preprocessing_summary_md']}`\n"
    )


def run_graph_building(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    run_started = datetime.now(timezone.utc)
    config = _load_config(config_path.resolve())

    input_paths = [
        config.input_contracts_path,
        config.input_labels_path,
        config.input_instructions_path,
        config.input_basic_blocks_path,
        config.input_cfg_nodes_path,
        config.input_cfg_edges_path,
        config.input_dfg_edges_path,
    ]
    missing_inputs = [str(path) for path in input_paths if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError(f"Missing required input artifact(s): {missing_inputs}")

    contracts = pd.read_parquet(config.input_contracts_path).copy()
    labels = pd.read_parquet(config.input_labels_path).copy()
    instructions = pd.read_parquet(config.input_instructions_path).copy()
    basic_blocks = pd.read_parquet(config.input_basic_blocks_path).copy()
    cfg_nodes = pd.read_parquet(config.input_cfg_nodes_path).copy()
    cfg_edges = pd.read_parquet(config.input_cfg_edges_path).copy()
    dfg_edges = pd.read_parquet(config.input_dfg_edges_path).copy()

    required_contract_cols = {
        CONTRACT_ID_COLUMN,
        "split",
        "sources",
        "source_count",
        "has_cgt",
        "has_dappscan",
        "is_proxy_like",
        "is_stub_like",
    }
    missing_contract_cols = sorted(required_contract_cols - set(contracts.columns))
    if missing_contract_cols:
        raise ValueError(f"Input contracts missing required columns: {missing_contract_cols}")

    for frame in [contracts, instructions, basic_blocks, cfg_nodes, cfg_edges, dfg_edges, labels]:
        frame[CONTRACT_ID_COLUMN] = frame[CONTRACT_ID_COLUMN].fillna("").astype(str).str.strip()
    contracts = contracts[contracts[CONTRACT_ID_COLUMN] != ""].drop_duplicates(
        subset=[CONTRACT_ID_COLUMN], keep="first"
    )
    if contracts.empty:
        raise ValueError("No contracts found in input contracts parquet after cleaning contract IDs.")

    label_matrix, label_conflict_pairs = _load_label_matrix(labels, config.main_swc_ids)

    instruction_groups = instructions.groupby(CONTRACT_ID_COLUMN, sort=False).groups
    block_groups = basic_blocks.groupby(CONTRACT_ID_COLUMN, sort=False).groups
    cfg_node_groups = cfg_nodes.groupby(CONTRACT_ID_COLUMN, sort=False).groups
    cfg_edge_groups = cfg_edges.groupby(CONTRACT_ID_COLUMN, sort=False).groups
    dfg_edge_groups = dfg_edges.groupby(CONTRACT_ID_COLUMN, sort=False).groups

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    graphs_run_dir = config.graphs_dir / f"phase2_task5_{timestamp}"
    while graphs_run_dir.exists():
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_%f")
        graphs_run_dir = config.graphs_dir / f"phase2_task5_{timestamp}"
    graphs_run_dir.mkdir(parents=True, exist_ok=False)

    index_rows: List[Dict[str, Any]] = []
    total_edge_type_counts: Dict[str, int] = {}

    contracts_sorted = contracts.sort_values(["split", CONTRACT_ID_COLUMN]).reset_index(drop=True)
    for _, contract in contracts_sorted.iterrows():
        contract_id = str(contract[CONTRACT_ID_COLUMN])
        contract_instructions = (
            instructions.loc[instruction_groups[contract_id]].copy()
            if contract_id in instruction_groups
            else instructions.iloc[0:0].copy()
        )
        contract_blocks = (
            basic_blocks.loc[block_groups[contract_id]].copy()
            if contract_id in block_groups
            else basic_blocks.iloc[0:0].copy()
        )
        contract_cfg_nodes = (
            cfg_nodes.loc[cfg_node_groups[contract_id]].copy()
            if contract_id in cfg_node_groups
            else cfg_nodes.iloc[0:0].copy()
        )
        contract_cfg_edges = (
            cfg_edges.loc[cfg_edge_groups[contract_id]].copy()
            if contract_id in cfg_edge_groups
            else cfg_edges.iloc[0:0].copy()
        )
        contract_dfg_edges = (
            dfg_edges.loc[dfg_edge_groups[contract_id]].copy()
            if contract_id in dfg_edge_groups
            else dfg_edges.iloc[0:0].copy()
        )

        node_table, instruction_parse_success, instruction_failure_mode = _build_node_table(contract_instructions)
        node_id_set = {int(node_id) for node_id in node_table["node_id"]}
        block_end_map = _build_block_end_map(contract_blocks)
        cfg_edge_records = _build_cfg_instruction_edges(contract_cfg_edges, node_id_set=node_id_set, block_end_map=block_end_map)
        dfg_edge_records = _build_dfg_edges(contract_dfg_edges, node_id_set=node_id_set)

        edge_records = cfg_edge_records + dfg_edge_records
        edge_table = _edge_table_to_columns(edge_records)
        label_vector, label_mask = _label_vector_for_contract(label_matrix, contract_id, config.main_swc_ids)

        cfg_success = bool(contract_cfg_nodes["cfg_success"].fillna(False).astype(bool).any()) if not contract_cfg_nodes.empty else False
        dfg_success = bool(contract_dfg_edges["dfg_success"].fillna(False).astype(bool).any()) if not contract_dfg_edges.empty else False
        graph_build_success = bool(instruction_parse_success and len(node_table["node_id"]) > 0)

        cfg_edge_count = int(len(cfg_edge_records))
        dfg_edge_count = int(len(dfg_edge_records))
        total_edge_count = int(len(edge_records))
        for edge in edge_records:
            token = f"{edge['edge_family']}_{edge['edge_type']}"
            total_edge_type_counts[token] = total_edge_type_counts.get(token, 0) + 1

        positive_labels = int(sum(1 for value in label_vector if value == 1))
        assessed_labels = int(sum(1 for value in label_mask if value))

        source_list = _safe_json_list(contract.get("sources"))
        has_cgt = _as_nullable_bool(contract.get("has_cgt"))
        has_dappscan = _as_nullable_bool(contract.get("has_dappscan"))
        is_proxy_like = _as_nullable_bool(contract.get("is_proxy_like"))
        is_stub_like = _as_nullable_bool(contract.get("is_stub_like"))
        split = str(contract.get("split", "")).strip() or "unknown"
        source_group = _source_group(has_cgt, has_dappscan)

        pyg_payload = _build_pyg_payload(node_table, edge_records, label_vector, label_mask)
        graph_stats = {
            "node_count": int(len(node_table["node_id"])),
            "basic_block_count": int(len(block_end_map)),
            "cfg_edge_count": cfg_edge_count,
            "dfg_edge_count": dfg_edge_count,
            "edge_count_total": total_edge_count,
            "instruction_parse_success": instruction_parse_success,
            "cfg_success": cfg_success,
            "dfg_success": dfg_success,
            "label_positive_count": positive_labels,
            "label_assessed_count": assessed_labels,
        }

        package = {
            "contract_id": contract_id,
            "graph_id": contract_id,
            "label": {
                "swc_ids": list(config.main_swc_ids),
                "vector": label_vector,
                "mask": label_mask,
                "positive_count": positive_labels,
                "assessed_count": assessed_labels,
            },
            "split": split,
            "provenance": {
                "sources": source_list,
                "source_count": _as_int(contract.get("source_count"), default=len(source_list)),
                "has_cgt": has_cgt,
                "has_dappscan": has_dappscan,
            },
            "flags": {
                "is_proxy_like": is_proxy_like,
                "is_stub_like": is_stub_like,
            },
            "opcode_sequence": {
                "pc": node_table["pc"],
                "opcode_id": node_table["opcode_id"],
                "opcode": node_table["opcode"],
            },
            "node_table": node_table,
            "edge_table": edge_table,
            "graph_stats": graph_stats,
            "build_metadata": {
                "format": "phase2_graph_pt_v1",
                "graph_artifact_format": config.graph_artifact_format,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "instruction_failure_mode": instruction_failure_mode,
            },
            "pyg": pyg_payload,
            "pyg_edge_type_vocab": EDGE_TYPE_VOCAB,
        }

        artifact_path = graphs_run_dir / f"{contract_id}.{config.graph_artifact_format}"
        torch.save(package, artifact_path)
        artifact_relpath = _rel(artifact_path)

        row = {
            "graph_id": contract_id,
            CONTRACT_ID_COLUMN: contract_id,
            "artifact_path": artifact_relpath,
            "artifact_format": config.graph_artifact_format,
            "split": split,
            "source_group": source_group,
            "has_cgt": has_cgt,
            "has_dappscan": has_dappscan,
            "is_proxy_like": is_proxy_like,
            "is_stub_like": is_stub_like,
            "graph_build_success": graph_build_success,
            "instruction_parse_success": instruction_parse_success,
            "cfg_success": cfg_success,
            "dfg_success": dfg_success,
            "node_count": graph_stats["node_count"],
            "basic_block_count": graph_stats["basic_block_count"],
            "cfg_edge_count": cfg_edge_count,
            "dfg_edge_count": dfg_edge_count,
            "edge_count_total": total_edge_count,
            "label_positive_count": positive_labels,
            "label_assessed_count": assessed_labels,
            "label_vector": json.dumps(label_vector),
            "label_mask": json.dumps(label_mask),
        }
        for idx, swc_id in enumerate(config.main_swc_ids):
            row[f"swc_{swc_id}"] = int(label_vector[idx])
            row[f"swc_{swc_id}_assessed"] = bool(label_mask[idx])
        index_rows.append(row)

    graph_index_df = pd.DataFrame(index_rows).sort_values(["split", CONTRACT_ID_COLUMN]).reset_index(drop=True)
    mapping_validation = _validate_graph_index_mapping(graph_index_df, graphs_run_dir)

    output_notes: List[Dict[str, str]] = []
    resolved_graph_index_path = _resolve_output_path(config.graph_index_out_path, output_notes)
    resolved_quality_report_path = _resolve_output_path(config.quality_report_json_path, output_notes)
    resolved_summary_path = _resolve_output_path(config.preprocessing_summary_md_path, output_notes)
    resolved_run_manifest_path = _resolve_output_path(config.run_manifest_json_path, output_notes)

    manifest_run_dir = config.variant_manifests_dir / f"phase2_task5_{timestamp}"
    while manifest_run_dir.exists():
        manifest_run_dir = config.variant_manifests_dir / f"phase2_task5_{timestamp}_{len(output_notes) + 1}"
    manifest_run_dir.mkdir(parents=True, exist_ok=False)

    resolved_graph_index_path.parent.mkdir(parents=True, exist_ok=True)
    graph_index_df.to_parquet(resolved_graph_index_path, index=False)
    variant_manifests = _create_variant_manifests(graph_index_df, manifest_run_dir, resolved_graph_index_path)

    graph_build_success_count = int(graph_index_df["graph_build_success"].sum())
    cfg_success_count = int(graph_index_df["cfg_success"].sum())
    dfg_success_count = int(graph_index_df["dfg_success"].sum())
    empty_graph_count = int((graph_index_df["node_count"] == 0).sum())
    split_counts = graph_index_df["split"].value_counts().to_dict()
    split_counts = {str(key): int(value) for key, value in split_counts.items()}
    source_breakdown = graph_index_df["source_group"].value_counts().to_dict()
    source_breakdown = {str(key): int(value) for key, value in source_breakdown.items()}

    report: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "phase2_task5_graph_assembly",
        "main_benchmark_swcs": list(config.main_swc_ids),
        "graph_artifact_format": config.graph_artifact_format,
        "inputs": {
            "contracts_parquet": _rel(config.input_contracts_path),
            "labels_parquet": _rel(config.input_labels_path),
            "instructions_parquet": _rel(config.input_instructions_path),
            "basic_blocks_parquet": _rel(config.input_basic_blocks_path),
            "cfg_nodes_parquet": _rel(config.input_cfg_nodes_path),
            "cfg_edges_parquet": _rel(config.input_cfg_edges_path),
            "dfg_edges_parquet": _rel(config.input_dfg_edges_path),
        },
        "outputs": {
            "graph_index_parquet": _rel(resolved_graph_index_path),
            "graphs_dir": _rel(graphs_run_dir),
            "graph_quality_report_json": _rel(resolved_quality_report_path),
            "preprocessing_summary_md": _rel(resolved_summary_path),
            "variant_manifests_dir": _rel(manifest_run_dir),
            "run_manifest_json": _rel(resolved_run_manifest_path),
        },
        "configured_outputs": {
            "graph_index_parquet": _rel(config.graph_index_out_path),
            "graphs_dir": _rel(config.graphs_dir),
            "graph_quality_report_json": _rel(config.quality_report_json_path),
            "preprocessing_summary_md": _rel(config.preprocessing_summary_md_path),
            "variant_manifests_dir": _rel(config.variant_manifests_dir),
            "run_manifest_json": _rel(config.run_manifest_json_path),
        },
        "output_resolution_notes": output_notes,
        "contracts_processed": int(len(graph_index_df)),
        "graph_build_success_count": graph_build_success_count,
        "graph_build_success_rate": (graph_build_success_count / len(graph_index_df)) if len(graph_index_df) else 0.0,
        "cfg_success_count": cfg_success_count,
        "cfg_success_rate": (cfg_success_count / len(graph_index_df)) if len(graph_index_df) else 0.0,
        "dfg_success_count": dfg_success_count,
        "dfg_success_rate": (dfg_success_count / len(graph_index_df)) if len(graph_index_df) else 0.0,
        "empty_graph_count": empty_graph_count,
        "split_counts": split_counts,
        "source_breakdown": source_breakdown,
        "edge_type_counts": total_edge_type_counts,
        "node_count_stats": _quantiles(graph_index_df["node_count"].tolist()),
        "cfg_edge_count_stats": _quantiles(graph_index_df["cfg_edge_count"].tolist()),
        "dfg_edge_count_stats": _quantiles(graph_index_df["dfg_edge_count"].tolist()),
        "total_edge_count_stats": _quantiles(graph_index_df["edge_count_total"].tolist()),
        "label_conflict_pairs_detected": int(label_conflict_pairs),
        "mapping_validation": mapping_validation,
        "variant_manifests": variant_manifests,
        "known_limitations": [
            "Graphs use instruction-level nodes; CFG edges are mapped from basic-block edges to terminator->target instruction links.",
            "DFG coverage follows conservative assumptions from Phase 2 Task 4 and intentionally under-claims unresolved data dependencies.",
            "Unknown labels are encoded as -1 in label_vector with explicit label_mask for downstream filtering.",
        ],
    }
    _write_json(report, resolved_quality_report_path)

    summary_md = _build_preprocessing_summary(report, variant_manifests)
    _write_text(summary_md, resolved_summary_path)

    run_manifest = {
        "run_started_utc": run_started.isoformat(),
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "entrypoint": "python -m src.preprocessing.graph_builder --config configs/phase2.yaml",
        "config_path": _rel(config_path.resolve()),
        "inputs": report["inputs"],
        "outputs": report["outputs"],
        "summary": {
            "contracts_processed": report["contracts_processed"],
            "graph_build_success_count": report["graph_build_success_count"],
            "cfg_success_count": report["cfg_success_count"],
            "dfg_success_count": report["dfg_success_count"],
            "empty_graph_count": report["empty_graph_count"],
        },
    }
    _write_json(run_manifest, resolved_run_manifest_path)
    return report


def _print_report_summary(report: Mapping[str, Any]) -> None:
    print(f"contracts processed: {report['contracts_processed']}")
    print(f"graph build success: {report['graph_build_success_count']}")
    print(f"empty graphs: {report['empty_graph_count']}")
    print(f"cfg success count: {report['cfg_success_count']}")
    print(f"dfg success count: {report['dfg_success_count']}")
    print(f"graph index: {report['outputs']['graph_index_parquet']}")
    print(f"graphs dir: {report['outputs']['graphs_dir']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble final Phase 2 graph artifacts for the main benchmark.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    report = run_graph_building(args.config.resolve())
    _print_report_summary(report)


if __name__ == "__main__":
    main()
