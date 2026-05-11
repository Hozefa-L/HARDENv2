import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import pandas as pd
import pyevmasm
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs/phase2.yaml"
DEFAULT_INPUT_INSTRUCTIONS = PROJECT_ROOT / "data/curated/phase2_instructions.parquet"
DEFAULT_INPUT_BASIC_BLOCKS = PROJECT_ROOT / "data/curated/phase2_basic_blocks.parquet"
DEFAULT_INPUT_CFG_EDGES = PROJECT_ROOT / "data/curated/phase2_cfg_edges.parquet"
DEFAULT_DFG_EDGES_OUT = PROJECT_ROOT / "data/curated/phase2_dfg_edges.parquet"
DEFAULT_REPORT_JSON = PROJECT_ROOT / "reports/phase2/dfg_report.json"
DEFAULT_RUN_MANIFEST_JSON = PROJECT_ROOT / "reports/phase2/dfg_run_manifest.json"

CONTRACT_ID_COLUMN = "fp_runtime_unified"
VALID_DFG_EDGE_TYPES = {"stack_flow", "storage_flow", "memory_flow", "control_dependency"}
TERMINAL_OPCODES = {0x00, 0xF3, 0xFD, 0xFE, 0xFF}
STORAGE_READ_OPCODE = 0x54
STORAGE_WRITE_OPCODE = 0x55
MEMORY_READ_OPCODE = 0x51
MEMORY_WRITE_OPCODES = {0x52, 0x53}
CONDITIONAL_JUMP_OPCODE = 0x57
UNCERTAIN_MEMORY_WRITE_OPCODES = {0x37, 0x39, 0x3C, 0x3E, 0x5E}
EXTERNAL_EFFECT_OPCODES = {0xF0, 0xF1, 0xF2, 0xF4, 0xF5, 0xFA, 0xFF}
STACK_EFFECT_OVERRIDES = {
    0x49: (1, 1),  # BLOBHASH
    0x4A: (0, 1),  # BLOBBASEFEE
    0x5F: (0, 1),  # PUSH0
    0x5C: (1, 1),  # TLOAD
    0x5D: (2, 0),  # TSTORE
    0x5E: (3, 0),  # MCOPY
}
EVM_TABLE = pyevmasm.instruction_tables["istanbul"]


@dataclass(frozen=True)
class DfgConfig:
    input_instructions_path: Path
    input_basic_blocks_path: Path
    input_cfg_edges_path: Path
    dfg_edges_out_path: Path
    report_json_path: Path
    run_manifest_json_path: Path


@dataclass(frozen=True)
class StackValue:
    producers: Tuple[int, ...]
    literal: Optional[int] = None


UNKNOWN_STACK_VALUE = StackValue(producers=(), literal=None)


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


def _safe_read_mapping(config_path: Path) -> Mapping[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp)
    if not isinstance(data, dict):
        raise ValueError("Phase 2 config must be a mapping.")
    return data


def _load_config(config_path: Path) -> DfgConfig:
    raw = _safe_read_mapping(config_path)
    dfg_section = raw.get("dfg", {})
    if dfg_section is None:
        dfg_section = {}
    if not isinstance(dfg_section, dict):
        raise ValueError("`dfg` must be a mapping when provided.")

    input_instructions = dfg_section.get("input_instructions") or str(DEFAULT_INPUT_INSTRUCTIONS)
    input_basic_blocks = dfg_section.get("input_basic_blocks") or str(DEFAULT_INPUT_BASIC_BLOCKS)
    input_cfg_edges = dfg_section.get("input_cfg_edges") or str(DEFAULT_INPUT_CFG_EDGES)
    dfg_edges_out = dfg_section.get("dfg_edges_parquet") or str(DEFAULT_DFG_EDGES_OUT)
    report_json = dfg_section.get("report_json") or str(DEFAULT_REPORT_JSON)
    run_manifest_json = dfg_section.get("run_manifest_json") or str(DEFAULT_RUN_MANIFEST_JSON)

    return DfgConfig(
        input_instructions_path=_resolve_path(input_instructions),
        input_basic_blocks_path=_resolve_path(input_basic_blocks),
        input_cfg_edges_path=_resolve_path(input_cfg_edges),
        dfg_edges_out_path=_resolve_path(dfg_edges_out),
        report_json_path=_resolve_path(report_json),
        run_manifest_json_path=_resolve_path(run_manifest_json),
    )


def _as_nullable_bool(value: Any) -> Optional[bool]:
    if pd.isna(value):
        return None
    return bool(value)


def _quantiles(values: Iterable[int]) -> Dict[str, Optional[float]]:
    series = pd.Series(list(values), dtype="float64")
    if series.empty:
        return {"median": None, "p95": None}
    return {
        "median": float(series.quantile(0.5)),
        "p95": float(series.quantile(0.95)),
    }


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


def _non_empty_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _first_non_empty_text(values: Sequence[Any]) -> str:
    for value in values:
        text = _non_empty_text(value)
        if text:
            return text
    return ""


def _stack_effect(opcode_id: int) -> Tuple[int, int]:
    if opcode_id in STACK_EFFECT_OVERRIDES:
        return STACK_EFFECT_OVERRIDES[opcode_id]
    try:
        instruction = EVM_TABLE[opcode_id]
        return int(instruction.pops), int(instruction.pushes)
    except Exception:
        if 0x60 <= opcode_id <= 0x7F:
            return 0, 1
        if 0x80 <= opcode_id <= 0x8F:
            depth = opcode_id - 0x7F
            return depth, depth + 1
        if 0x90 <= opcode_id <= 0x9F:
            depth = opcode_id - 0x8F
            return depth + 1, depth + 1
        return 0, 0


def _normalize_producers(producers: Iterable[int]) -> Tuple[int, ...]:
    return tuple(sorted({int(pc) for pc in producers}))


def _stack_value(producers: Iterable[int], literal: Optional[int] = None) -> StackValue:
    return StackValue(producers=_normalize_producers(producers), literal=literal)


def _push_literal(opcode_id: int, push_data: Any) -> Optional[int]:
    if opcode_id == 0x5F:
        return 0
    text = _non_empty_text(push_data)
    if not text:
        return None
    try:
        return int(text, 16)
    except ValueError:
        return None


def _pop_stack_value(stack: List[StackValue]) -> StackValue:
    if stack:
        return stack.pop()
    return UNKNOWN_STACK_VALUE


def _append_edge(
    edge_keys: Set[Tuple[int, int, str, Optional[int], Optional[int]]],
    edges: List[Dict[str, Any]],
    src_pc: Optional[int],
    dst_pc: Optional[int],
    edge_type: str,
    storage_slot: Optional[int] = None,
    memory_offset: Optional[int] = None,
) -> None:
    if src_pc is None or dst_pc is None:
        return
    src = int(src_pc)
    dst = int(dst_pc)
    if src < 0 or dst < 0:
        return
    key = (src, dst, edge_type, storage_slot, memory_offset)
    if key in edge_keys:
        return
    edge_keys.add(key)
    edges.append(
        {
            "src_instruction_pc": src,
            "dst_instruction_pc": dst,
            "edge_type": edge_type,
            "storage_slot": storage_slot,
            "memory_offset": memory_offset,
        }
    )


def _build_control_targets(contract_cfg_edges: pd.DataFrame) -> Dict[int, Dict[str, Set[int]]]:
    control_targets: Dict[int, Dict[str, Set[int]]] = {}
    if contract_cfg_edges.empty:
        return control_targets

    required = {"cfg_success", "src_node_id", "dst_node_id", "edge_type"}
    if not required.issubset(set(contract_cfg_edges.columns)):
        return control_targets

    valid = contract_cfg_edges[
        contract_cfg_edges["cfg_success"].fillna(False).astype(bool)
        & contract_cfg_edges["src_node_id"].notna()
        & contract_cfg_edges["dst_node_id"].notna()
    ].copy()
    if valid.empty:
        return control_targets

    valid["src_node_id"] = pd.to_numeric(valid["src_node_id"], errors="coerce").astype("Int64")
    valid["dst_node_id"] = pd.to_numeric(valid["dst_node_id"], errors="coerce").astype("Int64")
    valid = valid[valid["src_node_id"].notna() & valid["dst_node_id"].notna()]
    for _, row in valid.iterrows():
        edge_type = _non_empty_text(row.get("edge_type"))
        if edge_type not in {"jumpi_true", "jumpi_false"}:
            continue
        src_pc = int(row["src_node_id"])
        dst_pc = int(row["dst_node_id"])
        src_entry = control_targets.setdefault(src_pc, {"jumpi_true": set(), "jumpi_false": set()})
        src_entry[edge_type].add(dst_pc)
    return control_targets


def _simulate_basic_block(
    block_instructions: pd.DataFrame,
    block_start_pc: int,
    control_targets: Dict[int, Dict[str, Set[int]]],
) -> List[Dict[str, Any]]:
    edges: List[Dict[str, Any]] = []
    edge_keys: Set[Tuple[int, int, str, Optional[int], Optional[int]]] = set()
    stack: List[StackValue] = []
    storage_state: Dict[int, int] = {}
    memory_state: Dict[int, int] = {}

    rows = block_instructions.sort_values("pc").itertuples(index=False)
    for row in rows:
        if pd.isna(row.pc) or pd.isna(row.opcode_id):
            continue
        pc = int(row.pc)
        opcode_id = int(row.opcode_id)
        opcode = _non_empty_text(getattr(row, "opcode", ""))

        if opcode.startswith("UNKNOWN_"):
            break

        if opcode_id == 0x5F or 0x60 <= opcode_id <= 0x7F:
            literal = _push_literal(opcode_id, getattr(row, "push_data", ""))
            stack.append(_stack_value([pc], literal=literal))
            continue

        if 0x80 <= opcode_id <= 0x8F:
            depth = opcode_id - 0x7F
            source = stack[-depth] if len(stack) >= depth else UNKNOWN_STACK_VALUE
            for producer_pc in source.producers:
                _append_edge(edge_keys, edges, producer_pc, pc, "stack_flow")
            stack.append(_stack_value(source.producers, source.literal))
            continue

        if 0x90 <= opcode_id <= 0x9F:
            depth = opcode_id - 0x8F
            if len(stack) >= depth + 1:
                top_idx = -1
                other_idx = -(depth + 1)
                stack[top_idx], stack[other_idx] = stack[other_idx], stack[top_idx]
            continue

        pop_count, push_count = _stack_effect(opcode_id)
        popped_values = [_pop_stack_value(stack) for _ in range(max(pop_count, 0))]
        for value in popped_values:
            for producer_pc in value.producers:
                _append_edge(edge_keys, edges, producer_pc, pc, "stack_flow")

        handled_push = False
        if opcode_id == STORAGE_WRITE_OPCODE:
            slot = popped_values[0].literal if len(popped_values) >= 1 else None
            if slot is not None:
                storage_state[int(slot)] = pc
            else:
                storage_state.clear()
        elif opcode_id == STORAGE_READ_OPCODE:
            slot = popped_values[0].literal if len(popped_values) >= 1 else None
            if slot is not None and int(slot) in storage_state:
                _append_edge(edge_keys, edges, storage_state[int(slot)], pc, "storage_flow", storage_slot=int(slot))
            stack.append(_stack_value([pc]))
            handled_push = True
        elif opcode_id in MEMORY_WRITE_OPCODES:
            offset = popped_values[0].literal if len(popped_values) >= 1 else None
            if offset is not None:
                memory_state[int(offset)] = pc
            else:
                memory_state.clear()
        elif opcode_id == MEMORY_READ_OPCODE:
            offset = popped_values[0].literal if len(popped_values) >= 1 else None
            if offset is not None and int(offset) in memory_state:
                _append_edge(
                    edge_keys,
                    edges,
                    memory_state[int(offset)],
                    pc,
                    "memory_flow",
                    memory_offset=int(offset),
                )
            stack.append(_stack_value([pc]))
            handled_push = True
        elif opcode_id == CONDITIONAL_JUMP_OPCODE:
            branch_targets = control_targets.get(block_start_pc, {})
            true_targets = sorted(branch_targets.get("jumpi_true", set()))
            false_targets = sorted(branch_targets.get("jumpi_false", set()))
            condition_value = popped_values[1] if len(popped_values) >= 2 else UNKNOWN_STACK_VALUE
            if condition_value.producers and true_targets and false_targets:
                for producer_pc in condition_value.producers:
                    for target_pc in true_targets + false_targets:
                        _append_edge(edge_keys, edges, producer_pc, int(target_pc), "control_dependency")

        if opcode_id in UNCERTAIN_MEMORY_WRITE_OPCODES:
            memory_state.clear()
        if opcode_id in EXTERNAL_EFFECT_OPCODES:
            storage_state.clear()
            memory_state.clear()

        if not handled_push:
            for _ in range(max(push_count, 0)):
                stack.append(_stack_value([pc]))

        if opcode_id in TERMINAL_OPCODES:
            break

    return edges


def build_contract_dfg(
    contract_instructions: pd.DataFrame,
    contract_basic_blocks: pd.DataFrame,
    contract_cfg_edges: pd.DataFrame,
) -> Dict[str, Any]:
    if contract_instructions.empty:
        return {
            "dfg_success": False,
            "failure_mode": "missing_instruction_rows",
            "failure_detail": "",
            "edges": [],
        }

    parse_success = contract_instructions["parse_success"].fillna(False).astype(bool)
    success_rows = contract_instructions[
        parse_success & contract_instructions["pc"].notna() & contract_instructions["opcode_id"].notna()
    ].copy()
    if success_rows.empty:
        failure_mode = _first_non_empty_text(contract_instructions.get("failure_mode", pd.Series(dtype=object)).tolist())
        return {
            "dfg_success": False,
            "failure_mode": failure_mode or "instruction_parse_failed",
            "failure_detail": "",
            "edges": [],
        }

    success_rows["pc"] = pd.to_numeric(success_rows["pc"], errors="coerce").astype("Int64")
    success_rows["opcode_id"] = pd.to_numeric(success_rows["opcode_id"], errors="coerce").astype("Int64")
    success_rows = success_rows[success_rows["pc"].notna() & success_rows["opcode_id"].notna()].copy()
    if success_rows.empty:
        return {
            "dfg_success": False,
            "failure_mode": "empty_instruction_stream",
            "failure_detail": "",
            "edges": [],
        }
    success_rows["pc"] = success_rows["pc"].astype(int)
    success_rows["opcode_id"] = success_rows["opcode_id"].astype(int)

    block_map: Dict[int, pd.DataFrame] = {}
    if "basic_block_id" in success_rows.columns:
        block_rows = success_rows[success_rows["basic_block_id"].notna()].copy()
        if not block_rows.empty:
            block_rows["basic_block_id"] = pd.to_numeric(block_rows["basic_block_id"], errors="coerce").astype("Int64")
            block_rows = block_rows[block_rows["basic_block_id"].notna()].copy()
            block_rows["basic_block_id"] = block_rows["basic_block_id"].astype(int)
            for block_id, frame in block_rows.groupby("basic_block_id", sort=False):
                block_map[int(block_id)] = frame.copy()

    if not block_map:
        fallback = success_rows.sort_values("pc").copy()
        block_map[0] = fallback

    block_order: List[Tuple[int, int]] = []
    if not contract_basic_blocks.empty and {"parse_success", "basic_block_id", "start_pc"}.issubset(
        set(contract_basic_blocks.columns)
    ):
        basic_blocks = contract_basic_blocks[
            contract_basic_blocks["parse_success"].fillna(False).astype(bool)
            & contract_basic_blocks["basic_block_id"].notna()
            & contract_basic_blocks["start_pc"].notna()
        ].copy()
        if not basic_blocks.empty:
            basic_blocks["basic_block_id"] = pd.to_numeric(basic_blocks["basic_block_id"], errors="coerce").astype("Int64")
            basic_blocks["start_pc"] = pd.to_numeric(basic_blocks["start_pc"], errors="coerce").astype("Int64")
            basic_blocks = basic_blocks[basic_blocks["basic_block_id"].notna() & basic_blocks["start_pc"].notna()]
            for _, row in basic_blocks.iterrows():
                block_id = int(row["basic_block_id"])
                if block_id in block_map:
                    block_order.append((block_id, int(row["start_pc"])))

    if not block_order:
        for block_id, frame in block_map.items():
            block_order.append((int(block_id), int(frame["pc"].min())))
    block_order = sorted(set(block_order), key=lambda x: (x[1], x[0]))

    control_targets = _build_control_targets(contract_cfg_edges)
    all_edges: List[Dict[str, Any]] = []
    all_keys: Set[Tuple[int, int, str, Optional[int], Optional[int]]] = set()
    for block_id, block_start_pc in block_order:
        frame = block_map.get(block_id)
        if frame is None or frame.empty:
            continue
        block_edges = _simulate_basic_block(frame, block_start_pc=block_start_pc, control_targets=control_targets)
        for edge in block_edges:
            key = (
                int(edge["src_instruction_pc"]),
                int(edge["dst_instruction_pc"]),
                str(edge["edge_type"]),
                edge.get("storage_slot"),
                edge.get("memory_offset"),
            )
            if key in all_keys:
                continue
            all_keys.add(key)
            all_edges.append(edge)

    return {
        "dfg_success": True,
        "failure_mode": "",
        "failure_detail": "",
        "edges": all_edges,
    }


def _finalize_edges_df(rows: List[Tuple[Any, ...]]) -> pd.DataFrame:
    columns = [
        "graph_id",
        CONTRACT_ID_COLUMN,
        "split",
        "has_cgt",
        "has_dappscan",
        "is_proxy_like",
        "is_stub_like",
        "instruction_parse_success",
        "cfg_success",
        "dfg_success",
        "backend_used",
        "failure_mode",
        "src_instruction_pc",
        "dst_instruction_pc",
        "edge_type",
        "storage_slot",
        "memory_offset",
    ]
    df = pd.DataFrame.from_records(rows, columns=columns)
    if df.empty:
        return df

    for col in ["src_instruction_pc", "dst_instruction_pc", "storage_slot", "memory_offset"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ["has_cgt", "has_dappscan", "is_proxy_like", "is_stub_like", "instruction_parse_success", "cfg_success"]:
        df[col] = df[col].astype("boolean")
    df["dfg_success"] = df["dfg_success"].astype(bool)
    df["backend_used"] = df["backend_used"].fillna("").astype(str)
    df["failure_mode"] = df["failure_mode"].fillna("").astype(str)
    df["edge_type"] = df["edge_type"].fillna("").astype(str)
    return df


def validate_dfg_edge_schema(dfg_edges_df: pd.DataFrame) -> None:
    required_columns = {
        "graph_id",
        CONTRACT_ID_COLUMN,
        "dfg_success",
        "backend_used",
        "failure_mode",
        "src_instruction_pc",
        "dst_instruction_pc",
        "edge_type",
    }
    missing = sorted(required_columns - set(dfg_edges_df.columns))
    if missing:
        raise ValueError(f"DFG edges parquet missing required columns: {missing}")

    if dfg_edges_df.empty:
        return

    successful_edges = dfg_edges_df[dfg_edges_df["dfg_success"]].copy()
    if successful_edges.empty:
        return
    if bool(successful_edges["src_instruction_pc"].isna().any()) or bool(
        successful_edges["dst_instruction_pc"].isna().any()
    ):
        raise ValueError("Successful DFG edges must include non-null src_instruction_pc and dst_instruction_pc.")

    edge_types = successful_edges["edge_type"].fillna("").astype(str)
    invalid_edge_types = sorted(set(edge_types.tolist()) - VALID_DFG_EDGE_TYPES)
    if invalid_edge_types:
        raise ValueError(f"Unsupported DFG edge type(s): {invalid_edge_types}")


def run_dfg_building(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    run_started = datetime.now(timezone.utc)
    config = _load_config(config_path.resolve())

    required_paths = [
        config.input_instructions_path,
        config.input_basic_blocks_path,
        config.input_cfg_edges_path,
    ]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing required input artifact(s): {missing_paths}")

    instructions = pd.read_parquet(config.input_instructions_path).copy()
    basic_blocks = pd.read_parquet(config.input_basic_blocks_path).copy()
    cfg_edges = pd.read_parquet(config.input_cfg_edges_path).copy()

    required_instruction_cols = {
        CONTRACT_ID_COLUMN,
        "split",
        "has_cgt",
        "has_dappscan",
        "is_proxy_like",
        "is_stub_like",
        "parse_success",
        "failure_mode",
        "pc",
        "opcode",
        "opcode_id",
        "push_data",
        "basic_block_id",
    }
    required_block_cols = {CONTRACT_ID_COLUMN, "parse_success", "failure_mode", "basic_block_id", "start_pc"}
    required_cfg_cols = {CONTRACT_ID_COLUMN, "cfg_success", "failure_mode", "src_node_id", "dst_node_id", "edge_type"}
    missing_instruction_cols = sorted(required_instruction_cols - set(instructions.columns))
    missing_block_cols = sorted(required_block_cols - set(basic_blocks.columns))
    missing_cfg_cols = sorted(required_cfg_cols - set(cfg_edges.columns))
    if missing_instruction_cols:
        raise ValueError(f"Input instructions missing required columns: {missing_instruction_cols}")
    if missing_block_cols:
        raise ValueError(f"Input basic blocks missing required columns: {missing_block_cols}")
    if missing_cfg_cols:
        raise ValueError(f"Input cfg edges missing required columns: {missing_cfg_cols}")

    for frame in [instructions, basic_blocks, cfg_edges]:
        frame[CONTRACT_ID_COLUMN] = frame[CONTRACT_ID_COLUMN].fillna("").astype(str).str.strip()
    instructions = instructions[instructions[CONTRACT_ID_COLUMN] != ""].copy()
    basic_blocks = basic_blocks[basic_blocks[CONTRACT_ID_COLUMN] != ""].copy()
    cfg_edges = cfg_edges[cfg_edges[CONTRACT_ID_COLUMN] != ""].copy()
    if instructions.empty:
        raise ValueError("No contract rows found in instructions parquet after cleaning contract IDs.")

    instruction_group_index = instructions.groupby(CONTRACT_ID_COLUMN, sort=False).groups
    block_group_index = basic_blocks.groupby(CONTRACT_ID_COLUMN, sort=False).groups
    cfg_group_index = cfg_edges.groupby(CONTRACT_ID_COLUMN, sort=False).groups
    contract_ids = sorted(set(instruction_group_index) | set(block_group_index) | set(cfg_group_index))

    dfg_rows: List[Tuple[Any, ...]] = []
    contract_stats: List[Dict[str, Any]] = []

    for contract_id in contract_ids:
        instruction_idx = instruction_group_index.get(contract_id)
        block_idx = block_group_index.get(contract_id)
        cfg_idx = cfg_group_index.get(contract_id)
        contract_instructions = instructions.loc[instruction_idx].copy() if instruction_idx is not None else instructions.iloc[0:0]
        contract_blocks = basic_blocks.loc[block_idx].copy() if block_idx is not None else basic_blocks.iloc[0:0]
        contract_cfg_edges = cfg_edges.loc[cfg_idx].copy() if cfg_idx is not None else cfg_edges.iloc[0:0]

        if not contract_instructions.empty:
            meta_row = contract_instructions.iloc[0]
        elif not contract_blocks.empty:
            meta_row = contract_blocks.iloc[0]
        elif not contract_cfg_edges.empty:
            meta_row = contract_cfg_edges.iloc[0]
        else:
            continue

        split = _non_empty_text(meta_row.get("split")) or "unknown"
        has_cgt = _as_nullable_bool(meta_row.get("has_cgt"))
        has_dappscan = _as_nullable_bool(meta_row.get("has_dappscan"))
        is_proxy_like = _as_nullable_bool(meta_row.get("is_proxy_like"))
        is_stub_like = _as_nullable_bool(meta_row.get("is_stub_like"))

        instruction_parse_success = bool(contract_instructions["parse_success"].fillna(False).astype(bool).any())
        instruction_failure_mode = _first_non_empty_text(contract_instructions["failure_mode"].tolist())
        cfg_success = bool(contract_cfg_edges["cfg_success"].fillna(False).astype(bool).any()) if not contract_cfg_edges.empty else False
        cfg_failure_mode = _first_non_empty_text(contract_cfg_edges.get("failure_mode", pd.Series(dtype=object)).tolist())

        dfg_result = build_contract_dfg(contract_instructions, contract_blocks, contract_cfg_edges)
        dfg_success = bool(dfg_result["dfg_success"])
        edges = dfg_result["edges"] if dfg_success else []
        edge_count = int(len(edges))

        contract_stats.append(
            {
                CONTRACT_ID_COLUMN: contract_id,
                "graph_id": contract_id,
                "split": split,
                "has_cgt": has_cgt,
                "has_dappscan": has_dappscan,
                "is_proxy_like": is_proxy_like,
                "is_stub_like": is_stub_like,
                "instruction_parse_success": instruction_parse_success,
                "instruction_failure_mode": instruction_failure_mode,
                "cfg_success": cfg_success,
                "cfg_failure_mode": cfg_failure_mode,
                "dfg_success": dfg_success,
                "failure_mode": dfg_result["failure_mode"],
                "failure_detail": dfg_result["failure_detail"],
                "dfg_edge_count": edge_count,
            }
        )

        if dfg_success:
            for edge in edges:
                dfg_rows.append(
                    (
                        contract_id,
                        contract_id,
                        split,
                        has_cgt,
                        has_dappscan,
                        is_proxy_like,
                        is_stub_like,
                        instruction_parse_success,
                        cfg_success,
                        True,
                        "abstract_stack_conservative_v1",
                        "",
                        edge.get("src_instruction_pc"),
                        edge.get("dst_instruction_pc"),
                        edge.get("edge_type"),
                        edge.get("storage_slot"),
                        edge.get("memory_offset"),
                    )
                )
            continue

        dfg_rows.append(
            (
                contract_id,
                contract_id,
                split,
                has_cgt,
                has_dappscan,
                is_proxy_like,
                is_stub_like,
                instruction_parse_success,
                cfg_success,
                False,
                "abstract_stack_conservative_v1",
                dfg_result["failure_mode"],
                None,
                None,
                "",
                None,
                None,
            )
        )

    dfg_edges_df = _finalize_edges_df(dfg_rows)
    validate_dfg_edge_schema(dfg_edges_df)

    if not dfg_edges_df.empty and not bool((dfg_edges_df["graph_id"] == dfg_edges_df[CONTRACT_ID_COLUMN]).all()):
        raise ValueError("Graph ID alignment check failed: graph_id must match fp_runtime_unified.")

    stats_df = pd.DataFrame(contract_stats)
    successful_contracts = stats_df[stats_df["dfg_success"]].copy()
    failed_contracts = stats_df[~stats_df["dfg_success"]].copy()
    empty_dfg_contracts = successful_contracts[successful_contracts["dfg_edge_count"] == 0].copy()

    contracts_total = int(len(stats_df))
    dfg_success_count = int(successful_contracts.shape[0])
    dfg_success_rate = (dfg_success_count / contracts_total) if contracts_total else 0.0
    edge_count_stats = _quantiles(successful_contracts["dfg_edge_count"].tolist())

    successful_edges = dfg_edges_df[dfg_edges_df["dfg_success"] & (dfg_edges_df["edge_type"] != "")]
    edge_type_counts = (
        successful_edges["edge_type"].value_counts().to_dict() if not successful_edges.empty else {}
    )
    edge_type_counts = {str(key): int(value) for key, value in edge_type_counts.items()}

    output_notes: List[Dict[str, str]] = []
    resolved_edges_out = _resolve_output_path(config.dfg_edges_out_path, output_notes)
    resolved_report_out = _resolve_output_path(config.report_json_path, output_notes)
    resolved_manifest_out = _resolve_output_path(config.run_manifest_json_path, output_notes)

    resolved_edges_out.parent.mkdir(parents=True, exist_ok=True)
    dfg_edges_df.to_parquet(resolved_edges_out, index=False)

    report: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "phase2_task4_dfg_construction",
        "inputs": {
            "instructions_parquet": _rel(config.input_instructions_path),
            "basic_blocks_parquet": _rel(config.input_basic_blocks_path),
            "cfg_edges_parquet": _rel(config.input_cfg_edges_path),
        },
        "outputs": {
            "dfg_edges_parquet": _rel(resolved_edges_out),
            "dfg_report_json": _rel(resolved_report_out),
            "run_manifest_json": _rel(resolved_manifest_out),
        },
        "configured_outputs": {
            "dfg_edges_parquet": _rel(config.dfg_edges_out_path),
            "dfg_report_json": _rel(config.report_json_path),
            "run_manifest_json": _rel(config.run_manifest_json_path),
        },
        "output_resolution_notes": output_notes,
        "contracts_processed": contracts_total,
        "contracts_with_dfg": dfg_success_count,
        "contracts_without_dfg_count": int(failed_contracts.shape[0]),
        "dfg_success_rate": dfg_success_rate,
        "contracts_with_empty_dfg_count": int(empty_dfg_contracts.shape[0]),
        "contracts_with_empty_dfg": empty_dfg_contracts[CONTRACT_ID_COLUMN].tolist(),
        "contracts_without_dfg": failed_contracts[CONTRACT_ID_COLUMN].tolist(),
        "contracts_without_dfg_details": failed_contracts[
            [
                CONTRACT_ID_COLUMN,
                "split",
                "is_proxy_like",
                "is_stub_like",
                "instruction_parse_success",
                "instruction_failure_mode",
                "cfg_success",
                "cfg_failure_mode",
                "failure_mode",
                "failure_detail",
            ]
        ].to_dict(orient="records"),
        "per_edge_type_counts": edge_type_counts,
        "dfg_edge_count_stats": edge_count_stats,
        "failure_mode_counts": (
            failed_contracts["failure_mode"].value_counts().to_dict() if not failed_contracts.empty else {}
        ),
        "known_limitations": [
            "Stack, storage, and memory state are reset at basic-block boundaries to avoid over-claiming cross-branch dependencies.",
            "storage_flow is emitted only for exact constant-slot SSTORE->SLOAD matches recoverable from local abstract state.",
            "memory_flow is emitted only for exact constant-offset MSTORE/MSTORE8->MLOAD matches recoverable from local abstract state.",
            "control_dependency is emitted only when CFG provides explicit jumpi_true and jumpi_false edges and the JUMPI condition producer is known.",
            "Unknown/unsupported opcodes are treated conservatively and do not produce speculative dependencies.",
        ],
        "conservative_assumptions": [
            "Prefer missing a dependency over introducing an uncertain one.",
            "Do not infer dependencies when slot/address values are symbolic or unresolved.",
            "Do not infer aliasing across dynamic memory ranges or unresolved storage keys.",
            "Preserve edge_type labels for straightforward CFG-only vs CFG+DFG ablations.",
        ],
        "rows_written": {"dfg_edges_rows": int(len(dfg_edges_df))},
        "graph_id_alignment_ok": True,
    }
    _write_json(report, resolved_report_out)

    run_manifest = {
        "run_started_utc": run_started.isoformat(),
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "entrypoint": "python -m src.preprocessing.dfg_builder --config configs/phase2.yaml",
        "config_path": _rel(config_path.resolve()),
        "inputs": report["inputs"],
        "outputs": report["outputs"],
        "summary": {
            "contracts_processed": contracts_total,
            "contracts_with_dfg": dfg_success_count,
            "contracts_without_dfg": int(failed_contracts.shape[0]),
            "contracts_with_empty_dfg": int(empty_dfg_contracts.shape[0]),
            "dfg_success_rate": dfg_success_rate,
        },
    }
    _write_json(run_manifest, resolved_manifest_out)
    return report


def _print_report_summary(report: Mapping[str, Any]) -> None:
    print(f"contracts processed: {report['contracts_processed']}")
    print(f"contracts with dfg: {report['contracts_with_dfg']}")
    print(f"contracts without dfg: {report['contracts_without_dfg_count']}")
    print(f"contracts with empty dfg: {report['contracts_with_empty_dfg_count']}")
    print(f"dfg success rate: {report['dfg_success_rate']:.4f}")
    stats = report.get("dfg_edge_count_stats", {})
    print(f"dfg edge count median/p95: {stats.get('median')} / {stats.get('p95')}")
    print(f"per-edge-type counts: {report.get('per_edge_type_counts', {})}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build conservative Phase 2 DFG edges from opcode instructions.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    report = run_dfg_building(args.config.resolve())
    _print_report_summary(report)


if __name__ == "__main__":
    main()
