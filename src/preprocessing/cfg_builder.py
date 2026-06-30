import argparse
import json
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import pyevmasm
import yaml

from src.preprocessing.opcodes import disassemble_runtime_bytecode, normalize_runtime_hex_safely

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs/phase2.yaml"
DEFAULT_INPUT_CONTRACTS = PROJECT_ROOT / "data/curated/main_benchmark_contracts.parquet"
DEFAULT_INPUT_INSTRUCTIONS = PROJECT_ROOT / "data/curated/phase2_instructions.parquet"
DEFAULT_ETHERSOLVE_JAR = PROJECT_ROOT / "tools/EtherSolve.jar"
DEFAULT_ETHERSOLVE_JAR_FALLBACK = PROJECT_ROOT / "tools/EtherSolve/artifact/EtherSolve.jar"
DEFAULT_CFG_NODES_OUT = PROJECT_ROOT / "data/curated/phase2_cfg_nodes.parquet"
DEFAULT_CFG_EDGES_OUT = PROJECT_ROOT / "data/curated/phase2_cfg_edges.parquet"
DEFAULT_REPORT_JSON = PROJECT_ROOT / "reports/phase2/cfg_report.json"
DEFAULT_RUN_MANIFEST_JSON = PROJECT_ROOT / "reports/phase2/cfg_run_manifest.json"

CONTRACT_ID_COLUMN = "fp_runtime_unified"
RUNTIME_COLUMN = "runtime_bytecode_hex_normalized"
BLOCK_TERMINATOR_OPCODES = {0x00, 0x56, 0x57, 0xF3, 0xFD, 0xFE, 0xFF}
TERMINAL_OPCODES = {0x00, 0xF3, 0xFD, 0xFE, 0xFF}
JUMP_OPCODE = 0x56
JUMPI_OPCODE = 0x57
VALID_EDGE_TYPES = {"fallthrough", "jump", "jumpi_true", "jumpi_false", "terminal"}


@dataclass(frozen=True)
class CfgConfig:
    input_contracts_path: Path
    input_instructions_path: Path
    ethersolve_jar_path: Path
    cfg_nodes_out_path: Path
    cfg_edges_out_path: Path
    report_json_path: Path
    run_manifest_json_path: Path
    ethersolve_timeout_seconds: int
    java_bin: str


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _write_json(payload: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def _resolve_path(path_value: Any) -> Path:
    raw = Path(str(path_value))
    if raw.is_absolute():
        return raw
    return (PROJECT_ROOT / raw).resolve()


def _safe_read_mapping(config_path: Path) -> Mapping[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp)
    if not isinstance(data, dict):
        raise ValueError("Phase 2 config must be a mapping.")
    return data


def _load_config(config_path: Path) -> CfgConfig:
    raw = _safe_read_mapping(config_path)
    outputs = raw.get("outputs", {})
    if outputs is None:
        outputs = {}
    if not isinstance(outputs, dict):
        raise ValueError("`outputs` must be a mapping when provided.")

    cfg_section = raw.get("cfg", {})
    if cfg_section is None:
        cfg_section = {}
    if not isinstance(cfg_section, dict):
        raise ValueError("`cfg` must be a mapping when provided.")

    input_contracts = cfg_section.get("input_contracts") or outputs.get("contracts_parquet") or str(
        DEFAULT_INPUT_CONTRACTS
    )
    input_instructions = cfg_section.get("input_instructions") or str(DEFAULT_INPUT_INSTRUCTIONS)
    ethersolve_jar = cfg_section.get("ethersolve_jar") or str(DEFAULT_ETHERSOLVE_JAR)
    cfg_nodes = cfg_section.get("cfg_nodes_parquet") or str(DEFAULT_CFG_NODES_OUT)
    cfg_edges = cfg_section.get("cfg_edges_parquet") or str(DEFAULT_CFG_EDGES_OUT)
    report_json = cfg_section.get("report_json") or str(DEFAULT_REPORT_JSON)
    run_manifest_json = cfg_section.get("run_manifest_json") or str(DEFAULT_RUN_MANIFEST_JSON)
    timeout_seconds = int(cfg_section.get("ethersolve_timeout_seconds", 30))
    if timeout_seconds <= 0:
        raise ValueError("`cfg.ethersolve_timeout_seconds` must be a positive integer.")
    java_bin = str(cfg_section.get("java_bin", "java")).strip() or "java"

    return CfgConfig(
        input_contracts_path=_resolve_path(input_contracts),
        input_instructions_path=_resolve_path(input_instructions),
        ethersolve_jar_path=_resolve_path(ethersolve_jar),
        cfg_nodes_out_path=_resolve_path(cfg_nodes),
        cfg_edges_out_path=_resolve_path(cfg_edges),
        report_json_path=_resolve_path(report_json),
        run_manifest_json_path=_resolve_path(run_manifest_json),
        ethersolve_timeout_seconds=timeout_seconds,
        java_bin=java_bin,
    )


def _as_nullable_bool(value: Any) -> Optional[bool]:
    if pd.isna(value):
        return None
    return bool(value)


def _first_non_empty(values: pd.Series) -> str:
    for value in values.tolist():
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _load_instruction_status(instructions: pd.DataFrame) -> pd.DataFrame:
    required = {CONTRACT_ID_COLUMN, "parse_success", "failure_mode"}
    missing = sorted(required - set(instructions.columns))
    if missing:
        raise ValueError(f"Input instructions parquet missing required columns: {missing}")

    working = instructions[list(required)].copy()
    working[CONTRACT_ID_COLUMN] = working[CONTRACT_ID_COLUMN].fillna("").astype(str).str.strip()
    working = working[working[CONTRACT_ID_COLUMN] != ""]
    working["parse_success"] = working["parse_success"].fillna(False).astype(bool)
    working["failure_mode"] = working["failure_mode"].fillna("").astype(str).str.strip()

    status = (
        working.groupby(CONTRACT_ID_COLUMN, sort=False)
        .agg(
            instruction_parse_success=("parse_success", "max"),
            instruction_failure_mode=("failure_mode", _first_non_empty),
        )
        .reset_index()
    )
    status["instruction_parse_success"] = status["instruction_parse_success"].astype(bool)
    status["instruction_failure_mode"] = status["instruction_failure_mode"].fillna("")
    return status


def _summarize_block_bytecode(bytecode_hex: str) -> Tuple[int, str, Optional[int]]:
    text = str(bytecode_hex).strip()
    if not text:
        return 0, "", None
    try:
        instructions = disassemble_runtime_bytecode(text)
    except ValueError:
        return 0, "", None
    if not instructions:
        return 0, "", None
    last = instructions[-1]
    return len(instructions), str(last["opcode"]), int(last["opcode_id"])


def _classify_edge(
    terminator_opcode_id: Optional[int],
    src_start_pc: int,
    src_length: int,
    dst_start_pc: int,
) -> str:
    fallthrough_pc = src_start_pc + max(src_length, 0)
    if terminator_opcode_id == JUMPI_OPCODE:
        return "jumpi_false" if dst_start_pc == fallthrough_pc else "jumpi_true"
    if terminator_opcode_id == JUMP_OPCODE:
        return "jump"
    if terminator_opcode_id in TERMINAL_OPCODES:
        return "terminal"
    return "fallthrough" if dst_start_pc == fallthrough_pc else "jump"


def run_ethersolve_cfg(
    runtime_hex: str,
    ethersolve_jar_path: Path,
    timeout_seconds: int,
    java_bin: str = "java",
) -> Dict[str, Any]:
    if not ethersolve_jar_path.exists():
        fallback_jar = DEFAULT_ETHERSOLVE_JAR_FALLBACK.resolve()
        if ethersolve_jar_path.resolve() == DEFAULT_ETHERSOLVE_JAR.resolve() and fallback_jar.exists():
            ethersolve_jar_path = fallback_jar
        else:
            return {
                "success": False,
                "failure_mode": "ethersolve_jar_missing",
                "failure_detail": str(ethersolve_jar_path),
                "nodes": [],
                "edges": [],
            }

    with tempfile.TemporaryDirectory(prefix="ethersolve_cfg_") as tmp_dir:
        output_path = Path(tmp_dir) / "report.json"
        command = [
            java_bin,
            "-jar",
            str(ethersolve_jar_path),
            "-r",
            "-j",
            "-o",
            output_path.name,
            runtime_hex,
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=tmp_dir,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except FileNotFoundError:
            return {
                "success": False,
                "failure_mode": "java_not_found",
                "failure_detail": java_bin,
                "nodes": [],
                "edges": [],
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "failure_mode": "ethersolve_timeout",
                "failure_detail": f"{timeout_seconds}s",
                "nodes": [],
                "edges": [],
            }

        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        error_excerpt = stderr or stdout
        if completed.returncode != 0:
            return {
                "success": False,
                "failure_mode": "ethersolve_nonzero_exit",
                "failure_detail": error_excerpt[:400],
                "nodes": [],
                "edges": [],
            }
        if not output_path.exists():
            return {
                "success": False,
                "failure_mode": "ethersolve_missing_output",
                "failure_detail": "",
                "nodes": [],
                "edges": [],
            }

        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return {
                "success": False,
                "failure_mode": "ethersolve_invalid_json",
                "failure_detail": str(exc),
                "nodes": [],
                "edges": [],
            }

    runtime_cfg = payload.get("runtimeCfg")
    if not isinstance(runtime_cfg, dict):
        return {
            "success": False,
            "failure_mode": "ethersolve_missing_runtime_cfg",
            "failure_detail": "",
            "nodes": [],
            "edges": [],
        }

    raw_nodes = runtime_cfg.get("nodes")
    raw_successors = runtime_cfg.get("successors")
    if not isinstance(raw_nodes, list) or not isinstance(raw_successors, list):
        return {
            "success": False,
            "failure_mode": "ethersolve_invalid_runtime_cfg_schema",
            "failure_detail": "",
            "nodes": [],
            "edges": [],
        }

    nodes: List[Dict[str, Any]] = []
    node_lookup: Dict[int, Dict[str, Any]] = {}
    for raw_node in raw_nodes:
        if not isinstance(raw_node, dict):
            return {
                "success": False,
                "failure_mode": "ethersolve_invalid_node_schema",
                "failure_detail": "",
                "nodes": [],
                "edges": [],
            }
        try:
            start_pc = int(raw_node.get("offset"))
            length = int(raw_node.get("length"))
            stack_balance = int(raw_node.get("stackBalance"))
        except (TypeError, ValueError):
            return {
                "success": False,
                "failure_mode": "ethersolve_invalid_node_schema",
                "failure_detail": "",
                "nodes": [],
                "edges": [],
            }
        if start_pc < 0 or length < 0:
            return {
                "success": False,
                "failure_mode": "ethersolve_invalid_node_bounds",
                "failure_detail": "",
                "nodes": [],
                "edges": [],
            }

        instruction_count, end_opcode, terminator_opcode_id = _summarize_block_bytecode(
            str(raw_node.get("bytecodeHex", ""))
        )
        end_pc = start_pc + (length - 1 if length > 0 else 0)
        node = {
            "node_id": int(start_pc),
            "start_pc": int(start_pc),
            "end_pc": int(end_pc),
            "instruction_count": int(instruction_count),
            "node_type": str(raw_node.get("type", "unknown")),
            "end_opcode": end_opcode,
            "stack_balance": int(stack_balance),
        }
        nodes.append(node)
        node_lookup[node["node_id"]] = {
            "length": int(length),
            "terminator_opcode_id": terminator_opcode_id,
        }

    if not nodes:
        return {
            "success": False,
            "failure_mode": "ethersolve_empty_cfg",
            "failure_detail": "",
            "nodes": [],
            "edges": [],
        }

    edges: List[Dict[str, Any]] = []
    seen_edges = set()
    for raw_successor in raw_successors:
        if not isinstance(raw_successor, dict):
            return {
                "success": False,
                "failure_mode": "ethersolve_invalid_successor_schema",
                "failure_detail": "",
                "nodes": [],
                "edges": [],
            }
        if "from" not in raw_successor or "to" not in raw_successor:
            return {
                "success": False,
                "failure_mode": "ethersolve_invalid_successor_schema",
                "failure_detail": "",
                "nodes": [],
                "edges": [],
            }
        targets = raw_successor.get("to")
        if not isinstance(targets, list):
            return {
                "success": False,
                "failure_mode": "ethersolve_invalid_successor_schema",
                "failure_detail": "",
                "nodes": [],
                "edges": [],
            }
        try:
            src_node_id = int(raw_successor["from"])
        except (TypeError, ValueError):
            return {
                "success": False,
                "failure_mode": "ethersolve_invalid_successor_schema",
                "failure_detail": "",
                "nodes": [],
                "edges": [],
            }
        if src_node_id not in node_lookup:
            continue

        for target in targets:
            try:
                dst_node_id = int(target)
            except (TypeError, ValueError):
                return {
                    "success": False,
                    "failure_mode": "ethersolve_invalid_successor_schema",
                    "failure_detail": "",
                    "nodes": [],
                    "edges": [],
                }
            src_meta = node_lookup[src_node_id]
            edge_type = _classify_edge(
                terminator_opcode_id=src_meta["terminator_opcode_id"],
                src_start_pc=src_node_id,
                src_length=src_meta["length"],
                dst_start_pc=dst_node_id,
            )
            edge_key = (src_node_id, dst_node_id, edge_type)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            edges.append(
                {
                    "src_node_id": int(src_node_id),
                    "dst_node_id": int(dst_node_id),
                    "edge_type": edge_type,
                }
            )

    return {
        "success": True,
        "failure_mode": "",
        "failure_detail": "",
        "nodes": nodes,
        "edges": edges,
    }


def _resolve_static_jump_target(block_instructions: Sequence[Any]) -> Optional[int]:
    if len(block_instructions) < 2:
        return None
    candidate = block_instructions[-2]
    opcode = int(candidate.opcode)
    if 0x60 <= opcode <= 0x7F and candidate.operand is not None:
        return int(candidate.operand)
    return None


def build_pyevmasm_fallback_cfg(runtime_hex: str) -> Dict[str, Any]:
    instructions = list(pyevmasm.disassemble_all(bytes.fromhex(runtime_hex)))
    if not instructions:
        return {
            "success": False,
            "failure_mode": "fallback_empty_disassembly",
            "failure_detail": "",
            "nodes": [],
            "edges": [],
        }

    block_starts = {int(instructions[0].pc)}
    for idx, instruction in enumerate(instructions):
        opcode = int(instruction.opcode)
        if opcode == 0x5B:
            block_starts.add(int(instruction.pc))
        if opcode in BLOCK_TERMINATOR_OPCODES and (idx + 1) < len(instructions):
            block_starts.add(int(instructions[idx + 1].pc))
    sorted_starts = sorted(block_starts)
    pc_to_index = {int(instruction.pc): idx for idx, instruction in enumerate(instructions)}

    nodes: List[Dict[str, Any]] = []
    block_lookup: Dict[int, Dict[str, Any]] = {}
    for idx, start_pc in enumerate(sorted_starts):
        start_idx = pc_to_index[start_pc]
        next_start_idx = (
            pc_to_index[sorted_starts[idx + 1]] if idx + 1 < len(sorted_starts) else len(instructions)
        )
        block_instructions = instructions[start_idx:next_start_idx]
        if not block_instructions:
            continue

        last = block_instructions[-1]
        node = {
            "node_id": int(start_pc),
            "start_pc": int(start_pc),
            "end_pc": int(last.pc),
            "instruction_count": int(len(block_instructions)),
            "node_type": "basic_block",
            "end_opcode": str(last.name),
            "stack_balance": None,
        }
        nodes.append(node)
        block_lookup[start_pc] = {
            "last_opcode": int(last.opcode),
            "block_instructions": block_instructions,
            "fallthrough": sorted_starts[idx + 1] if idx + 1 < len(sorted_starts) else None,
        }

    if not nodes:
        return {
            "success": False,
            "failure_mode": "fallback_empty_blocks",
            "failure_detail": "",
            "nodes": [],
            "edges": [],
        }

    edges: List[Dict[str, Any]] = []
    seen_edges = set()

    def _add_edge(src_node_id: int, dst_node_id: int, edge_type: str) -> None:
        edge_key = (src_node_id, dst_node_id, edge_type)
        if edge_key in seen_edges:
            return
        seen_edges.add(edge_key)
        edges.append(
            {
                "src_node_id": int(src_node_id),
                "dst_node_id": int(dst_node_id),
                "edge_type": edge_type,
            }
        )

    unresolved_jump_count = 0
    for node in nodes:
        src_node_id = int(node["node_id"])
        block_meta = block_lookup[src_node_id]
        last_opcode = int(block_meta["last_opcode"])
        block_instructions = block_meta["block_instructions"]
        fallthrough = block_meta["fallthrough"]

        if last_opcode == JUMP_OPCODE:
            target = _resolve_static_jump_target(block_instructions)
            if target is not None and target in block_lookup:
                _add_edge(src_node_id, target, "jump")
            else:
                unresolved_jump_count += 1
            continue

        if last_opcode == JUMPI_OPCODE:
            target = _resolve_static_jump_target(block_instructions)
            if target is not None and target in block_lookup:
                _add_edge(src_node_id, target, "jumpi_true")
            else:
                unresolved_jump_count += 1
            if fallthrough is not None:
                _add_edge(src_node_id, int(fallthrough), "jumpi_false")
            continue

        if last_opcode in TERMINAL_OPCODES:
            continue
        if fallthrough is not None:
            _add_edge(src_node_id, int(fallthrough), "fallthrough")

    return {
        "success": True,
        "failure_mode": "",
        "failure_detail": "",
        "nodes": nodes,
        "edges": edges,
        "unresolved_jump_count": int(unresolved_jump_count),
    }


def build_cfg_for_runtime(
    runtime_hex: Any,
    ethersolve_jar_path: Path,
    timeout_seconds: int,
    java_bin: str = "java",
) -> Dict[str, Any]:
    normalized_runtime_hex, normalize_failure = normalize_runtime_hex_safely(runtime_hex)
    if normalize_failure:
        return {
            "cfg_success": False,
            "backend_used": "none",
            "failure_mode": normalize_failure,
            "failure_detail": "",
            "ethersolve_failure_mode": "",
            "ethersolve_failure_detail": "",
            "fallback_failure_mode": normalize_failure,
            "fallback_failure_detail": "",
            "nodes": [],
            "edges": [],
        }

    ethersolve_result = run_ethersolve_cfg(
        runtime_hex=normalized_runtime_hex,
        ethersolve_jar_path=ethersolve_jar_path,
        timeout_seconds=timeout_seconds,
        java_bin=java_bin,
    )
    if ethersolve_result["success"]:
        return {
            "cfg_success": True,
            "backend_used": "ethersolve",
            "failure_mode": "",
            "failure_detail": "",
            "ethersolve_failure_mode": "",
            "ethersolve_failure_detail": "",
            "fallback_failure_mode": "",
            "fallback_failure_detail": "",
            "nodes": ethersolve_result["nodes"],
            "edges": ethersolve_result["edges"],
        }

    fallback_result = build_pyevmasm_fallback_cfg(normalized_runtime_hex)
    if fallback_result["success"]:
        return {
            "cfg_success": True,
            "backend_used": "pyevmasm_fallback",
            "failure_mode": "",
            "failure_detail": "",
            "ethersolve_failure_mode": ethersolve_result["failure_mode"],
            "ethersolve_failure_detail": ethersolve_result["failure_detail"],
            "fallback_failure_mode": "",
            "fallback_failure_detail": "",
            "nodes": fallback_result["nodes"],
            "edges": fallback_result["edges"],
        }

    return {
        "cfg_success": False,
        "backend_used": "none",
        "failure_mode": fallback_result["failure_mode"],
        "failure_detail": fallback_result["failure_detail"],
        "ethersolve_failure_mode": ethersolve_result["failure_mode"],
        "ethersolve_failure_detail": ethersolve_result["failure_detail"],
        "fallback_failure_mode": fallback_result["failure_mode"],
        "fallback_failure_detail": fallback_result["failure_detail"],
        "nodes": [],
        "edges": [],
    }


def _quantiles(values: Iterable[int]) -> Dict[str, Optional[float]]:
    series = pd.Series(list(values), dtype="float64")
    if series.empty:
        return {"median": None, "p95": None}
    return {
        "median": float(series.quantile(0.5)),
        "p95": float(series.quantile(0.95)),
    }


def _resolve_output_path(
    configured_path: Path,
    output_notes: List[Dict[str, str]],
) -> Path:
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


def _finalize_nodes_df(rows: List[Tuple[Any, ...]]) -> pd.DataFrame:
    columns = [
        "graph_id",
        CONTRACT_ID_COLUMN,
        "split",
        "has_cgt",
        "has_dappscan",
        "is_proxy_like",
        "is_stub_like",
        "instruction_parse_success",
        "instruction_failure_mode",
        "cfg_success",
        "backend_used",
        "failure_mode",
        "ethersolve_failure_mode",
        "fallback_failure_mode",
        "node_id",
        "start_pc",
        "end_pc",
        "instruction_count",
        "node_type",
        "end_opcode",
        "stack_balance",
    ]
    df = pd.DataFrame.from_records(rows, columns=columns)
    if df.empty:
        return df

    for col in ["node_id", "start_pc", "end_pc", "instruction_count", "stack_balance"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ["has_cgt", "has_dappscan", "is_proxy_like", "is_stub_like", "instruction_parse_success"]:
        df[col] = df[col].astype("boolean")
    df["cfg_success"] = df["cfg_success"].astype(bool)
    for col in ["instruction_failure_mode", "backend_used", "failure_mode", "ethersolve_failure_mode", "fallback_failure_mode"]:
        df[col] = df[col].fillna("").astype(str)
    df["node_type"] = df["node_type"].fillna("").astype(str)
    df["end_opcode"] = df["end_opcode"].fillna("").astype(str)
    return df


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
        "instruction_failure_mode",
        "cfg_success",
        "backend_used",
        "failure_mode",
        "ethersolve_failure_mode",
        "fallback_failure_mode",
        "src_node_id",
        "dst_node_id",
        "edge_type",
    ]
    df = pd.DataFrame.from_records(rows, columns=columns)
    if df.empty:
        return df

    for col in ["src_node_id", "dst_node_id"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ["has_cgt", "has_dappscan", "is_proxy_like", "is_stub_like", "instruction_parse_success"]:
        df[col] = df[col].astype("boolean")
    df["cfg_success"] = df["cfg_success"].astype(bool)
    for col in ["instruction_failure_mode", "backend_used", "failure_mode", "ethersolve_failure_mode", "fallback_failure_mode"]:
        df[col] = df[col].fillna("").astype(str)
    df["edge_type"] = df["edge_type"].fillna("").astype(str)
    return df


def validate_cfg_edge_schema(edges_df: pd.DataFrame) -> None:
    required_columns = {
        "graph_id",
        CONTRACT_ID_COLUMN,
        "cfg_success",
        "backend_used",
        "failure_mode",
        "src_node_id",
        "dst_node_id",
        "edge_type",
    }
    missing = sorted(required_columns - set(edges_df.columns))
    if missing:
        raise ValueError(f"CFG edges parquet missing required columns: {missing}")

    if edges_df.empty:
        return

    successful_edges = edges_df[edges_df["cfg_success"]].copy()
    if successful_edges.empty:
        return

    if bool(successful_edges["src_node_id"].isna().any()) or bool(successful_edges["dst_node_id"].isna().any()):
        raise ValueError("Successful CFG edges must include non-null src_node_id and dst_node_id.")

    edge_types = successful_edges["edge_type"].fillna("").astype(str)
    invalid_edge_types = sorted(set(edge_types.tolist()) - VALID_EDGE_TYPES)
    if invalid_edge_types:
        raise ValueError(f"Unsupported CFG edge type(s): {invalid_edge_types}")


def run_cfg_building(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    run_started = datetime.now(timezone.utc)
    config = _load_config(config_path.resolve())

    if not config.input_contracts_path.exists():
        raise FileNotFoundError(f"Input contracts parquet not found: {config.input_contracts_path}")
    if not config.input_instructions_path.exists():
        raise FileNotFoundError(f"Input instructions parquet not found: {config.input_instructions_path}")

    contracts = pd.read_parquet(config.input_contracts_path).copy()
    instructions = pd.read_parquet(config.input_instructions_path).copy()
    required_contract_cols = {CONTRACT_ID_COLUMN, RUNTIME_COLUMN}
    missing_contract_cols = sorted(required_contract_cols - set(contracts.columns))
    if missing_contract_cols:
        raise ValueError(f"Input contracts missing required columns: {missing_contract_cols}")

    contracts[CONTRACT_ID_COLUMN] = contracts[CONTRACT_ID_COLUMN].fillna("").astype(str).str.strip()
    contracts = contracts[contracts[CONTRACT_ID_COLUMN] != ""].drop_duplicates(
        subset=[CONTRACT_ID_COLUMN], keep="first"
    )
    if contracts.empty:
        raise ValueError("No contracts found in input parquet after cleaning contract IDs.")

    instruction_status = _load_instruction_status(instructions)
    contracts = contracts.merge(instruction_status, on=CONTRACT_ID_COLUMN, how="left")
    contracts["instruction_parse_success"] = contracts["instruction_parse_success"].fillna(False).astype(bool)
    contracts["instruction_failure_mode"] = contracts["instruction_failure_mode"].fillna("missing_instruction_rows")
    contracts.loc[contracts["instruction_parse_success"], "instruction_failure_mode"] = (
        contracts.loc[contracts["instruction_parse_success"], "instruction_failure_mode"].fillna("")
    )
    contracts["instruction_failure_mode"] = contracts["instruction_failure_mode"].astype(str).str.strip()

    nodes_rows: List[Tuple[Any, ...]] = []
    edges_rows: List[Tuple[Any, ...]] = []
    contract_rows: List[Dict[str, Any]] = []

    for _, row in contracts.iterrows():
        contract_id = str(row[CONTRACT_ID_COLUMN]).strip()
        split = str(row["split"]).strip() if "split" in row and not pd.isna(row["split"]) else "unknown"
        has_cgt = _as_nullable_bool(row.get("has_cgt"))
        has_dappscan = _as_nullable_bool(row.get("has_dappscan"))
        is_proxy_like = _as_nullable_bool(row.get("is_proxy_like"))
        is_stub_like = _as_nullable_bool(row.get("is_stub_like"))
        instruction_parse_success = bool(row.get("instruction_parse_success", False))
        instruction_failure_mode = str(row.get("instruction_failure_mode", "")).strip()

        cfg_result = build_cfg_for_runtime(
            runtime_hex=row[RUNTIME_COLUMN],
            ethersolve_jar_path=config.ethersolve_jar_path,
            timeout_seconds=config.ethersolve_timeout_seconds,
            java_bin=config.java_bin,
        )

        node_count = int(len(cfg_result["nodes"])) if cfg_result["cfg_success"] else 0
        edge_count = int(len(cfg_result["edges"])) if cfg_result["cfg_success"] else 0
        contract_rows.append(
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
                "cfg_success": bool(cfg_result["cfg_success"]),
                "backend_used": str(cfg_result["backend_used"]),
                "failure_mode": str(cfg_result["failure_mode"]),
                "failure_detail": str(cfg_result.get("failure_detail", "")),
                "ethersolve_failure_mode": str(cfg_result.get("ethersolve_failure_mode", "")),
                "ethersolve_failure_detail": str(cfg_result.get("ethersolve_failure_detail", "")),
                "fallback_failure_mode": str(cfg_result.get("fallback_failure_mode", "")),
                "fallback_failure_detail": str(cfg_result.get("fallback_failure_detail", "")),
                "node_count": node_count,
                "edge_count": edge_count,
            }
        )

        if cfg_result["cfg_success"]:
            for node in cfg_result["nodes"]:
                nodes_rows.append(
                    (
                        contract_id,
                        contract_id,
                        split,
                        has_cgt,
                        has_dappscan,
                        is_proxy_like,
                        is_stub_like,
                        instruction_parse_success,
                        instruction_failure_mode,
                        True,
                        cfg_result["backend_used"],
                        "",
                        cfg_result.get("ethersolve_failure_mode", ""),
                        "",
                        node["node_id"],
                        node["start_pc"],
                        node["end_pc"],
                        node["instruction_count"],
                        node["node_type"],
                        node["end_opcode"],
                        node["stack_balance"],
                    )
                )
            for edge in cfg_result["edges"]:
                edges_rows.append(
                    (
                        contract_id,
                        contract_id,
                        split,
                        has_cgt,
                        has_dappscan,
                        is_proxy_like,
                        is_stub_like,
                        instruction_parse_success,
                        instruction_failure_mode,
                        True,
                        cfg_result["backend_used"],
                        "",
                        cfg_result.get("ethersolve_failure_mode", ""),
                        "",
                        edge["src_node_id"],
                        edge["dst_node_id"],
                        edge["edge_type"],
                    )
                )
            continue

        nodes_rows.append(
            (
                contract_id,
                contract_id,
                split,
                has_cgt,
                has_dappscan,
                is_proxy_like,
                is_stub_like,
                instruction_parse_success,
                instruction_failure_mode,
                False,
                "none",
                cfg_result["failure_mode"],
                cfg_result.get("ethersolve_failure_mode", ""),
                cfg_result.get("fallback_failure_mode", ""),
                None,
                None,
                None,
                None,
                "",
                "",
                None,
            )
        )
        edges_rows.append(
            (
                contract_id,
                contract_id,
                split,
                has_cgt,
                has_dappscan,
                is_proxy_like,
                is_stub_like,
                instruction_parse_success,
                instruction_failure_mode,
                False,
                "none",
                cfg_result["failure_mode"],
                cfg_result.get("ethersolve_failure_mode", ""),
                cfg_result.get("fallback_failure_mode", ""),
                None,
                None,
                "",
            )
        )

    nodes_df = _finalize_nodes_df(nodes_rows)
    edges_df = _finalize_edges_df(edges_rows)
    validate_cfg_edge_schema(edges_df)

    graph_id_alignment_ok = True
    if not nodes_df.empty:
        graph_id_alignment_ok = graph_id_alignment_ok and bool(
            (nodes_df["graph_id"] == nodes_df[CONTRACT_ID_COLUMN]).all()
        )
    if not edges_df.empty:
        graph_id_alignment_ok = graph_id_alignment_ok and bool(
            (edges_df["graph_id"] == edges_df[CONTRACT_ID_COLUMN]).all()
        )
    if not graph_id_alignment_ok:
        raise ValueError("Graph ID alignment check failed: graph_id must match fp_runtime_unified.")

    contract_df = pd.DataFrame(contract_rows)
    successful = contract_df[contract_df["cfg_success"]].copy()
    failed = contract_df[~contract_df["cfg_success"]].copy()

    contracts_total = int(len(contract_df))
    cfg_success_count = int(successful.shape[0])
    ethersolve_success_count = int((contract_df["backend_used"] == "ethersolve").sum())
    fallback_success_count = int((contract_df["backend_used"] == "pyevmasm_fallback").sum())
    cfg_success_rate = (cfg_success_count / contracts_total) if contracts_total else 0.0
    ethersolve_success_rate = (ethersolve_success_count / contracts_total) if contracts_total else 0.0
    fallback_rate = (fallback_success_count / contracts_total) if contracts_total else 0.0

    node_stats = _quantiles(successful["node_count"].tolist())
    edge_stats = _quantiles(successful["edge_count"].tolist())

    no_cfg_contracts = failed[CONTRACT_ID_COLUMN].tolist()
    no_cfg_failure_counts = (
        failed["failure_mode"].value_counts() if not failed.empty else pd.Series(dtype=int)
    )
    ethersolve_failure_counts = (
        contract_df[contract_df["ethersolve_failure_mode"].astype(str) != ""]["ethersolve_failure_mode"].value_counts()
    )
    fallback_failure_counts = (
        contract_df[contract_df["fallback_failure_mode"].astype(str) != ""]["fallback_failure_mode"].value_counts()
    )
    split_backend = (
        contract_df.groupby(["split", "backend_used"], dropna=False)
        .size()
        .reset_index(name="contracts")
        .sort_values(["split", "backend_used"])
    )

    output_notes: List[Dict[str, str]] = []
    resolved_nodes_out = _resolve_output_path(config.cfg_nodes_out_path, output_notes)
    resolved_edges_out = _resolve_output_path(config.cfg_edges_out_path, output_notes)
    resolved_report_out = _resolve_output_path(config.report_json_path, output_notes)
    resolved_manifest_out = _resolve_output_path(config.run_manifest_json_path, output_notes)

    resolved_nodes_out.parent.mkdir(parents=True, exist_ok=True)
    resolved_edges_out.parent.mkdir(parents=True, exist_ok=True)
    nodes_df.to_parquet(resolved_nodes_out, index=False)
    edges_df.to_parquet(resolved_edges_out, index=False)

    report: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "phase2_task3_cfg_construction",
        "inputs": {
            "contracts_parquet": _rel(config.input_contracts_path),
            "instructions_parquet": _rel(config.input_instructions_path),
            "ethersolve_jar_configured": _rel(config.ethersolve_jar_path),
            "java_bin": config.java_bin,
            "ethersolve_timeout_seconds": config.ethersolve_timeout_seconds,
        },
        "outputs": {
            "cfg_nodes_parquet": _rel(resolved_nodes_out),
            "cfg_edges_parquet": _rel(resolved_edges_out),
            "cfg_report_json": _rel(resolved_report_out),
            "run_manifest_json": _rel(resolved_manifest_out),
        },
        "configured_outputs": {
            "cfg_nodes_parquet": _rel(config.cfg_nodes_out_path),
            "cfg_edges_parquet": _rel(config.cfg_edges_out_path),
            "cfg_report_json": _rel(config.report_json_path),
            "run_manifest_json": _rel(config.run_manifest_json_path),
        },
        "output_resolution_notes": output_notes,
        "contracts_processed": contracts_total,
        "contracts_with_cfg": cfg_success_count,
        "contracts_without_cfg_count": int(failed.shape[0]),
        "cfg_success_rate": cfg_success_rate,
        "ethersolve_success_rate": ethersolve_success_rate,
        "fallback_rate": fallback_rate,
        "backend_counts": {
            "ethersolve": ethersolve_success_count,
            "pyevmasm_fallback": fallback_success_count,
            "none": int((contract_df["backend_used"] == "none").sum()),
        },
        "contracts_without_cfg": no_cfg_contracts,
        "contracts_without_cfg_details": failed[
            [
                CONTRACT_ID_COLUMN,
                "split",
                "is_proxy_like",
                "is_stub_like",
                "instruction_parse_success",
                "instruction_failure_mode",
                "failure_mode",
                "failure_detail",
                "ethersolve_failure_mode",
                "ethersolve_failure_detail",
                "fallback_failure_mode",
                "fallback_failure_detail",
            ]
        ].to_dict(orient="records"),
        "failure_modes": {
            "final_failure_mode_counts": [
                {"failure_mode": str(mode), "count": int(count)}
                for mode, count in no_cfg_failure_counts.items()
            ],
            "ethersolve_failure_mode_counts": [
                {"failure_mode": str(mode), "count": int(count)}
                for mode, count in ethersolve_failure_counts.items()
            ],
            "fallback_failure_mode_counts": [
                {"failure_mode": str(mode), "count": int(count)}
                for mode, count in fallback_failure_counts.items()
            ],
        },
        "node_count_stats": node_stats,
        "edge_count_stats": edge_stats,
        "split_backend_counts": split_backend.to_dict(orient="records"),
        "graph_id_alignment_ok": graph_id_alignment_ok,
        "rows_written": {
            "cfg_nodes_rows": int(len(nodes_df)),
            "cfg_edges_rows": int(len(edges_df)),
        },
    }
    _write_json(report, resolved_report_out)

    run_manifest = {
        "run_started_utc": run_started.isoformat(),
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "entrypoint": "python -m src.preprocessing.cfg_builder --config configs/phase2.yaml",
        "config_path": _rel(config_path.resolve()),
        "inputs": report["inputs"],
        "outputs": report["outputs"],
        "summary": {
            "contracts_processed": contracts_total,
            "contracts_with_cfg": cfg_success_count,
            "contracts_without_cfg": int(failed.shape[0]),
            "cfg_success_rate": cfg_success_rate,
            "ethersolve_success_rate": ethersolve_success_rate,
            "fallback_rate": fallback_rate,
        },
    }
    _write_json(run_manifest, resolved_manifest_out)
    return report


def _print_report_summary(report: Mapping[str, Any]) -> None:
    print(f"contracts processed: {report['contracts_processed']}")
    print(f"contracts with cfg: {report['contracts_with_cfg']}")
    print(f"contracts without cfg: {report['contracts_without_cfg_count']}")
    print(
        "rates (cfg / ethersolve / fallback): "
        f"{report['cfg_success_rate']:.4f} / {report['ethersolve_success_rate']:.4f} / {report['fallback_rate']:.4f}"
    )
    node_stats = report.get("node_count_stats", {})
    edge_stats = report.get("edge_count_stats", {})
    print(f"node count median/p95: {node_stats.get('median')} / {node_stats.get('p95')}")
    print(f"edge count median/p95: {edge_stats.get('median')} / {edge_stats.get('p95')}")
    print(f"graph id alignment ok: {report.get('graph_id_alignment_ok', False)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CFG nodes/edges for Phase 2 main benchmark contracts.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    report = run_cfg_building(args.config.resolve())
    _print_report_summary(report)


if __name__ == "__main__":
    main()
