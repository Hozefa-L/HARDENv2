import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import yaml

from src.curation.bytecode_normalize import normalize_hex

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs/phase2.yaml"
DEFAULT_INPUT_CONTRACTS = PROJECT_ROOT / "data/curated/main_benchmark_contracts.parquet"
DEFAULT_INSTRUCTIONS_OUT = PROJECT_ROOT / "data/curated/phase2_instructions.parquet"
DEFAULT_BASIC_BLOCKS_OUT = PROJECT_ROOT / "data/curated/phase2_basic_blocks.parquet"
DEFAULT_REPORT_JSON = PROJECT_ROOT / "reports/phase2/opcode_preprocessing_report.json"
DEFAULT_RUN_MANIFEST_JSON = PROJECT_ROOT / "reports/phase2/opcode_preprocessing_run_manifest.json"

RUNTIME_COLUMN = "runtime_bytecode_hex_normalized"
CONTRACT_ID_COLUMN = "fp_runtime_unified"
JUMPDEST_OPCODE = 0x5B
TERMINAL_BLOCK_OPCODES = {0x00, 0x56, 0x57, 0xF3, 0xFD, 0xFE, 0xFF}
MISSING_RUNTIME_TOKENS = {"", "nan", "none", "null"}

BASE_OPCODE_NAMES: Dict[int, str] = {
    0x00: "STOP",
    0x01: "ADD",
    0x02: "MUL",
    0x03: "SUB",
    0x04: "DIV",
    0x05: "SDIV",
    0x06: "MOD",
    0x07: "SMOD",
    0x08: "ADDMOD",
    0x09: "MULMOD",
    0x0A: "EXP",
    0x0B: "SIGNEXTEND",
    0x10: "LT",
    0x11: "GT",
    0x12: "SLT",
    0x13: "SGT",
    0x14: "EQ",
    0x15: "ISZERO",
    0x16: "AND",
    0x17: "OR",
    0x18: "XOR",
    0x19: "NOT",
    0x1A: "BYTE",
    0x1B: "SHL",
    0x1C: "SHR",
    0x1D: "SAR",
    0x20: "KECCAK256",
    0x30: "ADDRESS",
    0x31: "BALANCE",
    0x32: "ORIGIN",
    0x33: "CALLER",
    0x34: "CALLVALUE",
    0x35: "CALLDATALOAD",
    0x36: "CALLDATASIZE",
    0x37: "CALLDATACOPY",
    0x38: "CODESIZE",
    0x39: "CODECOPY",
    0x3A: "GASPRICE",
    0x3B: "EXTCODESIZE",
    0x3C: "EXTCODECOPY",
    0x3D: "RETURNDATASIZE",
    0x3E: "RETURNDATACOPY",
    0x3F: "EXTCODEHASH",
    0x40: "BLOCKHASH",
    0x41: "COINBASE",
    0x42: "TIMESTAMP",
    0x43: "NUMBER",
    0x44: "PREVRANDAO",
    0x45: "GASLIMIT",
    0x46: "CHAINID",
    0x47: "SELFBALANCE",
    0x48: "BASEFEE",
    0x49: "BLOBHASH",
    0x4A: "BLOBBASEFEE",
    0x50: "POP",
    0x51: "MLOAD",
    0x52: "MSTORE",
    0x53: "MSTORE8",
    0x54: "SLOAD",
    0x55: "SSTORE",
    0x56: "JUMP",
    0x57: "JUMPI",
    0x58: "PC",
    0x59: "MSIZE",
    0x5A: "GAS",
    0x5B: "JUMPDEST",
    0x5F: "PUSH0",
    0xF0: "CREATE",
    0xF1: "CALL",
    0xF2: "CALLCODE",
    0xF3: "RETURN",
    0xF4: "DELEGATECALL",
    0xF5: "CREATE2",
    0xFA: "STATICCALL",
    0xFD: "REVERT",
    0xFE: "INVALID",
    0xFF: "SELFDESTRUCT",
}


@dataclass(frozen=True)
class OpcodesConfig:
    input_contracts_path: Path
    instructions_out_path: Path
    basic_blocks_out_path: Path
    report_json_path: Path
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


def _load_config(config_path: Path) -> OpcodesConfig:
    raw = _safe_read_mapping(config_path)
    outputs = raw.get("outputs", {})
    if outputs is None:
        outputs = {}
    if not isinstance(outputs, dict):
        raise ValueError("`outputs` must be a mapping when provided.")

    opcodes_cfg = raw.get("opcodes", {})
    if opcodes_cfg is None:
        opcodes_cfg = {}
    if not isinstance(opcodes_cfg, dict):
        raise ValueError("`opcodes` must be a mapping when provided.")

    input_contracts = opcodes_cfg.get("input_contracts") or outputs.get("contracts_parquet") or str(
        DEFAULT_INPUT_CONTRACTS
    )
    instructions_out = opcodes_cfg.get("instructions_parquet") or str(DEFAULT_INSTRUCTIONS_OUT)
    basic_blocks_out = opcodes_cfg.get("basic_blocks_parquet") or str(DEFAULT_BASIC_BLOCKS_OUT)
    report_json = opcodes_cfg.get("report_json") or str(DEFAULT_REPORT_JSON)
    run_manifest_json = opcodes_cfg.get("run_manifest_json") or str(DEFAULT_RUN_MANIFEST_JSON)

    return OpcodesConfig(
        input_contracts_path=_resolve_path(input_contracts),
        instructions_out_path=_resolve_path(instructions_out),
        basic_blocks_out_path=_resolve_path(basic_blocks_out),
        report_json_path=_resolve_path(report_json),
        run_manifest_json_path=_resolve_path(run_manifest_json),
    )


def _opcode_name(opcode_id: int) -> str:
    if 0x60 <= opcode_id <= 0x7F:
        return f"PUSH{opcode_id - 0x5F}"
    if 0x80 <= opcode_id <= 0x8F:
        return f"DUP{opcode_id - 0x7F}"
    if 0x90 <= opcode_id <= 0x9F:
        return f"SWAP{opcode_id - 0x8F}"
    if 0xA0 <= opcode_id <= 0xA4:
        return f"LOG{opcode_id - 0xA0}"
    return BASE_OPCODE_NAMES.get(opcode_id, f"UNKNOWN_{opcode_id:02X}")


def normalize_runtime_hex_safely(runtime_hex: Any) -> Tuple[str, Optional[str]]:
    raw_text = "" if runtime_hex is None else str(runtime_hex).strip()
    if raw_text.lower() in MISSING_RUNTIME_TOKENS:
        return "", "missing_runtime_bytecode"

    try:
        normalized = normalize_hex(raw_text)
    except ValueError as exc:
        message = str(exc).lower()
        if "odd length" in message:
            return "", "odd_length_hex"
        if "non-hex" in message:
            return "", "invalid_hex"
        return "", "invalid_runtime_hex"

    if not normalized:
        return "", "missing_runtime_bytecode"
    return normalized, None


def disassemble_runtime_bytecode(normalized_runtime_hex: str) -> List[Dict[str, Any]]:
    code = bytes.fromhex(normalized_runtime_hex)
    instructions: List[Dict[str, Any]] = []
    pc = 0
    while pc < len(code):
        opcode_id = int(code[pc])
        opcode = _opcode_name(opcode_id)
        push_data = ""
        size = 1
        if 0x60 <= opcode_id <= 0x7F:
            push_size = opcode_id - 0x5F
            push_start = pc + 1
            push_end = min(len(code), push_start + push_size)
            push_data = code[push_start:push_end].hex()
            size = 1 + (push_end - push_start)

        instructions.append(
            {
                "pc": int(pc),
                "opcode": opcode,
                "opcode_id": int(opcode_id),
                "push_data": push_data,
                "size": int(size),
            }
        )
        pc += size
    return instructions


def _build_block(block_id: int, block_instructions: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    first = block_instructions[0]
    last = block_instructions[-1]
    end_opcode_id = int(last["opcode_id"])
    return {
        "basic_block_id": int(block_id),
        "start_pc": int(first["pc"]),
        "end_pc": int(last["pc"]),
        "instruction_count": int(len(block_instructions)),
        "starts_with_jumpdest": bool(int(first["opcode_id"]) == JUMPDEST_OPCODE),
        "ends_with_terminal": bool(end_opcode_id in TERMINAL_BLOCK_OPCODES),
        "end_opcode": str(last["opcode"]),
        "end_opcode_id": end_opcode_id,
    }


def segment_basic_blocks(
    instructions: Sequence[Mapping[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not instructions:
        return [], []

    annotated: List[Dict[str, Any]] = [dict(item) for item in instructions]
    blocks: List[Dict[str, Any]] = []
    current_block_id = 0
    block_start = 0

    for idx, instr in enumerate(annotated):
        opcode_id = int(instr["opcode_id"])
        if opcode_id == JUMPDEST_OPCODE and idx > block_start:
            blocks.append(_build_block(current_block_id, annotated[block_start:idx]))
            current_block_id += 1
            block_start = idx

        instr["basic_block_id"] = int(current_block_id)
        if opcode_id in TERMINAL_BLOCK_OPCODES and (idx + 1) < len(annotated):
            blocks.append(_build_block(current_block_id, annotated[block_start : idx + 1]))
            current_block_id += 1
            block_start = idx + 1

    if block_start < len(annotated):
        blocks.append(_build_block(current_block_id, annotated[block_start:]))
    return annotated, blocks


def _as_nullable_bool(value: Any) -> Optional[bool]:
    if pd.isna(value):
        return None
    return bool(value)


def _contract_metadata(row: Mapping[str, Any]) -> Dict[str, Any]:
    split = str(row["split"]).strip() if "split" in row and not pd.isna(row["split"]) else "unknown"
    return {
        "split": split or "unknown",
        "has_cgt": _as_nullable_bool(row.get("has_cgt")),
        "has_dappscan": _as_nullable_bool(row.get("has_dappscan")),
        "is_proxy_like": _as_nullable_bool(row.get("is_proxy_like")),
        "is_stub_like": _as_nullable_bool(row.get("is_stub_like")),
    }


def _append_failure_rows(
    instructions_rows: List[Tuple[Any, ...]],
    blocks_rows: List[Tuple[Any, ...]],
    contract_id: str,
    metadata: Mapping[str, Any],
    failure_mode: str,
) -> None:
    instructions_rows.append(
        (
            contract_id,
            metadata["split"],
            metadata["has_cgt"],
            metadata["has_dappscan"],
            metadata["is_proxy_like"],
            metadata["is_stub_like"],
            False,
            failure_mode,
            None,
            None,
            None,
            "",
            None,
            None,
        )
    )
    blocks_rows.append(
        (
            contract_id,
            metadata["split"],
            metadata["has_cgt"],
            metadata["has_dappscan"],
            metadata["is_proxy_like"],
            metadata["is_stub_like"],
            False,
            failure_mode,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    )


def _finalize_instruction_df(rows: List[Tuple[Any, ...]]) -> pd.DataFrame:
    columns = [
        "fp_runtime_unified",
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
        "size",
        "basic_block_id",
    ]
    df = pd.DataFrame.from_records(rows, columns=columns)
    if df.empty:
        return df

    for col in ["pc", "opcode_id", "size", "basic_block_id"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ["has_cgt", "has_dappscan", "is_proxy_like", "is_stub_like"]:
        df[col] = df[col].astype("boolean")
    df["parse_success"] = df["parse_success"].astype(bool)
    df["failure_mode"] = df["failure_mode"].fillna("")
    df["opcode"] = df["opcode"].fillna("")
    df["push_data"] = df["push_data"].fillna("")
    return df


def _finalize_blocks_df(rows: List[Tuple[Any, ...]]) -> pd.DataFrame:
    columns = [
        "fp_runtime_unified",
        "split",
        "has_cgt",
        "has_dappscan",
        "is_proxy_like",
        "is_stub_like",
        "parse_success",
        "failure_mode",
        "basic_block_id",
        "start_pc",
        "end_pc",
        "instruction_count",
        "starts_with_jumpdest",
        "ends_with_terminal",
        "end_opcode",
        "end_opcode_id",
    ]
    df = pd.DataFrame.from_records(rows, columns=columns)
    if df.empty:
        return df

    for col in ["basic_block_id", "start_pc", "end_pc", "instruction_count", "end_opcode_id"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ["has_cgt", "has_dappscan", "is_proxy_like", "is_stub_like"]:
        df[col] = df[col].astype("boolean")
    for col in ["starts_with_jumpdest", "ends_with_terminal"]:
        df[col] = df[col].astype("boolean")
    df["parse_success"] = df["parse_success"].astype(bool)
    df["failure_mode"] = df["failure_mode"].fillna("")
    df["end_opcode"] = df["end_opcode"].fillna("")
    return df


def _quantiles(values: Iterable[int]) -> Dict[str, Optional[float]]:
    series = pd.Series(list(values), dtype="float64")
    if series.empty:
        return {"median": None, "p95": None}
    return {
        "median": float(series.quantile(0.5)),
        "p95": float(series.quantile(0.95)),
    }


def run_opcode_preprocessing(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    run_started = datetime.now(timezone.utc)
    config = _load_config(config_path.resolve())
    if not config.input_contracts_path.exists():
        raise FileNotFoundError(f"Input contracts parquet not found: {config.input_contracts_path}")

    contracts = pd.read_parquet(config.input_contracts_path).copy()
    required_cols = {CONTRACT_ID_COLUMN, RUNTIME_COLUMN}
    missing_cols = sorted(required_cols - set(contracts.columns))
    if missing_cols:
        raise ValueError(f"Input contracts missing required columns: {missing_cols}")

    contracts[CONTRACT_ID_COLUMN] = contracts[CONTRACT_ID_COLUMN].fillna("").astype(str).str.strip()
    contracts = contracts[contracts[CONTRACT_ID_COLUMN] != ""].drop_duplicates(
        subset=[CONTRACT_ID_COLUMN], keep="first"
    )
    if contracts.empty:
        raise ValueError("No contracts found in input parquet after cleaning contract IDs.")

    instructions_rows: List[Tuple[Any, ...]] = []
    blocks_rows: List[Tuple[Any, ...]] = []
    contract_stats: List[Dict[str, Any]] = []

    for _, row in contracts.iterrows():
        contract_id = str(row[CONTRACT_ID_COLUMN]).strip()
        metadata = _contract_metadata(row)

        normalized_runtime_hex, failure_mode = normalize_runtime_hex_safely(row[RUNTIME_COLUMN])
        if failure_mode:
            _append_failure_rows(instructions_rows, blocks_rows, contract_id, metadata, failure_mode)
            contract_stats.append(
                {
                    "fp_runtime_unified": contract_id,
                    "split": metadata["split"],
                    "parse_success": False,
                    "failure_mode": failure_mode,
                    "opcode_count": 0,
                    "basic_block_count": 0,
                }
            )
            continue

        try:
            instructions = disassemble_runtime_bytecode(normalized_runtime_hex)
            annotated_instructions, blocks = segment_basic_blocks(instructions)
        except Exception:
            _append_failure_rows(instructions_rows, blocks_rows, contract_id, metadata, "disassembly_error")
            contract_stats.append(
                {
                    "fp_runtime_unified": contract_id,
                    "split": metadata["split"],
                    "parse_success": False,
                    "failure_mode": "disassembly_error",
                    "opcode_count": 0,
                    "basic_block_count": 0,
                }
            )
            continue

        for instruction in annotated_instructions:
            instructions_rows.append(
                (
                    contract_id,
                    metadata["split"],
                    metadata["has_cgt"],
                    metadata["has_dappscan"],
                    metadata["is_proxy_like"],
                    metadata["is_stub_like"],
                    True,
                    "",
                    int(instruction["pc"]),
                    str(instruction["opcode"]),
                    int(instruction["opcode_id"]),
                    str(instruction["push_data"]),
                    int(instruction["size"]),
                    int(instruction["basic_block_id"]),
                )
            )

        for block in blocks:
            blocks_rows.append(
                (
                    contract_id,
                    metadata["split"],
                    metadata["has_cgt"],
                    metadata["has_dappscan"],
                    metadata["is_proxy_like"],
                    metadata["is_stub_like"],
                    True,
                    "",
                    int(block["basic_block_id"]),
                    int(block["start_pc"]),
                    int(block["end_pc"]),
                    int(block["instruction_count"]),
                    bool(block["starts_with_jumpdest"]),
                    bool(block["ends_with_terminal"]),
                    str(block["end_opcode"]),
                    int(block["end_opcode_id"]),
                )
            )

        contract_stats.append(
            {
                "fp_runtime_unified": contract_id,
                "split": metadata["split"],
                "parse_success": True,
                "failure_mode": "",
                "opcode_count": int(len(annotated_instructions)),
                "basic_block_count": int(len(blocks)),
            }
        )

    instructions_df = _finalize_instruction_df(instructions_rows)
    blocks_df = _finalize_blocks_df(blocks_rows)
    contracts_df = pd.DataFrame(contract_stats)
    successful_contracts = contracts_df[contracts_df["parse_success"]].copy()
    failed_contracts = contracts_df[~contracts_df["parse_success"]].copy()

    config.instructions_out_path.parent.mkdir(parents=True, exist_ok=True)
    config.basic_blocks_out_path.parent.mkdir(parents=True, exist_ok=True)
    instructions_df.to_parquet(config.instructions_out_path, index=False)
    blocks_df.to_parquet(config.basic_blocks_out_path, index=False)

    opcode_quantiles = _quantiles(successful_contracts["opcode_count"].tolist())
    block_quantiles = _quantiles(successful_contracts["basic_block_count"].tolist())
    failure_counts = failed_contracts["failure_mode"].value_counts() if not failed_contracts.empty else pd.Series(dtype=int)
    top_failure_modes = [
        {"failure_mode": str(mode), "count": int(count)}
        for mode, count in failure_counts.head(10).items()
    ]

    split_stats: Dict[str, Dict[str, int]] = {}
    for split_name, frame in contracts_df.groupby("split"):
        split_stats[str(split_name)] = {
            "contracts_processed": int(len(frame)),
            "contracts_successful": int(frame["parse_success"].sum()),
            "parse_failures": int((~frame["parse_success"]).sum()),
        }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "phase2_task2_opcode_preprocessing",
        "input_contracts_parquet": _rel(config.input_contracts_path),
        "output_instructions_parquet": _rel(config.instructions_out_path),
        "output_basic_blocks_parquet": _rel(config.basic_blocks_out_path),
        "contracts_processed": int(len(contracts_df)),
        "contracts_successful": int(successful_contracts.shape[0]),
        "parse_failures": int(failed_contracts.shape[0]),
        "opcode_count_stats": opcode_quantiles,
        "basic_block_count_stats": block_quantiles,
        "top_parse_failure_modes": top_failure_modes,
        "split_stats": split_stats,
        "rows_written": {
            "instructions_rows": int(len(instructions_df)),
            "basic_blocks_rows": int(len(blocks_df)),
        },
    }
    _write_json(report, config.report_json_path)

    run_manifest = {
        "run_started_utc": run_started.isoformat(),
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "entrypoint": "python -m src.preprocessing.opcodes --config configs/phase2.yaml",
        "config_path": _rel(config_path.resolve()),
        "inputs": {"contracts_parquet": _rel(config.input_contracts_path)},
        "outputs": {
            "instructions_parquet": _rel(config.instructions_out_path),
            "basic_blocks_parquet": _rel(config.basic_blocks_out_path),
            "report_json": _rel(config.report_json_path),
            "run_manifest_json": _rel(config.run_manifest_json_path),
        },
        "summary": {
            "contracts_processed": report["contracts_processed"],
            "contracts_successful": report["contracts_successful"],
            "parse_failures": report["parse_failures"],
        },
    }
    _write_json(run_manifest, config.run_manifest_json_path)
    return report


def _print_report_summary(report: Mapping[str, Any]) -> None:
    print(f"contracts processed: {report['contracts_processed']}")
    print(f"parse failures: {report['parse_failures']}")
    opcode_stats = report.get("opcode_count_stats", {})
    block_stats = report.get("basic_block_count_stats", {})
    print(f"opcode count median/p95: {opcode_stats.get('median')} / {opcode_stats.get('p95')}")
    print(f"basic-block count median/p95: {block_stats.get('median')} / {block_stats.get('p95')}")
    print("top parse failure modes:")
    top_modes = report.get("top_parse_failure_modes", [])
    if not top_modes:
        print("- none")
    else:
        for item in top_modes:
            print(f"- {item['failure_mode']}: {item['count']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Disassemble Phase 2 runtime bytecode into instructions and blocks.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    report = run_opcode_preprocessing(args.config.resolve())
    _print_report_summary(report)


if __name__ == "__main__":
    main()
