import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DAPPSCAN_ROOT = PROJECT_ROOT / "data/raw/dappscan"
DEFAULT_REPORT_OUT = PROJECT_ROOT / "reports/phase1/runtime_extraction_report.json"
DEFAULT_RECORDS_OUT = PROJECT_ROOT / "checkpoints/runtime_extraction_records.jsonl"

U256_MOD = 1 << 256
U256_MASK = U256_MOD - 1
UNKNOWN_AUDIT_TEAM = "<UNKNOWN>"

OPCODE_NAMES = {
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
    0x20: "SHA3",
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
    0x44: "DIFFICULTY",
    0x45: "GASLIMIT",
    0x46: "CHAINID",
    0x47: "SELFBALANCE",
    0x48: "BASEFEE",
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

STACK_EFFECTS = {
    0x00: (0, 0),
    0x05: (2, 1),
    0x07: (2, 1),
    0x08: (3, 1),
    0x09: (3, 1),
    0x0A: (2, 1),
    0x0B: (2, 1),
    0x12: (2, 1),
    0x13: (2, 1),
    0x1D: (2, 1),
    0x20: (2, 1),
    0x30: (0, 1),
    0x31: (1, 1),
    0x32: (0, 1),
    0x33: (0, 1),
    0x34: (0, 1),
    0x35: (1, 1),
    0x36: (0, 1),
    0x37: (3, 0),
    0x38: (0, 1),
    0x3A: (0, 1),
    0x3B: (1, 1),
    0x3C: (4, 0),
    0x3D: (0, 1),
    0x3E: (3, 0),
    0x3F: (1, 1),
    0x40: (1, 1),
    0x41: (0, 1),
    0x42: (0, 1),
    0x43: (0, 1),
    0x44: (0, 1),
    0x45: (0, 1),
    0x46: (0, 1),
    0x47: (0, 1),
    0x48: (0, 1),
    0x50: (1, 0),
    0x51: (1, 1),
    0x52: (2, 0),
    0x53: (2, 0),
    0x54: (1, 1),
    0x55: (2, 0),
    0x56: (1, 0),
    0x57: (2, 0),
    0x58: (0, 1),
    0x59: (0, 1),
    0x5A: (0, 1),
    0x5B: (0, 0),
    0xA0: (2, 0),
    0xA1: (3, 0),
    0xA2: (4, 0),
    0xA3: (5, 0),
    0xA4: (6, 0),
    0xF0: (3, 1),
    0xF1: (7, 1),
    0xF2: (7, 1),
    0xF4: (6, 1),
    0xF5: (4, 1),
    0xFA: (6, 1),
    0xFD: (2, 0),
    0xFE: (0, 0),
    0xFF: (1, 0),
}


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        return str(path)


def _normalize_hex(hex_str: Any) -> str:
    if hex_str is None:
        return ""
    value = str(hex_str).strip()
    if value.startswith("0x") or value.startswith("0X"):
        value = value[2:]
    return "".join(value.split()).lower()


def _opcode_name(opcode: int) -> str:
    if 0x60 <= opcode <= 0x7F:
        return f"PUSH{opcode - 0x5F}"
    if 0x80 <= opcode <= 0x8F:
        return f"DUP{opcode - 0x7F}"
    if 0x90 <= opcode <= 0x9F:
        return f"SWAP{opcode - 0x8F}"
    if 0xA0 <= opcode <= 0xA4:
        return f"LOG{opcode - 0xA0}"
    return OPCODE_NAMES.get(opcode, f"OP_{opcode:02x}")


def disassemble_initcode(initcode: bytes) -> List[Dict[str, Any]]:
    instructions: List[Dict[str, Any]] = []
    pc = 0
    while pc < len(initcode):
        opcode = initcode[pc]
        mnemonic = _opcode_name(opcode)
        instruction: Dict[str, Any] = {"pc": int(pc), "opcode": int(opcode), "mnemonic": mnemonic}
        if 0x60 <= opcode <= 0x7F:
            push_n = opcode - 0x5F
            data = initcode[pc + 1 : pc + 1 + push_n]
            instruction["push_size"] = int(push_n)
            instruction["push_data"] = int.from_bytes(data, byteorder="big", signed=False)
            instruction["push_data_hex"] = data.hex()
            pc += 1 + push_n
        else:
            pc += 1
        instructions.append(instruction)
    return instructions


def _pop(stack: List[Optional[int]]) -> Optional[int]:
    if not stack:
        return None
    return stack.pop()


def _u256(value: int) -> int:
    return int(value) & U256_MASK


def _eval_binary(opcode: int, first: Optional[int], second: Optional[int]) -> Optional[int]:
    if first is None or second is None:
        return None
    a = int(first)
    b = int(second)
    if opcode == 0x01:  # ADD
        return _u256(b + a)
    if opcode == 0x02:  # MUL
        return _u256(b * a)
    if opcode == 0x03:  # SUB
        return _u256(b - a)
    if opcode == 0x04:  # DIV
        return 0 if a == 0 else _u256(b // a)
    if opcode == 0x06:  # MOD
        return 0 if a == 0 else _u256(b % a)
    if opcode == 0x10:  # LT
        return 1 if b < a else 0
    if opcode == 0x11:  # GT
        return 1 if b > a else 0
    if opcode == 0x14:  # EQ
        return 1 if b == a else 0
    if opcode == 0x16:  # AND
        return _u256(b & a)
    if opcode == 0x17:  # OR
        return _u256(b | a)
    if opcode == 0x18:  # XOR
        return _u256(b ^ a)
    if opcode == 0x1A:  # BYTE
        return 0 if a >= 32 else (b >> (8 * (31 - a))) & 0xFF
    if opcode == 0x1B:  # SHL
        return 0 if a >= 256 else _u256(b << a)
    if opcode == 0x1C:  # SHR
        return 0 if a >= 256 else _u256(b >> a)
    return None


def _eval_unary(opcode: int, value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    v = int(value)
    if opcode == 0x15:  # ISZERO
        return 1 if v == 0 else 0
    if opcode == 0x19:  # NOT
        return _u256(~v)
    return None


def _analyze_candidates(instructions: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int, List[Dict[str, Any]]]:
    stack: List[Optional[int]] = []
    codecopy_events: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []
    return_events = 0

    for ins in instructions:
        opcode = ins["opcode"]
        pc = int(ins["pc"])

        if 0x60 <= opcode <= 0x7F:
            stack.append(ins.get("push_data"))
            continue

        if 0x80 <= opcode <= 0x8F:
            dup_n = opcode - 0x7F
            stack.append(stack[-dup_n] if len(stack) >= dup_n else None)
            continue

        if 0x90 <= opcode <= 0x9F:
            swap_n = opcode - 0x8F
            if len(stack) > swap_n:
                stack[-1], stack[-1 - swap_n] = stack[-1 - swap_n], stack[-1]
            elif stack:
                stack[-1] = None
            continue

        if opcode in {0x01, 0x02, 0x03, 0x04, 0x06, 0x10, 0x11, 0x14, 0x16, 0x17, 0x18, 0x1A, 0x1B, 0x1C}:
            first = _pop(stack)
            second = _pop(stack)
            stack.append(_eval_binary(opcode, first, second))
            continue

        if opcode in {0x15, 0x19}:
            stack.append(_eval_unary(opcode, _pop(stack)))
            continue

        if opcode == 0x39:  # CODECOPY
            dest_offset = _pop(stack)
            code_offset = _pop(stack)
            length = _pop(stack)
            codecopy_events.append(
                {
                    "codecopy_pc": int(pc),
                    "dest_offset": dest_offset,
                    "code_offset": code_offset,
                    "length": length,
                }
            )
            continue

        if opcode == 0xF3:  # RETURN
            return_events += 1
            return_offset = _pop(stack)
            return_length = _pop(stack)
            if return_offset is None or return_length is None:
                continue
            for event in reversed(codecopy_events):
                if event["dest_offset"] != return_offset:
                    continue
                if event["length"] != return_length:
                    continue
                if event["code_offset"] is None:
                    continue
                candidates.append(
                    {
                        "codecopy_pc": int(event["codecopy_pc"]),
                        "return_pc": int(pc),
                        "dest_offset": int(event["dest_offset"]),
                        "runtime_offset": int(event["code_offset"]),
                        "runtime_length": int(event["length"]),
                        "return_offset": int(return_offset),
                        "return_length": int(return_length),
                    }
                )
                break
            continue

        pops, pushes = STACK_EFFECTS.get(opcode, (0, 0))
        for _ in range(pops):
            _pop(stack)
        for _ in range(pushes):
            stack.append(None)

    return codecopy_events, return_events, candidates


def extract_runtime_from_initcode(initcode_hex: str, min_runtime_len: int = 100) -> Dict[str, Any]:
    normalized = _normalize_hex(initcode_hex)
    if not normalized:
        return {
            "success": False,
            "failure_mode": "empty_initcode",
            "initcode_length": 0,
            "instruction_count": 0,
            "codecopy_event_count": 0,
            "return_event_count": 0,
            "candidate_count": 0,
            "selected_candidate": None,
        }
    if len(normalized) % 2 != 0:
        return {
            "success": False,
            "failure_mode": "invalid_hex_odd_length",
            "initcode_length": len(normalized) // 2,
            "instruction_count": 0,
            "codecopy_event_count": 0,
            "return_event_count": 0,
            "candidate_count": 0,
            "selected_candidate": None,
        }
    try:
        initcode_bytes = bytes.fromhex(normalized)
    except ValueError:
        return {
            "success": False,
            "failure_mode": "invalid_hex",
            "initcode_length": len(normalized) // 2,
            "instruction_count": 0,
            "codecopy_event_count": 0,
            "return_event_count": 0,
            "candidate_count": 0,
            "selected_candidate": None,
        }

    instructions = disassemble_initcode(initcode_bytes)
    codecopy_events, return_events, candidates = _analyze_candidates(instructions)
    selected_candidate = None
    if candidates:
        selected_candidate = max(candidates, key=lambda c: (c["return_pc"], c["codecopy_pc"]))

    base = {
        "success": False,
        "initcode_length": int(len(initcode_bytes)),
        "instruction_count": int(len(instructions)),
        "codecopy_event_count": int(len(codecopy_events)),
        "return_event_count": int(return_events),
        "candidate_count": int(len(candidates)),
        "selected_candidate": selected_candidate,
    }
    if selected_candidate is None:
        base["failure_mode"] = "no_return_linked_codecopy"
        return base

    runtime_offset = int(selected_candidate["runtime_offset"])
    runtime_length = int(selected_candidate["runtime_length"])
    if runtime_length < int(min_runtime_len):
        base["failure_mode"] = "runtime_too_short"
        base["runtime_offset"] = runtime_offset
        base["runtime_length"] = runtime_length
        return base
    if runtime_offset < 0 or runtime_length < 0:
        base["failure_mode"] = "negative_runtime_bounds"
        base["runtime_offset"] = runtime_offset
        base["runtime_length"] = runtime_length
        return base
    if runtime_offset + runtime_length > len(initcode_bytes):
        base["failure_mode"] = "runtime_out_of_bounds"
        base["runtime_offset"] = runtime_offset
        base["runtime_length"] = runtime_length
        return base

    runtime_bytes = initcode_bytes[runtime_offset : runtime_offset + runtime_length]
    base.update(
        {
            "success": True,
            "failure_mode": None,
            "runtime_offset": runtime_offset,
            "runtime_length": runtime_length,
            "runtime_hex": runtime_bytes.hex(),
        }
    )
    return base


def _find_metadata_workbook(dappscan_root: Path) -> Optional[Path]:
    for candidate in [
        dappscan_root / "DApp_list.xlsx",
        dappscan_root / "DApp_list.xls",
        dappscan_root / "Audit_and_Repository_link.xlsx",
    ]:
        if candidate.exists():
            return candidate
    return next(iter(dappscan_root.glob("**/*DApp*list*.xlsx")), None)


def _load_audit_mapping(dappscan_root: Path) -> Tuple[Dict[str, str], Dict[str, Any]]:
    workbook = _find_metadata_workbook(dappscan_root)
    if workbook is None:
        return {}, {"available": False, "used_file": None, "row_count": 0}

    try:
        xls = pd.ExcelFile(workbook)
        sheet_name = xls.sheet_names[0]
        df = pd.read_excel(workbook, sheet_name=sheet_name)
    except Exception as exc:
        return {}, {
            "available": False,
            "used_file": _rel(workbook),
            "row_count": 0,
            "parse_error": str(exc),
        }

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}

    file_col = None
    for target in ["file name", "filename", "file_name", "dapp", "dapp name"]:
        if target in lower_map:
            file_col = lower_map[target]
            break

    audit_col = None
    for c in df.columns:
        if "audit" in c.lower() and "company" in c.lower():
            audit_col = c
            break
    if audit_col is None:
        for c in df.columns:
            if "audit" in c.lower():
                audit_col = c
                break

    mapping: Dict[str, str] = {}
    if file_col is not None:
        for _, row in df.iterrows():
            dapp = str(row.get(file_col, "")).strip()
            if not dapp:
                continue
            audit_team = ""
            if audit_col is not None:
                audit_team = str(row.get(audit_col, "")).strip()
            mapping[dapp.lower()] = audit_team if audit_team else UNKNOWN_AUDIT_TEAM

    metadata = {
        "available": True,
        "used_file": _rel(workbook),
        "sheet_name": sheet_name,
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "mapping_entries": int(len(mapping)),
    }
    return mapping, metadata


def _runtime_length_stats(lengths: List[int]) -> Dict[str, Any]:
    if not lengths:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    arr = np.array(lengths, dtype=np.int64)
    return {
        "count": int(len(arr)),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(np.round(arr.mean(), 4)),
        "median": float(np.round(float(np.median(arr)), 4)),
        "p95": float(np.round(float(np.percentile(arr, 95)), 4)),
        "p99": float(np.round(float(np.percentile(arr, 99)), 4)),
    }


def _new_group() -> Dict[str, Any]:
    return {"processed": 0, "success": 0, "runtime_lengths": [], "failure_modes": Counter()}


def _update_group(
    groups: Dict[str, Dict[str, Any]],
    group_key: str,
    success: bool,
    runtime_length: Optional[int],
    failure_mode: Optional[str],
) -> None:
    if group_key not in groups:
        groups[group_key] = _new_group()
    group = groups[group_key]
    group["processed"] += 1
    if success:
        group["success"] += 1
        if runtime_length is not None:
            group["runtime_lengths"].append(int(runtime_length))
    elif failure_mode:
        group["failure_modes"][failure_mode] += 1


def _group_summary(groups: Dict[str, Dict[str, Any]], key_name: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key, stats in groups.items():
        processed = int(stats["processed"])
        success = int(stats["success"])
        failure = int(processed - success)
        top_failure = None
        if stats["failure_modes"]:
            mode, count = stats["failure_modes"].most_common(1)[0]
            top_failure = {"failure_mode": mode, "count": int(count)}
        rows.append(
            {
                key_name: key,
                "processed": processed,
                "success": success,
                "failure": failure,
                "success_rate": round(success / processed, 6) if processed else 0.0,
                "runtime_length_stats": _runtime_length_stats(stats["runtime_lengths"]),
                "top_failure_mode": top_failure,
            }
        )
    rows.sort(key=lambda item: (-item["processed"], str(item[key_name])))
    return rows


def _top_failure_modes(counter: Counter, processed_count: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    total_failures = int(sum(counter.values()))
    for failure_mode, count in counter.most_common(20):
        rows.append(
            {
                "failure_mode": str(failure_mode),
                "count": int(count),
                "rate_over_processed": round(count / processed_count, 6) if processed_count else 0.0,
                "rate_over_failures": round(count / total_failures, 6) if total_failures else 0.0,
            }
        )
    return rows


def run_runtime_extraction(
    dappscan_root: Path,
    report_out: Path,
    records_out: Path,
    min_runtime_len: int = 100,
    max_failure_samples: int = 30,
) -> Dict[str, Any]:
    bytecode_root = dappscan_root / "DAppSCAN-bytecode" / "bytecode"
    dappscan_bytecode_root = dappscan_root / "DAppSCAN-bytecode"

    audit_mapping, audit_meta = _load_audit_mapping(dappscan_root)

    inventory: Dict[str, int] = {
        "json_files_scanned": 0,
        "json_parse_errors": 0,
        "json_without_contracts_dict": 0,
        "contract_entries_total": 0,
        "non_dict_contract_entries": 0,
        "bin_files_scanned": 0,
        "skipped_empty_initcode": 0,
        "initcode_entries_processed": 0,
    }

    success_count = 0
    runtime_lengths: List[int] = []
    failure_counter: Counter = Counter()
    dapp_groups: Dict[str, Dict[str, Any]] = {}
    audit_groups: Dict[str, Dict[str, Any]] = {}
    failure_samples: List[Dict[str, Any]] = []

    records_out.parent.mkdir(parents=True, exist_ok=True)
    with records_out.open("w", encoding="utf-8") as records_fp:
        for json_path in bytecode_root.rglob("*.json"):
            inventory["json_files_scanned"] += 1
            try:
                rel = json_path.relative_to(bytecode_root)
                dapp_name = rel.parts[0] if rel.parts else "<ROOT>"
            except Exception:
                dapp_name = json_path.parent.name

            try:
                obj = json.loads(json_path.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                inventory["json_parse_errors"] += 1
                continue

            contracts = obj.get("contracts", {})
            if not isinstance(contracts, dict):
                inventory["json_without_contracts_dict"] += 1
                continue

            for contract_key, contract_obj in contracts.items():
                inventory["contract_entries_total"] += 1
                if not isinstance(contract_obj, dict):
                    inventory["non_dict_contract_entries"] += 1
                    continue
                initcode_hex = contract_obj.get("bin", "")
                if not _normalize_hex(initcode_hex):
                    inventory["skipped_empty_initcode"] += 1
                    continue

                inventory["initcode_entries_processed"] += 1
                audit_team = audit_mapping.get(dapp_name.lower(), UNKNOWN_AUDIT_TEAM)
                extraction = extract_runtime_from_initcode(
                    initcode_hex=str(initcode_hex), min_runtime_len=min_runtime_len
                )
                success = bool(extraction["success"])
                failure_mode = extraction.get("failure_mode")
                runtime_length = extraction.get("runtime_length")

                if success:
                    success_count += 1
                    runtime_lengths.append(int(runtime_length))
                elif failure_mode:
                    failure_counter[str(failure_mode)] += 1

                _update_group(
                    dapp_groups,
                    dapp_name,
                    success=success,
                    runtime_length=runtime_length if success else None,
                    failure_mode=str(failure_mode) if failure_mode else None,
                )
                _update_group(
                    audit_groups,
                    audit_team,
                    success=success,
                    runtime_length=runtime_length if success else None,
                    failure_mode=str(failure_mode) if failure_mode else None,
                )

                selected = extraction.get("selected_candidate") or {}
                record = {
                    "source_kind": "json_contract_bin",
                    "source_path": _rel(json_path),
                    "dapp": dapp_name,
                    "audit_team": audit_team,
                    "contract_key": str(contract_key),
                    "initcode_length": int(extraction.get("initcode_length", 0)),
                    "instruction_count": int(extraction.get("instruction_count", 0)),
                    "codecopy_event_count": int(extraction.get("codecopy_event_count", 0)),
                    "return_event_count": int(extraction.get("return_event_count", 0)),
                    "candidate_count": int(extraction.get("candidate_count", 0)),
                    "selected_codecopy_pc": selected.get("codecopy_pc"),
                    "selected_return_pc": selected.get("return_pc"),
                    "runtime_offset": extraction.get("runtime_offset"),
                    "runtime_length": extraction.get("runtime_length"),
                    "success": success,
                    "failure_mode": failure_mode,
                }
                if success and extraction.get("runtime_hex"):
                    runtime_hex = str(extraction["runtime_hex"])
                    record["runtime_sha256"] = hashlib.sha256(
                        bytes.fromhex(runtime_hex)
                    ).hexdigest()
                records_fp.write(json.dumps(record) + "\n")

                if (not success) and len(failure_samples) < max_failure_samples:
                    failure_samples.append(
                        {
                            "dapp": dapp_name,
                            "audit_team": audit_team,
                            "source_path": _rel(json_path),
                            "contract_key": str(contract_key),
                            "failure_mode": failure_mode,
                            "candidate_count": int(extraction.get("candidate_count", 0)),
                        }
                    )

        for bin_path in dappscan_bytecode_root.rglob("*.bin"):
            inventory["bin_files_scanned"] += 1
            try:
                rel_bin = bin_path.relative_to(bytecode_root)
                dapp_name = rel_bin.parts[0] if rel_bin.parts else bin_path.parent.name
            except Exception:
                dapp_name = bin_path.parent.name
            initcode_hex = bin_path.read_text(encoding="utf-8", errors="ignore")
            if not _normalize_hex(initcode_hex):
                inventory["skipped_empty_initcode"] += 1
                continue

            inventory["initcode_entries_processed"] += 1
            audit_team = audit_mapping.get(dapp_name.lower(), UNKNOWN_AUDIT_TEAM)
            extraction = extract_runtime_from_initcode(
                initcode_hex=str(initcode_hex), min_runtime_len=min_runtime_len
            )
            success = bool(extraction["success"])
            failure_mode = extraction.get("failure_mode")
            runtime_length = extraction.get("runtime_length")

            if success:
                success_count += 1
                runtime_lengths.append(int(runtime_length))
            elif failure_mode:
                failure_counter[str(failure_mode)] += 1

            _update_group(
                dapp_groups,
                dapp_name,
                success=success,
                runtime_length=runtime_length if success else None,
                failure_mode=str(failure_mode) if failure_mode else None,
            )
            _update_group(
                audit_groups,
                audit_team,
                success=success,
                runtime_length=runtime_length if success else None,
                failure_mode=str(failure_mode) if failure_mode else None,
            )

            selected = extraction.get("selected_candidate") or {}
            record = {
                "source_kind": "bin_file",
                "source_path": _rel(bin_path),
                "dapp": dapp_name,
                "audit_team": audit_team,
                "contract_key": None,
                "initcode_length": int(extraction.get("initcode_length", 0)),
                "instruction_count": int(extraction.get("instruction_count", 0)),
                "codecopy_event_count": int(extraction.get("codecopy_event_count", 0)),
                "return_event_count": int(extraction.get("return_event_count", 0)),
                "candidate_count": int(extraction.get("candidate_count", 0)),
                "selected_codecopy_pc": selected.get("codecopy_pc"),
                "selected_return_pc": selected.get("return_pc"),
                "runtime_offset": extraction.get("runtime_offset"),
                "runtime_length": extraction.get("runtime_length"),
                "success": success,
                "failure_mode": failure_mode,
            }
            if success and extraction.get("runtime_hex"):
                runtime_hex = str(extraction["runtime_hex"])
                record["runtime_sha256"] = hashlib.sha256(bytes.fromhex(runtime_hex)).hexdigest()
            records_fp.write(json.dumps(record) + "\n")

            if (not success) and len(failure_samples) < max_failure_samples:
                failure_samples.append(
                    {
                        "dapp": dapp_name,
                        "audit_team": audit_team,
                        "source_path": _rel(bin_path),
                        "contract_key": None,
                        "failure_mode": failure_mode,
                        "candidate_count": int(extraction.get("candidate_count", 0)),
                    }
                )

    processed = int(inventory["initcode_entries_processed"])
    failures = int(processed - success_count)
    by_dapp = _group_summary(dapp_groups, "dapp")
    by_audit = _group_summary(audit_groups, "audit_team")
    by_audit_known = [row for row in by_audit if row["audit_team"] != UNKNOWN_AUDIT_TEAM]

    report: Dict[str, Any] = {
        "dataset": "DAppSCAN-bytecode",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "paths": {
            "dappscan_root": _rel(dappscan_root),
            "bytecode_root": _rel(bytecode_root),
        },
        "parameters": {
            "min_runtime_len": int(min_runtime_len),
            "candidate_selection": "latest return-linked CODECOPY nearest initcode end",
        },
        "metadata": {
            "audit_mapping": audit_meta,
            "unknown_audit_team_label": UNKNOWN_AUDIT_TEAM,
        },
        "inventory": inventory,
        "overall": {
            "processed_initcodes": processed,
            "success_count": int(success_count),
            "failure_count": failures,
            "success_rate": round(success_count / processed, 6) if processed else 0.0,
        },
        "runtime_length_stats": _runtime_length_stats(runtime_lengths),
        "by_dapp": by_dapp,
        "by_audit_team": by_audit,
        "by_audit_team_known_only": by_audit_known,
        "top_failure_modes": _top_failure_modes(failure_counter, processed_count=processed),
        "failure_samples": failure_samples,
        "outputs": {
            "records_jsonl": _rel(records_out),
            "report_json": _rel(report_out),
        },
    }

    report_out.parent.mkdir(parents=True, exist_ok=True)
    with report_out.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract runtime bytecode from creation/initcode using CODECOPY/RETURN analysis."
    )
    parser.add_argument("--dappscan-root", type=Path, default=DEFAULT_DAPPSCAN_ROOT)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--records-out", type=Path, default=DEFAULT_RECORDS_OUT)
    parser.add_argument("--min-runtime-len", type=int, default=100)
    parser.add_argument("--max-failure-samples", type=int, default=30)
    args = parser.parse_args()

    report = run_runtime_extraction(
        dappscan_root=args.dappscan_root,
        report_out=args.report_out,
        records_out=args.records_out,
        min_runtime_len=args.min_runtime_len,
        max_failure_samples=args.max_failure_samples,
    )
    print(f"Processed initcodes: {report['overall']['processed_initcodes']}")
    print(f"Success rate: {report['overall']['success_rate']}")
    print(f"Report: {args.report_out}")
    print(f"Records: {args.records_out}")


if __name__ == "__main__":
    main()
