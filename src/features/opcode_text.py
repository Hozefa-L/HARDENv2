import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import torch

CONTRACT_ID_COLUMN = "fp_runtime_unified"
CALL_LIKE_OPCODES = {"CALL", "CALLCODE", "DELEGATECALL", "STATICCALL"}


def _load_opcode_sequence(artifact: Mapping[str, Any]) -> List[str]:
    opcode_sequence = artifact.get("opcode_sequence")
    if isinstance(opcode_sequence, dict):
        raw_opcodes = opcode_sequence.get("opcode")
        if isinstance(raw_opcodes, list):
            return [str(opcode).strip() for opcode in raw_opcodes if str(opcode).strip()]

    node_table = artifact.get("node_table")
    if isinstance(node_table, dict):
        raw_opcodes = node_table.get("opcode")
        if isinstance(raw_opcodes, list):
            return [str(opcode).strip() for opcode in raw_opcodes if str(opcode).strip()]

    return []


def _extract_failure_mode(artifact: Mapping[str, Any]) -> str:
    metadata = artifact.get("build_metadata")
    if not isinstance(metadata, dict):
        return ""
    value = metadata.get("instruction_failure_mode")
    if value is None:
        return ""
    return str(value).strip()


def _as_bool(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return bool(value)


def _normalize_failure_mode(value: Optional[str]) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _merge_failure_mode(
    graph_build_success: bool,
    index_failure_mode: str,
    artifact_failure_mode: str,
) -> str:
    if graph_build_success:
        return ""
    if index_failure_mode:
        return index_failure_mode
    if artifact_failure_mode:
        return artifact_failure_mode
    return "graph_unavailable"


def build_opcode_text_corpus(
    graph_index: pd.DataFrame,
    project_root: Path,
    failure_mode_by_contract: Mapping[str, str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    required = {
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
    }
    missing = sorted(required - set(graph_index.columns))
    if missing:
        raise ValueError(f"Graph index missing required columns for opcode corpus: {missing}")

    rows: List[Dict[str, Any]] = []
    for _, row in graph_index.iterrows():
        contract_id = str(row[CONTRACT_ID_COLUMN]).strip()
        if not contract_id:
            raise ValueError("Encountered empty contract_id while building opcode text corpus.")
        artifact_rel = str(row["artifact_path"]).strip()
        artifact_path = (project_root / artifact_rel).resolve()
        if not artifact_path.exists():
            raise FileNotFoundError(f"Graph artifact not found for contract `{contract_id}`: {artifact_path}")

        artifact = torch.load(artifact_path, map_location="cpu")
        if not isinstance(artifact, dict):
            raise ValueError(f"Graph artifact payload is not a dict for contract `{contract_id}`: {artifact_path}")

        opcodes = _load_opcode_sequence(artifact)
        opcode_text = " ".join(opcodes)
        opcode_count = int(len(opcodes))
        call_like_count = int(sum(1 for opcode in opcodes if opcode in CALL_LIKE_OPCODES))
        delegatecall_count = int(sum(1 for opcode in opcodes if opcode == "DELEGATECALL"))
        delegatecall_ratio = float(delegatecall_count / opcode_count) if opcode_count > 0 else 0.0
        call_like_ratio = float(call_like_count / opcode_count) if opcode_count > 0 else 0.0

        graph_build_success = _as_bool(row["graph_build_success"])
        opcode_text_available = bool(graph_build_success and opcode_count > 0)
        index_failure_mode = _normalize_failure_mode(failure_mode_by_contract.get(contract_id, ""))
        artifact_failure_mode = _extract_failure_mode(artifact)
        unavailable_cause = _merge_failure_mode(
            graph_build_success=graph_build_success,
            index_failure_mode=index_failure_mode,
            artifact_failure_mode=artifact_failure_mode,
        )

        rows.append(
            {
                CONTRACT_ID_COLUMN: contract_id,
                "graph_id": str(row["graph_id"]).strip(),
                "artifact_path": artifact_rel,
                "split": str(row["split"]).strip(),
                "source_group": str(row["source_group"]).strip(),
                "has_cgt": _as_bool(row["has_cgt"]),
                "has_dappscan": _as_bool(row["has_dappscan"]),
                "is_proxy_like": _as_bool(row["is_proxy_like"]),
                "is_stub_like": _as_bool(row["is_stub_like"]),
                "graph_build_success": graph_build_success,
                "instruction_parse_success": _as_bool(row["instruction_parse_success"]),
                "cfg_success": _as_bool(row["cfg_success"]),
                "dfg_success": _as_bool(row["dfg_success"]),
                "opcode_text_available": opcode_text_available,
                "unavailable_cause": unavailable_cause if not opcode_text_available else "",
                "opcode_text": opcode_text,
                "opcode_token_count": opcode_count,
                "call_like_count": call_like_count,
                "call_like_ratio": call_like_ratio,
                "delegatecall_count": delegatecall_count,
                "delegatecall_ratio": delegatecall_ratio,
            }
        )

    corpus = pd.DataFrame(rows)
    corpus = corpus.sort_values(["split", CONTRACT_ID_COLUMN]).reset_index(drop=True)

    unavailable = corpus[~corpus["opcode_text_available"]]
    unavailable_by_cause = unavailable["unavailable_cause"].value_counts().to_dict()
    unavailable_by_cause = {str(key): int(value) for key, value in unavailable_by_cause.items()}

    summary = {
        "rows_total": int(len(corpus)),
        "rows_available": int(corpus["opcode_text_available"].sum()),
        "rows_unavailable": int((~corpus["opcode_text_available"]).sum()),
        "unavailable_by_cause": unavailable_by_cause,
        "opcode_token_count_stats": {
            "median": float(corpus["opcode_token_count"].median()) if not corpus.empty else 0.0,
            "p95": float(corpus["opcode_token_count"].quantile(0.95)) if not corpus.empty else 0.0,
        },
    }
    return corpus, summary


def build_failure_mode_index(instructions: pd.DataFrame) -> Dict[str, str]:
    required = {CONTRACT_ID_COLUMN, "parse_success", "failure_mode"}
    missing = sorted(required - set(instructions.columns))
    if missing:
        raise ValueError(f"Instructions parquet missing required columns: {missing}")

    working = instructions[[CONTRACT_ID_COLUMN, "parse_success", "failure_mode"]].copy()
    working[CONTRACT_ID_COLUMN] = working[CONTRACT_ID_COLUMN].fillna("").astype(str).str.strip()
    working = working[working[CONTRACT_ID_COLUMN] != ""]
    working["parse_success"] = working["parse_success"].fillna(False).astype(bool)
    working["failure_mode"] = working["failure_mode"].fillna("").astype(str).str.strip()

    def first_non_empty(values: Sequence[str]) -> str:
        for value in values:
            text = str(value).strip()
            if text:
                return text
        return ""

    grouped = (
        working.groupby(CONTRACT_ID_COLUMN, sort=False)
        .agg(
            parse_success=("parse_success", "max"),
            failure_mode=("failure_mode", first_non_empty),
        )
        .reset_index()
    )
    grouped["parse_success"] = grouped["parse_success"].astype(bool)

    failures = grouped[~grouped["parse_success"]].copy()
    failure_map = {
        str(row[CONTRACT_ID_COLUMN]): str(row["failure_mode"]).strip() or "instruction_parse_failed"
        for _, row in failures.iterrows()
    }
    return failure_map


def serialize_opcode_token_list(opcodes: Iterable[str]) -> str:
    return json.dumps([str(opcode) for opcode in opcodes], ensure_ascii=True)
