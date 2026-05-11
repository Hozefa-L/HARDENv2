from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

CONTRACT_ID_COLUMN = "fp_runtime_unified"


def _as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)


def _as_bool_float(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool).astype(float)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numer = _as_numeric(numerator)
    denom = _as_numeric(denominator)
    result = np.zeros(len(numer), dtype=np.float64)
    valid = denom > 0.0
    result[valid.to_numpy()] = (numer[valid] / denom[valid]).to_numpy(dtype=np.float64)
    return pd.Series(result, index=numer.index, dtype="float64")


def build_graph_level_features(
    graph_index: pd.DataFrame,
    opcode_corpus: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    required_graph_columns = {
        CONTRACT_ID_COLUMN,
        "split",
        "node_count",
        "basic_block_count",
        "cfg_edge_count",
        "dfg_edge_count",
        "edge_count_total",
        "has_cgt",
        "has_dappscan",
        "is_proxy_like",
        "is_stub_like",
        "graph_build_success",
        "cfg_success",
        "dfg_success",
    }
    missing_graph = sorted(required_graph_columns - set(graph_index.columns))
    if missing_graph:
        raise ValueError(f"Graph index missing required columns for graph-level features: {missing_graph}")

    required_opcode_columns = {
        CONTRACT_ID_COLUMN,
        "delegatecall_ratio",
        "call_like_ratio",
        "opcode_token_count",
        "opcode_text_available",
    }
    missing_opcode = sorted(required_opcode_columns - set(opcode_corpus.columns))
    if missing_opcode:
        raise ValueError(f"Opcode corpus missing required columns for graph-level features: {missing_opcode}")

    left = graph_index.copy()
    right = opcode_corpus[list(required_opcode_columns)].copy()
    merged = left.merge(right, on=CONTRACT_ID_COLUMN, how="left", validate="one_to_one")
    if len(merged) != len(graph_index):
        raise ValueError("Graph-level feature merge changed row count unexpectedly.")

    merged["delegatecall_ratio"] = _as_numeric(merged["delegatecall_ratio"])
    merged["call_like_ratio"] = _as_numeric(merged["call_like_ratio"])
    merged["opcode_token_count"] = _as_numeric(merged["opcode_token_count"])

    feature_map = {
        "gf_node_count": _as_numeric(merged["node_count"]),
        "gf_basic_block_count": _as_numeric(merged["basic_block_count"]),
        "gf_cfg_edge_count": _as_numeric(merged["cfg_edge_count"]),
        "gf_dfg_edge_count": _as_numeric(merged["dfg_edge_count"]),
        "gf_edge_count_total": _as_numeric(merged["edge_count_total"]),
        "gf_avg_edges_per_node": _safe_ratio(merged["edge_count_total"], merged["node_count"]),
        "gf_cfg_edge_ratio": _safe_ratio(merged["cfg_edge_count"], merged["edge_count_total"]),
        "gf_dfg_edge_ratio": _safe_ratio(merged["dfg_edge_count"], merged["edge_count_total"]),
        "gf_delegatecall_ratio": _as_numeric(merged["delegatecall_ratio"]),
        "gf_call_like_ratio": _as_numeric(merged["call_like_ratio"]),
        "gf_opcode_token_count": _as_numeric(merged["opcode_token_count"]),
        "gf_has_cgt": _as_bool_float(merged["has_cgt"]),
        "gf_has_dappscan": _as_bool_float(merged["has_dappscan"]),
        "gf_is_proxy_like": _as_bool_float(merged["is_proxy_like"]),
        "gf_is_stub_like": _as_bool_float(merged["is_stub_like"]),
        "gf_graph_build_success": _as_bool_float(merged["graph_build_success"]),
        "gf_cfg_success": _as_bool_float(merged["cfg_success"]),
        "gf_dfg_success": _as_bool_float(merged["dfg_success"]),
        "gf_opcode_text_available": _as_bool_float(merged["opcode_text_available"]),
    }

    graph_feature_columns: List[str] = list(feature_map.keys())
    feature_values = pd.DataFrame(feature_map)
    feature_values = feature_values.astype("float32")

    output = merged[[CONTRACT_ID_COLUMN, "split"]].copy()
    output["graph_feature_available"] = True
    output["graph_structure_available"] = merged["graph_build_success"].fillna(False).astype(bool)
    output = pd.concat([output.reset_index(drop=True), feature_values.reset_index(drop=True)], axis=1)

    summary = {
        "rows_total": int(len(output)),
        "rows_with_graph_structure": int(output["graph_structure_available"].sum()),
        "rows_without_graph_structure": int((~output["graph_structure_available"]).sum()),
        "feature_dim": int(len(graph_feature_columns)),
        "feature_columns": graph_feature_columns,
    }
    return output, summary
