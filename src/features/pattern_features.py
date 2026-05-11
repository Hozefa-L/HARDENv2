"""Expert vulnerability-indicative pattern features from opcode text.

Extracts ~20 features based on known vulnerability signatures in EVM bytecode.
Each feature is a count, ratio, or boolean derived from opcode sequences.
"""

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OPCODE_CORPUS_PATH = PROJECT_ROOT / "data" / "features" / "main_benchmark" / "opcode_text_corpus.parquet"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "features" / "main_benchmark" / "pattern_features.parquet"


def _count_opcode(tokens: List[str], opcode: str) -> int:
    return tokens.count(opcode)


def _has_opcode(tokens: List[str], opcode: str) -> int:
    return int(opcode in tokens)


def _count_opcodes(tokens: List[str], opcodes: set) -> int:
    return sum(1 for t in tokens if t in opcodes)


def _extract_patterns(opcode_text: str) -> Dict[str, float]:
    """Extract all pattern features from a single contract's opcode text."""
    tokens = opcode_text.split()
    n = len(tokens) if tokens else 1  # avoid division by zero

    features: Dict[str, float] = {}

    # --- SWC-107 Reentrancy indicators ---
    features["pat_call_count"] = _count_opcode(tokens, "CALL")
    features["pat_callcode_count"] = _count_opcode(tokens, "CALLCODE")
    features["pat_delegatecall_count"] = _count_opcode(tokens, "DELEGATECALL")
    features["pat_staticcall_count"] = _count_opcode(tokens, "STATICCALL")
    features["pat_external_call_total"] = (
        features["pat_call_count"] + features["pat_callcode_count"]
        + features["pat_delegatecall_count"] + features["pat_staticcall_count"]
    )
    features["pat_external_call_ratio"] = features["pat_external_call_total"] / n

    # CALL followed by SSTORE pattern (state change after external call = reentrancy risk)
    text = " ".join(tokens)
    features["pat_call_then_sstore"] = len(re.findall(r"CALL\b.*?\bSSTORE", text[:50000]))

    # --- SWC-101 Integer overflow indicators ---
    features["pat_add_count"] = _count_opcode(tokens, "ADD")
    features["pat_mul_count"] = _count_opcode(tokens, "MUL")
    features["pat_sub_count"] = _count_opcode(tokens, "SUB")
    features["pat_div_count"] = _count_opcode(tokens, "DIV")
    features["pat_arithmetic_total"] = (
        features["pat_add_count"] + features["pat_mul_count"]
        + features["pat_sub_count"] + features["pat_div_count"]
    )
    features["pat_arithmetic_ratio"] = features["pat_arithmetic_total"] / n
    # Overflow check: ADD followed by LT (or GT for underflow)
    features["pat_add_then_lt"] = len(re.findall(r"ADD\b[^S]*?\bLT", text[:50000]))

    # --- SWC-114 TX.origin authentication ---
    features["pat_origin_count"] = _count_opcode(tokens, "ORIGIN")
    features["pat_has_origin"] = _has_opcode(tokens, "ORIGIN")
    # ORIGIN followed by EQ (tx.origin == msg.sender check)
    features["pat_origin_eq"] = len(re.findall(r"ORIGIN\b.*?\bEQ", text[:20000]))

    # --- SWC-115 Authorization via tx.origin ---
    # Same as 114 but captured by origin presence + caller absence
    features["pat_caller_count"] = _count_opcode(tokens, "CALLER")
    features["pat_origin_without_caller"] = int(
        features["pat_has_origin"] == 1 and features["pat_caller_count"] == 0
    )

    # --- SWC-104 Unchecked call return value ---
    features["pat_callvalue_count"] = _count_opcode(tokens, "CALLVALUE")
    # POP right after CALL (discarding return value)
    features["pat_call_then_pop"] = len(re.findall(r"\bCALL POP\b", text))

    # --- SWC-120 Weak PRNG ---
    features["pat_blockhash_count"] = _count_opcode(tokens, "BLOCKHASH")
    features["pat_timestamp_count"] = _count_opcode(tokens, "TIMESTAMP")
    features["pat_coinbase_count"] = _count_opcode(tokens, "COINBASE")
    features["pat_number_count"] = _count_opcode(tokens, "NUMBER")
    features["pat_randomness_source_total"] = (
        features["pat_blockhash_count"] + features["pat_timestamp_count"]
        + features["pat_coinbase_count"] + features["pat_number_count"]
    )

    # --- SWC-113 DoS with gas limit ---
    features["pat_gas_count"] = _count_opcode(tokens, "GAS")
    features["pat_gaslimit_count"] = _count_opcode(tokens, "GASLIMIT")

    # --- SWC-128 DoS with block gas limit ---
    # Loops with external calls (JUMPI + CALL patterns)
    features["pat_jumpi_count"] = _count_opcode(tokens, "JUMPI")
    features["pat_jump_count"] = _count_opcode(tokens, "JUMP")
    features["pat_loop_density"] = features["pat_jumpi_count"] / n

    # --- SWC-132 Unexpected Ether (balance check) ---
    features["pat_balance_count"] = _count_opcode(tokens, "BALANCE")
    features["pat_selfbalance_count"] = _count_opcode(tokens, "SELFBALANCE")

    # --- SWC-135 Code size check ---
    features["pat_extcodesize_count"] = _count_opcode(tokens, "EXTCODESIZE")
    features["pat_extcodecopy_count"] = _count_opcode(tokens, "EXTCODECOPY")
    features["pat_extcodehash_count"] = _count_opcode(tokens, "EXTCODEHASH")
    features["pat_codesize_count"] = _count_opcode(tokens, "CODESIZE")

    # --- SWC-103 Floating pragma (bytecode proxy: metadata patterns) ---
    features["pat_invalid_count"] = _count_opcode(tokens, "INVALID")

    # --- General complexity features ---
    features["pat_selfdestruct_count"] = _count_opcode(tokens, "SELFDESTRUCT")
    features["pat_create_count"] = _count_opcode(tokens, "CREATE")
    features["pat_create2_count"] = _count_opcode(tokens, "CREATE2")
    features["pat_sstore_count"] = _count_opcode(tokens, "SSTORE")
    features["pat_sload_count"] = _count_opcode(tokens, "SLOAD")
    features["pat_storage_ratio"] = (features["pat_sstore_count"] + features["pat_sload_count"]) / n
    features["pat_keccak256_count"] = _count_opcode(tokens, "KECCAK256")
    features["pat_log_total"] = _count_opcodes(tokens, {"LOG0", "LOG1", "LOG2", "LOG3", "LOG4"})
    features["pat_unique_opcodes"] = len(set(tokens))
    features["pat_total_opcodes"] = len(tokens)

    return features


def build_pattern_features(
    corpus_path: Path = DEFAULT_OPCODE_CORPUS_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Build expert pattern features for all contracts.

    Returns:
        DataFrame with columns: fp_runtime_unified, split, pat_* features
    """
    logger.info("Loading opcode text corpus from %s", corpus_path)
    corpus = pd.read_parquet(corpus_path)

    required_cols = {"fp_runtime_unified", "split", "opcode_text"}
    missing = required_cols - set(corpus.columns)
    if missing:
        raise ValueError(f"Opcode corpus missing columns: {missing}")

    corpus["fp_runtime_unified"] = corpus["fp_runtime_unified"].astype(str).str.strip()
    corpus["opcode_text"] = corpus["opcode_text"].fillna("").astype(str)

    logger.info("Extracting pattern features for %d contracts", len(corpus))
    records = []
    for idx, row in corpus.iterrows():
        features = _extract_patterns(row["opcode_text"])
        features["fp_runtime_unified"] = row["fp_runtime_unified"]
        features["split"] = row["split"]
        records.append(features)

    result = pd.DataFrame(records)

    # Reorder: id columns first, then features
    id_cols = ["fp_runtime_unified", "split"]
    feat_cols = [c for c in result.columns if c not in id_cols]
    result = result[id_cols + sorted(feat_cols)]

    pat_count = len([c for c in result.columns if c.startswith("pat_")])
    logger.info("Extracted %d pattern features for %d contracts", pat_count, len(result))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    logger.info("Saved pattern features to %s", output_path)

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Build expert pattern features from opcode text")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_OPCODE_CORPUS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    build_pattern_features(corpus_path=args.corpus, output_path=args.output)
