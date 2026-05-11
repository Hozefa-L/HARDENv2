from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

CONTRACT_ID_COLUMN = "fp_runtime_unified"
VALID_POOLING = {"cls", "mean"}


@dataclass(frozen=True)
class CodeBertExtractionConfig:
    model_name: str
    pooling: str
    max_length: int
    batch_size: int
    device: str
    local_files_only: bool
    sliding_window: bool = False
    stride: int = 256


def _resolve_device(configured: str) -> torch.device:
    requested = str(configured).strip().lower()
    if requested == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError("`codebert.device` must be either `cpu` or `cuda`.")


def _load_transformer_bundle(
    config: CodeBertExtractionConfig,
) -> Tuple[Any, torch.nn.Module, int]:
    try:
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        raise ImportError("Failed to import transformers for CodeBERT feature extraction.") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        local_files_only=config.local_files_only,
    )
    model = AutoModel.from_pretrained(
        config.model_name,
        local_files_only=config.local_files_only,
    )
    hidden_size = int(model.config.hidden_size)
    return tokenizer, model, hidden_size


def _pool_hidden_states(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling == "cls":
        return last_hidden_state[:, 0, :]
    if pooling == "mean":
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return summed / denom
    raise ValueError(f"Unsupported pooling strategy: {pooling}")


def _token_lengths(tokenizer: Any, texts: Sequence[str]) -> List[int]:
    encoded = tokenizer(
        list(texts),
        add_special_tokens=True,
        truncation=False,
        verbose=False,
    )
    token_ids = encoded.get("input_ids")
    if token_ids is None:
        raise ValueError("Tokenizer output missing `input_ids` while estimating token lengths.")
    return [int(len(ids)) for ids in token_ids]


def build_codebert_feature_columns(hidden_size: int) -> List[str]:
    return [f"cb_{idx:03d}" for idx in range(hidden_size)]


def _create_sliding_windows(
    token_ids: List[int],
    max_length: int,
    stride: int,
    cls_id: int,
    sep_id: int,
) -> List[List[int]]:
    """Split a full token sequence into overlapping windows.

    Each window is: [CLS] + content_chunk + [SEP], padded to max_length.
    Content size per window = max_length - 2 (for CLS/SEP).
    Stride applies to content tokens only.
    """
    # Strip existing CLS/SEP if present
    content = [t for t in token_ids if t not in (cls_id, sep_id)]
    content_max = max_length - 2
    if content_max <= 0:
        return [[cls_id, sep_id] + [0] * max(0, max_length - 2)]

    windows: List[List[int]] = []
    effective_stride = min(stride, content_max)
    pos = 0
    while pos < len(content):
        chunk = content[pos : pos + content_max]
        window = [cls_id] + chunk + [sep_id]
        # Pad to max_length
        if len(window) < max_length:
            window = window + [0] * (max_length - len(window))
        windows.append(window[:max_length])
        pos += effective_stride
        if pos >= len(content) and len(windows) == 0:
            break
    if not windows:
        windows = [[cls_id, sep_id] + [0] * max(0, max_length - 2)]
    return windows


def _extract_single_pass(
    tokenizer: Any,
    model: torch.nn.Module,
    batch_texts: List[str],
    config: CodeBertExtractionConfig,
    device: torch.device,
    hidden_size: int,
) -> np.ndarray:
    """Standard single-pass extraction (truncate at max_length)."""
    encoded = tokenizer(
        batch_texts,
        truncation=True,
        max_length=config.max_length,
        padding="max_length",
        return_tensors="pt",
        verbose=False,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = _pool_hidden_states(outputs.last_hidden_state, attention_mask, config.pooling)
        return pooled.detach().cpu().numpy().astype(np.float32)


def _extract_sliding_window(
    tokenizer: Any,
    model: torch.nn.Module,
    batch_texts: List[str],
    original_lengths: List[int],
    config: CodeBertExtractionConfig,
    device: torch.device,
    hidden_size: int,
) -> Tuple[np.ndarray, List[int]]:
    """Sliding window extraction: windows for long sequences, single-pass for short ones.

    Returns (feature_matrix [batch, hidden], num_windows_per_sample).
    """
    cls_id = tokenizer.cls_token_id if hasattr(tokenizer, "cls_token_id") and tokenizer.cls_token_id is not None else 101
    sep_id = tokenizer.sep_token_id if hasattr(tokenizer, "sep_token_id") and tokenizer.sep_token_id is not None else 102

    # Tokenize without truncation to get full token IDs
    full_encoded = tokenizer(
        batch_texts,
        add_special_tokens=True,
        truncation=False,
        verbose=False,
    )

    all_windows: List[List[int]] = []
    sample_window_counts: List[int] = []
    sample_indices: List[int] = []

    for i, (full_ids, orig_len) in enumerate(zip(full_encoded["input_ids"], original_lengths)):
        if orig_len <= config.max_length:
            # Short sequence — single window with padding
            padded = list(full_ids[:config.max_length])
            if len(padded) < config.max_length:
                padded = padded + [0] * (config.max_length - len(padded))
            all_windows.append(padded)
            sample_window_counts.append(1)
        else:
            # Long sequence — create overlapping windows
            windows = _create_sliding_windows(
                full_ids, config.max_length, config.stride, cls_id, sep_id
            )
            all_windows.extend(windows)
            sample_window_counts.append(len(windows))
        sample_indices.append(i)

    # Process all windows in batches through the model
    total_windows = len(all_windows)
    window_embeddings = np.zeros((total_windows, hidden_size), dtype=np.float32)

    for w_start in range(0, total_windows, config.batch_size):
        w_stop = min(w_start + config.batch_size, total_windows)
        w_batch = all_windows[w_start:w_stop]

        input_ids = torch.tensor(w_batch, dtype=torch.long, device=device)
        attention_mask = (input_ids != 0).long()

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pooled = _pool_hidden_states(outputs.last_hidden_state, attention_mask, config.pooling)
            window_embeddings[w_start:w_stop] = pooled.detach().cpu().numpy().astype(np.float32)

    # Aggregate windows per sample via mean-pooling
    result = np.zeros((len(batch_texts), hidden_size), dtype=np.float32)
    window_offset = 0
    for i, n_windows in enumerate(sample_window_counts):
        sample_embeds = window_embeddings[window_offset : window_offset + n_windows]
        result[i] = sample_embeds.mean(axis=0)
        window_offset += n_windows

    return result, sample_window_counts


def extract_codebert_features(
    opcode_corpus: pd.DataFrame,
    config: CodeBertExtractionConfig,
    tokenizer: Any = None,
    model: Optional[torch.nn.Module] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    required = {CONTRACT_ID_COLUMN, "split", "opcode_text", "opcode_text_available", "unavailable_cause"}
    missing = sorted(required - set(opcode_corpus.columns))
    if missing:
        raise ValueError(f"Opcode corpus missing required columns for CodeBERT extraction: {missing}")

    if config.pooling not in VALID_POOLING:
        raise ValueError(f"`codebert.pooling` must be one of {sorted(VALID_POOLING)}.")
    if config.max_length <= 0:
        raise ValueError("`codebert.max_length` must be positive.")
    if config.batch_size <= 0:
        raise ValueError("`codebert.batch_size` must be positive.")
    if config.sliding_window and config.stride <= 0:
        raise ValueError("`codebert.stride` must be positive when sliding_window is enabled.")

    if tokenizer is None or model is None:
        loaded_tokenizer, loaded_model, hidden_size = _load_transformer_bundle(config)
        tokenizer = loaded_tokenizer
        model = loaded_model
    else:
        if not hasattr(model, "config") or not hasattr(model.config, "hidden_size"):
            raise ValueError("Provided model must expose `model.config.hidden_size`.")
        hidden_size = int(model.config.hidden_size)

    feature_columns = build_codebert_feature_columns(hidden_size)
    device = _resolve_device(config.device)
    model = model.to(device)
    model.eval()

    working = opcode_corpus.copy()
    working[CONTRACT_ID_COLUMN] = working[CONTRACT_ID_COLUMN].fillna("").astype(str).str.strip()
    working["opcode_text"] = working["opcode_text"].fillna("").astype(str)
    working["opcode_text_available"] = working["opcode_text_available"].fillna(False).astype(bool)
    working["unavailable_cause"] = working["unavailable_cause"].fillna("").astype(str).str.strip()

    total_rows = int(len(working))
    feature_matrix = np.zeros((total_rows, hidden_size), dtype=np.float32)
    token_length_original = np.zeros(total_rows, dtype=np.int64)
    token_length_effective = np.zeros(total_rows, dtype=np.int64)
    token_truncated = np.zeros(total_rows, dtype=bool)
    num_windows = np.ones(total_rows, dtype=np.int64)  # default 1 (single pass)

    available_indices = np.flatnonzero(working["opcode_text_available"].to_numpy(dtype=bool))
    available_texts = working.iloc[available_indices]["opcode_text"].tolist()

    for start in range(0, len(available_indices), config.batch_size):
        stop = min(start + config.batch_size, len(available_indices))
        batch_indices = available_indices[start:stop]
        batch_texts = available_texts[start:stop]

        original_lengths = _token_lengths(tokenizer, batch_texts)
        token_length_original[batch_indices] = np.asarray(original_lengths, dtype=np.int64)
        token_truncated[batch_indices] = np.asarray([length > config.max_length for length in original_lengths], dtype=bool)

        if config.sliding_window:
            # Use effective length = original (all tokens covered via windows)
            token_length_effective[batch_indices] = np.asarray(original_lengths, dtype=np.int64)
            embeddings, window_counts = _extract_sliding_window(
                tokenizer, model, batch_texts, original_lengths, config, device, hidden_size
            )
            feature_matrix[batch_indices] = embeddings
            num_windows[batch_indices] = np.asarray(window_counts, dtype=np.int64)
        else:
            token_length_effective[batch_indices] = np.minimum(
                np.asarray(original_lengths, dtype=np.int64), config.max_length
            )
            feature_matrix[batch_indices] = _extract_single_pass(
                tokenizer, model, batch_texts, config, device, hidden_size
            )

    output = working[[CONTRACT_ID_COLUMN, "split", "opcode_text_available", "unavailable_cause"]].copy()
    output = output.rename(columns={"opcode_text_available": "codebert_feature_available"})
    output["tokenized_length_original"] = token_length_original
    output["tokenized_length_effective"] = token_length_effective
    output["tokenized_truncated"] = token_truncated
    output["num_windows"] = num_windows

    feature_df = pd.DataFrame(feature_matrix, columns=feature_columns)
    output = pd.concat([output.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)

    unavailable = output[~output["codebert_feature_available"]]
    unavailable_by_cause = unavailable["unavailable_cause"].value_counts().to_dict()
    unavailable_by_cause = {str(key): int(value) for key, value in unavailable_by_cause.items()}

    multi_window_rows = int((output["num_windows"] > 1).sum())
    summary = {
        "rows_total": total_rows,
        "rows_encoded": int(output["codebert_feature_available"].sum()),
        "rows_unavailable": int((~output["codebert_feature_available"]).sum()),
        "unavailable_by_cause": unavailable_by_cause,
        "hidden_size": hidden_size,
        "feature_columns": feature_columns,
        "model_name": config.model_name,
        "pooling": config.pooling,
        "max_length": int(config.max_length),
        "batch_size": int(config.batch_size),
        "device": str(device),
        "sliding_window": config.sliding_window,
        "stride": int(config.stride),
        "truncated_rows": int(output["tokenized_truncated"].sum()),
        "multi_window_rows": multi_window_rows,
    }
    return output, summary
