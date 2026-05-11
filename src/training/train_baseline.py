import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Subset

from src.baselines import (
    BaselineMLP, BaselineMLPConfig,
    BiLSTMBaseline, BiLSTMBaselineConfig,
    ClassicalBaselineConfig, MaskedOneVsRestLogisticRegression,
    MaskedOneVsRestLightGBM,
    CodeBERTClassifier, CodeBERTClassifierConfig,
    GATBaseline, GATBaselineConfig,
    GCNBaseline, GCNBaselineConfig,
)
from src.training.losses import masked_batch_metrics, masked_bce_with_logits
from src.training.phase4_dataset import DEFAULT_PHASE4_CONFIG_PATH, REQUIRED_SPLITS, Phase4Dataset, phase4_collate_fn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHASE5_CONFIG_PATH = PROJECT_ROOT / "configs/phase5.yaml"

BASELINE_REGISTRY: Dict[str, Dict[str, str]] = {
    "classical_graph_lr": {"family": "classical", "feature_set": "graph"},
    "classical_xgboost": {"family": "classical", "feature_set": "enriched"},
    "classical_rf": {"family": "classical", "feature_set": "enriched"},
    "classical_lgbm": {"family": "classical", "feature_set": "enriched"},
    "mlp_opcode": {"family": "mlp", "mode": "opcode_only"},
    "mlp_graph": {"family": "mlp", "mode": "graph_only"},
    "mlp_fusion_concat": {"family": "mlp", "mode": "fusion_concat"},
    "codebert_classifier": {"family": "codebert", "mode": "opcode_only"},
    "gcn_baseline": {"family": "gcn", "mode": "graph_only"},
    "gat_baseline": {"family": "gat", "mode": "graph_only"},
    "bilstm_baseline": {"family": "bilstm", "mode": "opcode_sequence"},
}
DEFAULT_BASELINE_ORDER = list(BASELINE_REGISTRY.keys())


@dataclass(frozen=True)
class Phase5TrainConfig:
    run_mode: str
    swc_ids: List[int]
    phase4_config_path: Path
    selected_baselines: List[str]
    mlp_hidden_dim: int
    mlp_dropout: float
    classical: ClassicalBaselineConfig
    batch_size: int
    lr: float
    weight_decay: float
    max_steps: int
    device: str
    seed: int
    num_workers: int
    train_subset_size: int
    eval_subset_size: int
    checkpoint_dir: Path
    smoke_report_json_path: Path
    run_manifest_json_path: Path


def _resolve_path(path_value: Any) -> Path:
    raw = Path(str(path_value))
    if raw.is_absolute():
        return raw
    return (PROJECT_ROOT / raw).resolve()


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _safe_read_mapping(path: Path, context: str) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {context}: {path}")
    with path.open("r", encoding="utf-8") as fp:
        payload = yaml.safe_load(fp)
    if not isinstance(payload, dict):
        raise ValueError(f"{context} must be a mapping: {path}")
    return payload


def _entrypoint_for_config(config_path: Path) -> str:
    return f"python -m src.training.train_baseline --config {_rel(config_path.resolve())}"


def _classical_checkpoint_filename(baseline_id: str, run_mode: str) -> str:
    suffix = "_smoke" if run_mode == "smoke" else ""
    return f"{baseline_id}{suffix}.pkl"


def _mlp_checkpoint_filename(baseline_id: str, run_mode: str) -> str:
    suffix = "_smoke_last" if run_mode == "smoke" else "_last"
    return f"{baseline_id}{suffix}.pt"


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


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


def _resolve_output_dir(configured_dir: Path, notes: List[Dict[str, str]]) -> Path:
    if not configured_dir.exists():
        return configured_dir
    if not configured_dir.is_dir():
        raise ValueError(f"Configured checkpoint output is not a directory path: {configured_dir}")
    if not any(configured_dir.iterdir()):
        return configured_dir

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = configured_dir.with_name(f"{configured_dir.name}_{timestamp}")
    suffix = 1
    while candidate.exists():
        candidate = configured_dir.with_name(f"{configured_dir.name}_{timestamp}_{suffix}")
        suffix += 1
    notes.append(
        {
            "configured_path": _rel(configured_dir),
            "resolved_path": _rel(candidate),
            "reason": "configured_output_already_exists",
        }
    )
    return candidate


def _normalize_swc_ids(raw_values: Sequence[Any]) -> List[int]:
    if not raw_values:
        raise ValueError("`main_benchmark.swc_ids` must be a non-empty list.")
    normalized: List[int] = []
    seen = set()
    for raw in raw_values:
        swc_id = int(raw)
        if swc_id not in seen:
            normalized.append(swc_id)
            seen.add(swc_id)
    return normalized


def _normalize_baseline_ids(raw_values: Sequence[Any]) -> List[str]:
    if not raw_values:
        raise ValueError("`baselines.selected` must be a non-empty list.")
    normalized: List[str] = []
    seen = set()
    for raw in raw_values:
        baseline_id = str(raw).strip()
        if baseline_id not in BASELINE_REGISTRY:
            raise ValueError(
                f"Unknown baseline `{baseline_id}`. Supported values: {sorted(BASELINE_REGISTRY.keys())}"
            )
        if baseline_id not in seen:
            normalized.append(baseline_id)
            seen.add(baseline_id)
    return normalized


def _load_train_config(config_path: Path) -> Phase5TrainConfig:
    raw = _safe_read_mapping(config_path, "Phase 5 config")

    run_cfg = raw.get("run", {})
    if run_cfg is None:
        run_cfg = {}
    if not isinstance(run_cfg, dict):
        raise ValueError("`run` must be a mapping when provided.")
    run_mode = str(run_cfg.get("mode", "smoke")).strip().lower()
    if run_mode not in {"smoke", "full"}:
        raise ValueError("`run.mode` must be either `smoke` or `full`.")

    benchmark = raw.get("main_benchmark", {})
    if benchmark is None:
        benchmark = {}
    if not isinstance(benchmark, dict):
        raise ValueError("`main_benchmark` must be a mapping when provided.")
    swc_ids = _normalize_swc_ids(benchmark.get("swc_ids", []))

    inputs = raw.get("inputs", {})
    if inputs is None:
        inputs = {}
    if not isinstance(inputs, dict):
        raise ValueError("`inputs` must be a mapping when provided.")
    phase4_config_path = _resolve_path(inputs.get("phase4_config_yaml") or str(DEFAULT_PHASE4_CONFIG_PATH))

    baselines_cfg = raw.get("baselines", {})
    if baselines_cfg is None:
        baselines_cfg = {}
    if not isinstance(baselines_cfg, dict):
        raise ValueError("`baselines` must be a mapping when provided.")
    selected_baselines = _normalize_baseline_ids(baselines_cfg.get("selected", DEFAULT_BASELINE_ORDER))

    mlp_cfg = baselines_cfg.get("mlp", {})
    if mlp_cfg is None:
        mlp_cfg = {}
    if not isinstance(mlp_cfg, dict):
        raise ValueError("`baselines.mlp` must be a mapping when provided.")
    mlp_hidden_dim = int(mlp_cfg.get("hidden_dim", 256))
    mlp_dropout = float(mlp_cfg.get("dropout", 0.1))
    if mlp_hidden_dim <= 0:
        raise ValueError("`baselines.mlp.hidden_dim` must be positive.")
    if mlp_dropout < 0.0 or mlp_dropout >= 1.0:
        raise ValueError("`baselines.mlp.dropout` must be in [0.0, 1.0).")

    classical_cfg = baselines_cfg.get("classical", {})
    if classical_cfg is None:
        classical_cfg = {}
    if not isinstance(classical_cfg, dict):
        raise ValueError("`baselines.classical` must be a mapping when provided.")
    classical = ClassicalBaselineConfig(
        max_iter=int(classical_cfg.get("max_iter", 200)),
        C=float(classical_cfg.get("C", 1.0)),
        solver=str(classical_cfg.get("solver", "liblinear")),
        random_state=int(classical_cfg.get("random_state", 42)),
    )
    if classical.max_iter <= 0:
        raise ValueError("`baselines.classical.max_iter` must be positive.")
    if classical.C <= 0.0:
        raise ValueError("`baselines.classical.C` must be positive.")

    training_cfg = raw.get("training", {})
    if training_cfg is None:
        training_cfg = {}
    if not isinstance(training_cfg, dict):
        raise ValueError("`training` must be a mapping when provided.")
    batch_size = int(training_cfg.get("batch_size", 16))
    lr = float(training_cfg.get("lr", 1e-3))
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    max_steps = int(training_cfg.get("max_steps", 1))
    device = str(training_cfg.get("device", "cpu")).strip().lower()
    seed = int(training_cfg.get("seed", 42))
    num_workers = int(training_cfg.get("num_workers", 0))
    if batch_size <= 0:
        raise ValueError("`training.batch_size` must be positive.")
    if lr <= 0.0:
        raise ValueError("`training.lr` must be positive.")
    if weight_decay < 0.0:
        raise ValueError("`training.weight_decay` must be >= 0.")
    if max_steps <= 0:
        raise ValueError("`training.max_steps` must be positive.")
    if num_workers < 0:
        raise ValueError("`training.num_workers` must be >= 0.")

    smoke_cfg = raw.get("smoke", {})
    if smoke_cfg is None:
        smoke_cfg = {}
    if not isinstance(smoke_cfg, dict):
        raise ValueError("`smoke` must be a mapping when provided.")
    train_subset_size = int(smoke_cfg.get("train_subset_size", 64))
    eval_subset_size = int(smoke_cfg.get("eval_subset_size", 64))
    if train_subset_size <= 0:
        raise ValueError("`smoke.train_subset_size` must be positive.")
    if eval_subset_size <= 0:
        raise ValueError("`smoke.eval_subset_size` must be positive.")

    outputs = raw.get("outputs", {})
    if outputs is None:
        outputs = {}
    if not isinstance(outputs, dict):
        raise ValueError("`outputs` must be a mapping when provided.")
    default_checkpoint_dir = "checkpoints/phase5_smoke" if run_mode == "smoke" else "checkpoints/phase5"
    default_report_json = (
        "reports/phase5/phase5_smoke_report.json"
        if run_mode == "smoke"
        else "reports/phase5/phase5_train_report.json"
    )
    default_run_manifest_json = (
        "reports/phase5/phase5_run_manifest.json"
        if run_mode == "smoke"
        else "reports/phase5/phase5_full_run_manifest.json"
    )
    checkpoint_dir = _resolve_path(outputs.get("checkpoint_dir") or default_checkpoint_dir)
    smoke_report_json_path = _resolve_path(
        outputs.get("report_json") or outputs.get("smoke_report_json") or default_report_json
    )
    run_manifest_json_path = _resolve_path(outputs.get("run_manifest_json") or default_run_manifest_json)

    return Phase5TrainConfig(
        run_mode=run_mode,
        swc_ids=swc_ids,
        phase4_config_path=phase4_config_path,
        selected_baselines=selected_baselines,
        mlp_hidden_dim=mlp_hidden_dim,
        mlp_dropout=mlp_dropout,
        classical=classical,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        max_steps=max_steps,
        device=device,
        seed=seed,
        num_workers=num_workers,
        train_subset_size=train_subset_size,
        eval_subset_size=eval_subset_size,
        checkpoint_dir=checkpoint_dir,
        smoke_report_json_path=smoke_report_json_path,
        run_manifest_json_path=run_manifest_json_path,
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(configured: str) -> torch.device:
    normalized = str(configured).strip().lower()
    if normalized == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cpu":
        return torch.device("cpu")
    raise ValueError("`training.device` must be either `cpu` or `cuda`.")


def _to_subset(dataset: Dataset, size: int) -> Subset:
    if size <= 0:
        raise ValueError("Subset size must be positive.")
    actual_size = min(int(size), len(dataset))
    if actual_size <= 0:
        raise ValueError("Subset size resolved to zero rows.")
    return Subset(dataset, list(range(actual_size)))


def _collect_full_batch(dataset: Dataset, subset_size: int) -> Tuple[Dict[str, Any], int]:
    subset = _to_subset(dataset, subset_size)
    loader = DataLoader(
        subset,
        batch_size=len(subset),
        shuffle=False,
        num_workers=0,
        collate_fn=phase4_collate_fn,
    )
    return next(iter(loader)), int(len(subset))


def _compute_masked_metrics(logits: torch.Tensor, targets: torch.Tensor, target_mask: torch.Tensor) -> Dict[str, float]:
    loss = masked_bce_with_logits(logits=logits, targets=targets, target_mask=target_mask, reduction="mean")
    metrics = masked_batch_metrics(logits=logits, targets=targets, target_mask=target_mask)
    metrics["loss"] = float(loss.item())
    return metrics


def _run_classical_graph_smoke(
    baseline_id: str,
    cfg: Phase5TrainConfig,
    dataset_train: Phase4Dataset,
    dataset_val: Phase4Dataset,
    train_subset_size: int,
    eval_subset_size: int,
    checkpoint_dir: Path,
) -> Dict[str, Any]:
    train_batch, train_rows = _collect_full_batch(dataset_train, train_subset_size)
    val_batch, val_rows = _collect_full_batch(dataset_val, eval_subset_size)

    x_train = train_batch["graph_features"].to(dtype=torch.float32).cpu().numpy()
    y_train = train_batch["targets"].to(dtype=torch.int64).cpu().numpy()
    m_train = train_batch["target_mask"].to(dtype=torch.bool).cpu().numpy()
    x_val = val_batch["graph_features"].to(dtype=torch.float32).cpu().numpy()
    y_val = val_batch["targets"].to(dtype=torch.int64).cpu().numpy()
    m_val = val_batch["target_mask"].to(dtype=torch.bool).cpu().numpy()

    model = MaskedOneVsRestLogisticRegression(cfg.classical)
    model.fit(features=x_train, targets=y_train, target_mask=m_train)

    train_logits = torch.tensor(model.predict_logits(x_train), dtype=torch.float32)
    val_logits = torch.tensor(model.predict_logits(x_val), dtype=torch.float32)

    train_metrics = _compute_masked_metrics(
        logits=train_logits,
        targets=torch.tensor(y_train, dtype=torch.float32),
        target_mask=torch.tensor(m_train, dtype=torch.bool),
    )
    val_metrics = _compute_masked_metrics(
        logits=val_logits,
        targets=torch.tensor(y_val, dtype=torch.float32),
        target_mask=torch.tensor(m_val, dtype=torch.bool),
    )

    checkpoint_path = (checkpoint_dir / _classical_checkpoint_filename(baseline_id, cfg.run_mode)).resolve()
    model.save(checkpoint_path)
    reloaded_model = MaskedOneVsRestLogisticRegression.load(checkpoint_path)
    roundtrip_match = bool(
        np.allclose(model.predict_logits(x_train), reloaded_model.predict_logits(x_train), atol=1e-6, rtol=1e-6)
    )
    if not roundtrip_match:
        raise ValueError(f"Checkpoint roundtrip logits mismatch for `{baseline_id}`.")

    return {
        "baseline_id": baseline_id,
        "family": "classical",
        "mode": "graph_flat_lr",
        "feature_set": "graph_features",
        "feature_dim": int(x_train.shape[1]),
        "target_dim": int(y_train.shape[1]),
        "parameter_count": int(model.parameter_count()),
        "train_rows": int(train_rows),
        "eval_rows": int(val_rows),
        "optimizer_step_executed": False,
        "steps_executed": 1,
        "forward_path_executed": True,
        "train_metrics": train_metrics,
        "eval_metrics": val_metrics,
        "fit_info": model.fit_info,
        "checkpoint_path": _rel(checkpoint_path),
        "checkpoint_roundtrip_match": roundtrip_match,
    }


def _run_mlp_smoke(
    baseline_id: str,
    mode: str,
    cfg: Phase5TrainConfig,
    dataset_train: Phase4Dataset,
    dataset_val: Phase4Dataset,
    train_subset_size: int,
    eval_subset_size: int,
    feature_dims: Mapping[str, int],
    checkpoint_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    train_subset = _to_subset(dataset_train, train_subset_size)
    train_loader = DataLoader(
        train_subset,
        batch_size=min(cfg.batch_size, len(train_subset)),
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=phase4_collate_fn,
    )
    train_iter = iter(train_loader)

    model_config = BaselineMLPConfig(
        mode=mode,
        opcode_input_dim=int(feature_dims["opcode_dim"]),
        graph_input_dim=int(feature_dims["graph_dim"]),
        hidden_dim=int(cfg.mlp_hidden_dim),
        num_labels=int(feature_dims["target_dim"]),
        dropout=float(cfg.mlp_dropout),
    )
    model = BaselineMLP(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    step_metrics: List[Dict[str, float]] = []
    last_batch: Dict[str, torch.Tensor] = {}
    for step in range(1, cfg.max_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        opcode_features = batch["opcode_features"].to(device=device, dtype=torch.float32)
        graph_features = batch["graph_features"].to(device=device, dtype=torch.float32)
        targets = batch["targets"].to(device=device, dtype=torch.float32)
        target_mask = batch["target_mask"].to(device=device, dtype=torch.bool)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(opcode_features=opcode_features, graph_features=graph_features)
        loss = masked_bce_with_logits(logits=logits, targets=targets, target_mask=target_mask, reduction="mean")
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite smoke loss for `{baseline_id}` at step {step}: {float(loss.item())}")
        loss.backward()
        optimizer.step()

        metrics = masked_batch_metrics(logits=logits.detach(), targets=targets, target_mask=target_mask)
        metrics["step"] = float(step)
        metrics["loss"] = float(loss.item())
        step_metrics.append(metrics)

        last_batch = {
            "opcode_features": opcode_features.detach().clone(),
            "graph_features": graph_features.detach().clone(),
            "targets": targets.detach().clone(),
            "target_mask": target_mask.detach().clone(),
        }

    val_batch, val_rows = _collect_full_batch(dataset_val, eval_subset_size)
    val_opcode = val_batch["opcode_features"].to(device=device, dtype=torch.float32)
    val_graph = val_batch["graph_features"].to(device=device, dtype=torch.float32)
    val_targets = val_batch["targets"].to(device=device, dtype=torch.float32)
    val_mask = val_batch["target_mask"].to(device=device, dtype=torch.bool)

    model.eval()
    with torch.no_grad():
        val_logits = model(opcode_features=val_opcode, graph_features=val_graph)
    val_metrics = _compute_masked_metrics(logits=val_logits, targets=val_targets, target_mask=val_mask)

    checkpoint_path = (checkpoint_dir / _mlp_checkpoint_filename(baseline_id, cfg.run_mode)).resolve()
    checkpoint_payload = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_id": baseline_id,
        "mode": mode,
        "model_config": {
            "mode": mode,
            "opcode_input_dim": int(model_config.opcode_input_dim),
            "graph_input_dim": int(model_config.graph_input_dim),
            "hidden_dim": int(model_config.hidden_dim),
            "num_labels": int(model_config.num_labels),
            "dropout": float(model_config.dropout),
        },
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step_count": int(cfg.max_steps),
    }
    torch.save(checkpoint_payload, checkpoint_path)

    loaded_payload = torch.load(checkpoint_path, map_location=device)
    loaded_cfg = BaselineMLPConfig(
        mode=str(loaded_payload["model_config"]["mode"]),
        opcode_input_dim=int(loaded_payload["model_config"]["opcode_input_dim"]),
        graph_input_dim=int(loaded_payload["model_config"]["graph_input_dim"]),
        hidden_dim=int(loaded_payload["model_config"]["hidden_dim"]),
        num_labels=int(loaded_payload["model_config"]["num_labels"]),
        dropout=float(loaded_payload["model_config"]["dropout"]),
    )
    reloaded_model = BaselineMLP(loaded_cfg).to(device)
    reloaded_model.load_state_dict(loaded_payload["model_state_dict"])

    model.eval()
    reloaded_model.eval()
    with torch.no_grad():
        original_logits = model(
            opcode_features=last_batch["opcode_features"],
            graph_features=last_batch["graph_features"],
        )
        reloaded_logits = reloaded_model(
            opcode_features=last_batch["opcode_features"],
            graph_features=last_batch["graph_features"],
        )
    roundtrip_match = bool(torch.allclose(original_logits, reloaded_logits, atol=1e-6, rtol=1e-6))
    if not roundtrip_match:
        raise ValueError(f"Checkpoint roundtrip logits mismatch for `{baseline_id}`.")

    return {
        "baseline_id": baseline_id,
        "family": "mlp",
        "mode": mode,
        "feature_set": "opcode_features" if mode == "opcode_only" else "graph_features" if mode == "graph_only" else "opcode_plus_graph_concat",
        "feature_dim": int(
            feature_dims["opcode_dim"]
            if mode == "opcode_only"
            else feature_dims["graph_dim"]
            if mode == "graph_only"
            else feature_dims["opcode_dim"] + feature_dims["graph_dim"]
        ),
        "target_dim": int(feature_dims["target_dim"]),
        "parameter_count": int(model.parameter_count()),
        "train_rows": int(len(train_subset)),
        "eval_rows": int(val_rows),
        "optimizer_step_executed": True,
        "steps_executed": int(cfg.max_steps),
        "forward_path_executed": True,
        "smoke_step_metrics": step_metrics,
        "train_metrics": dict(step_metrics[-1]),
        "eval_metrics": val_metrics,
        "checkpoint_path": _rel(checkpoint_path),
        "checkpoint_roundtrip_match": roundtrip_match,
    }


def _build_metadata_summary(frame: pd.DataFrame) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for split in REQUIRED_SPLITS:
        split_frame = frame[frame["split"] == split]
        summary[split] = {
            "rows": int(len(split_frame)),
            "proxy_like_count": int(split_frame["is_proxy_like"].astype(bool).sum()),
            "stub_like_count": int(split_frame["is_stub_like"].astype(bool).sum()),
            "graph_unavailable_count": int(split_frame["graph_unavailable"].astype(bool).sum()),
            "source_group_counts": {
                str(name): int(count)
                for name, count in split_frame["source_group"].fillna("").astype(str).value_counts().to_dict().items()
            },
        }
    return summary


def run_phase5_smoke(
    config_path: Path = DEFAULT_PHASE5_CONFIG_PATH,
    baseline_override: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    run_started = datetime.now(timezone.utc)
    resolved_config_path = config_path.resolve()
    train_cfg = _load_train_config(resolved_config_path)
    selected_baselines = (
        _normalize_baseline_ids(list(baseline_override))
        if baseline_override is not None and len(list(baseline_override)) > 0
        else list(train_cfg.selected_baselines)
    )

    _set_seed(train_cfg.seed)
    device = _resolve_device(train_cfg.device)

    dataset_all = Phase4Dataset(train_cfg.phase4_config_path)
    dataset_train = Phase4Dataset(train_cfg.phase4_config_path, split="train")
    dataset_val = Phase4Dataset(train_cfg.phase4_config_path, split="val")
    if len(dataset_train) == 0:
        raise ValueError("Train split is empty; baseline smoke cannot proceed.")
    if len(dataset_val) == 0:
        raise ValueError("Validation split is empty; baseline smoke cannot proceed.")

    dataset_swc_ids = [int(column.split("_")[1]) for column in dataset_all.target_columns]
    if dataset_swc_ids != train_cfg.swc_ids:
        raise ValueError(
            "SWC order mismatch between Phase 5 config and Phase 4 dataset. "
            f"config={train_cfg.swc_ids}, dataset={dataset_swc_ids}"
        )

    feature_dims = dataset_all.feature_shapes()
    if int(feature_dims["target_dim"]) != len(train_cfg.swc_ids):
        raise ValueError(
            "Target dimension mismatch between dataset and config SWCs. "
            f"dataset={feature_dims['target_dim']}, config={len(train_cfg.swc_ids)}"
        )

    if train_cfg.run_mode == "full":
        effective_train_subset_size = len(dataset_train)
        effective_eval_subset_size = len(dataset_val)
    else:
        effective_train_subset_size = int(train_cfg.train_subset_size)
        effective_eval_subset_size = int(train_cfg.eval_subset_size)

    output_notes: List[Dict[str, str]] = []
    resolved_checkpoint_dir = _resolve_output_dir(train_cfg.checkpoint_dir, output_notes)
    resolved_smoke_report_path = _resolve_output_path(train_cfg.smoke_report_json_path, output_notes)
    resolved_run_manifest_path = _resolve_output_path(train_cfg.run_manifest_json_path, output_notes)
    resolved_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    baseline_results: List[Dict[str, Any]] = []
    for baseline_id in selected_baselines:
        baseline_spec = BASELINE_REGISTRY[baseline_id]
        if baseline_spec["family"] == "classical":
            result = _run_classical_graph_smoke(
                baseline_id=baseline_id,
                cfg=train_cfg,
                dataset_train=dataset_train,
                dataset_val=dataset_val,
                train_subset_size=effective_train_subset_size,
                eval_subset_size=effective_eval_subset_size,
                checkpoint_dir=resolved_checkpoint_dir,
            )
        else:
            result = _run_mlp_smoke(
                baseline_id=baseline_id,
                mode=str(baseline_spec["mode"]),
                cfg=train_cfg,
                dataset_train=dataset_train,
                dataset_val=dataset_val,
                train_subset_size=effective_train_subset_size,
                eval_subset_size=effective_eval_subset_size,
                feature_dims=feature_dims,
                checkpoint_dir=resolved_checkpoint_dir,
                device=device,
            )
        baseline_results.append(result)

    all_roundtrip = all(bool(item["checkpoint_roundtrip_match"]) for item in baseline_results)
    if not all_roundtrip:
        raise ValueError("At least one baseline failed checkpoint roundtrip verification.")

    report: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "phase5_baseline_smoke" if train_cfg.run_mode == "smoke" else "phase5_baseline_training",
        "run_mode": train_cfg.run_mode,
        "main_benchmark_swcs": list(train_cfg.swc_ids),
        "inputs": {
            "phase5_config_yaml": _rel(resolved_config_path),
            "phase4_config_yaml": _rel(train_cfg.phase4_config_path),
            "phase3_run_manifest_json": _rel(dataset_all.config.phase3_run_manifest_path),
            "phase2_graph_builder_run_manifest_json": _rel(dataset_all.config.phase2_graph_builder_run_manifest_path),
            "phase3_feature_index_parquet": _rel(dataset_all.config.feature_index_path),
            "phase3_tfidf_features_parquet": _rel(dataset_all.config.tfidf_features_path),
            "phase3_pattern_features_parquet": _rel(dataset_all.config.pattern_features_path),
            "phase3_graph_level_features_parquet": _rel(dataset_all.config.graph_features_path),
            "split_root": _rel(dataset_all.config.split_root),
        },
        "outputs": {
            "checkpoint_dir": _rel(resolved_checkpoint_dir),
            "report_json": _rel(resolved_smoke_report_path),
            "smoke_report_json": _rel(resolved_smoke_report_path),
            "run_manifest_json": _rel(resolved_run_manifest_path),
        },
        "configured_outputs": {
            "checkpoint_dir": _rel(train_cfg.checkpoint_dir),
            "report_json": _rel(train_cfg.smoke_report_json_path),
            "smoke_report_json": _rel(train_cfg.smoke_report_json_path),
            "run_manifest_json": _rel(train_cfg.run_manifest_json_path),
        },
        "output_resolution_notes": output_notes,
        "selected_baselines": selected_baselines,
        "smoke": {
            "train_subset_size": int(effective_train_subset_size),
            "eval_subset_size": int(effective_eval_subset_size),
        },
        "training": {
            "batch_size": int(train_cfg.batch_size),
            "lr": float(train_cfg.lr),
            "weight_decay": float(train_cfg.weight_decay),
            "max_steps": int(train_cfg.max_steps),
            "seed": int(train_cfg.seed),
            "device": str(device),
        },
        "feature_dims": {
            "opcode_dim": int(feature_dims["opcode_dim"]),
            "graph_dim": int(feature_dims["graph_dim"]),
            "target_dim": int(feature_dims["target_dim"]),
        },
        "split_counts": dict(dataset_all.split_counts()),
        "preservation_checks": dict(dataset_all.preservation_checks()),
        "metadata_summary": _build_metadata_summary(dataset_all.frame),
        "baseline_results": baseline_results,
        "all_checkpoint_roundtrip_match": bool(all_roundtrip),
    }
    _write_json(report, resolved_smoke_report_path)

    run_manifest = {
        "run_started_utc": run_started.isoformat(),
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "entrypoint": _entrypoint_for_config(resolved_config_path),
        "config_path": _rel(resolved_config_path),
        "inputs": report["inputs"],
        "outputs": report["outputs"],
        "summary": {
            "baseline_count": int(len(baseline_results)),
            "selected_baselines": list(selected_baselines),
            "all_checkpoint_roundtrip_match": bool(all_roundtrip),
            "train_split_count": int(report["split_counts"].get("train", 0)),
            "val_split_count": int(report["split_counts"].get("val", 0)),
            "test_split_count": int(report["split_counts"].get("test", 0)),
        },
    }
    _write_json(run_manifest, resolved_run_manifest_path)
    return report


def run_phase5_training(
    config_path: Path = DEFAULT_PHASE5_CONFIG_PATH,
    baseline_override: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    return run_phase5_smoke(config_path=config_path, baseline_override=baseline_override)


def _print_summary(report: Mapping[str, Any]) -> None:
    print(f"baseline_count: {len(report['baseline_results'])}")
    print(f"all_checkpoint_roundtrip_match: {report['all_checkpoint_roundtrip_match']}")
    for item in report["baseline_results"]:
        print(
            f"{item['baseline_id']}: family={item['family']}, "
            f"train_loss={item['train_metrics']['loss']:.6f}, "
            f"eval_loss={item['eval_metrics']['loss']:.6f}"
        )
    print(f"report_json: {report['outputs']['report_json']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 5 baseline training (smoke/full configured in YAML)."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_PHASE5_CONFIG_PATH)
    parser.add_argument(
        "--baseline-id",
        action="append",
        dest="baseline_ids",
        help=f"Optional baseline override; repeatable. Supported values: {sorted(BASELINE_REGISTRY.keys())}",
    )
    args = parser.parse_args()
    report = run_phase5_training(args.config.resolve(), baseline_override=args.baseline_ids)
    _print_summary(report)


if __name__ == "__main__":
    main()
