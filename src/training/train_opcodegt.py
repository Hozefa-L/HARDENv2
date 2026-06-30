import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.models.opcodegt import OpcodeGT, OpcodeGTConfig, SimpleGINFusion, build_opcodegt, VALID_MODES
from src.training.losses import masked_batch_metrics, masked_bce_with_logits
from src.training.phase4_dataset import (
    DEFAULT_PHASE4_CONFIG_PATH,
    Phase4Dataset,
    phase4_collate_fn,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Phase4TrainConfig:
    run_mode: str
    mode: str
    hidden_dim: int
    dropout: float
    batch_size: int
    lr: float
    weight_decay: float
    max_steps: int
    device: str
    seed: int
    num_workers: int
    checkpoint_dir: Path
    model_spec_md_path: Path
    smoke_report_json_path: Path
    run_manifest_json_path: Path
    swc_ids: List[int]


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
    return f"python -m src.training.train_opcodegt --config {_rel(config_path.resolve())}"


def _checkpoint_filename(run_mode: str) -> str:
    return "opcodegt_smoke_step_last.pt" if run_mode == "smoke" else "opcodegt_last.pt"


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


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def _write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def _load_train_config(config_path: Path) -> Phase4TrainConfig:
    raw = _safe_read_mapping(config_path, "Phase 4 config")

    benchmark = raw.get("main_benchmark", {})
    if benchmark is None:
        benchmark = {}
    if not isinstance(benchmark, dict):
        raise ValueError("`main_benchmark` must be a mapping when provided.")
    swc_ids_raw = benchmark.get("swc_ids", [])
    if not isinstance(swc_ids_raw, list) or not swc_ids_raw:
        raise ValueError("`main_benchmark.swc_ids` must be a non-empty list.")
    swc_ids = [int(value) for value in swc_ids_raw]

    run_cfg = raw.get("run", {})
    if run_cfg is None:
        run_cfg = {}
    if not isinstance(run_cfg, dict):
        raise ValueError("`run` must be a mapping when provided.")
    run_mode = str(run_cfg.get("mode", "smoke")).strip().lower()
    if run_mode not in {"smoke", "full"}:
        raise ValueError("`run.mode` must be either `smoke` or `full`.")

    model_cfg = raw.get("model", {})
    if model_cfg is None:
        model_cfg = {}
    if not isinstance(model_cfg, dict):
        raise ValueError("`model` must be a mapping when provided.")

    training_cfg = raw.get("training", {})
    if training_cfg is None:
        training_cfg = {}
    if not isinstance(training_cfg, dict):
        raise ValueError("`training` must be a mapping when provided.")

    outputs = raw.get("outputs", {})
    if outputs is None:
        outputs = {}
    if not isinstance(outputs, dict):
        raise ValueError("`outputs` must be a mapping when provided.")

    mode = str(model_cfg.get("mode", "fused")).strip().lower()
    if mode not in VALID_MODES:
        raise ValueError(f"`model.mode` must be one of {sorted(VALID_MODES)}.")

    hidden_dim = int(model_cfg.get("hidden_dim", 256))
    dropout = float(model_cfg.get("dropout", 0.1))
    batch_size = int(training_cfg.get("batch_size", 16))
    lr = float(training_cfg.get("lr", 5e-4))
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    max_steps = int(training_cfg.get("max_steps", 1))
    device = str(training_cfg.get("device", "cpu")).strip().lower()
    seed = int(training_cfg.get("seed", 42))
    num_workers = int(training_cfg.get("num_workers", 0))

    if hidden_dim <= 0:
        raise ValueError("`model.hidden_dim` must be positive.")
    if dropout < 0.0 or dropout >= 1.0:
        raise ValueError("`model.dropout` must be in [0.0, 1.0).")
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

    default_checkpoint_dir = "checkpoints/phase4_smoke" if run_mode == "smoke" else "checkpoints/phase4"
    default_report_json = (
        "reports/phase4/phase4_smoke_train_report.json"
        if run_mode == "smoke"
        else "reports/phase4/phase4_train_report.json"
    )
    default_run_manifest = (
        "reports/phase4/phase4_run_manifest.json"
        if run_mode == "smoke"
        else "reports/phase4/phase4_full_run_manifest.json"
    )

    checkpoint_dir = _resolve_path(outputs.get("checkpoint_dir") or default_checkpoint_dir)
    model_spec_md_path = _resolve_path(outputs.get("model_spec_md") or "reports/phase4/model_spec.md")
    smoke_report_json_path = _resolve_path(
        outputs.get("report_json") or outputs.get("smoke_report_json") or default_report_json
    )
    run_manifest_json_path = _resolve_path(outputs.get("run_manifest_json") or default_run_manifest)

    return Phase4TrainConfig(
        run_mode=run_mode,
        mode=mode,
        hidden_dim=hidden_dim,
        dropout=dropout,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        max_steps=max_steps,
        device=device,
        seed=seed,
        num_workers=num_workers,
        checkpoint_dir=checkpoint_dir,
        model_spec_md_path=model_spec_md_path,
        smoke_report_json_path=smoke_report_json_path,
        run_manifest_json_path=run_manifest_json_path,
        swc_ids=swc_ids,
    )


def _build_model_spec(report: Mapping[str, Any]) -> str:
    validation_heading = "Smoke validation path" if str(report.get("run_mode", "smoke")) == "smoke" else "Training/validation path"
    return (
        "# Phase 4 OpcodeGT Model Spec\n\n"
        f"- Generated at (UTC): `{report['generated_at_utc']}`\n"
        f"- Mode: `{report['model']['mode']}`\n"
        f"- SWC target order: `{report['main_benchmark_swcs']}`\n\n"
        "## Architecture\n\n"
        "- Opcode branch: 2-layer MLP encoder over CodeBERT feature vectors (`cb_*`).\n"
        "- Graph branch: 2-layer MLP encoder over graph-level flat features (`gf_*`).\n"
        "- Fusion (fused mode): gated interpolation + layer norm.\n"
        "- Classifier head: linear multilabel logits over 11 SWCs.\n\n"
        "## Dimensions\n\n"
        f"- opcode_input_dim: **{report['feature_dims']['opcode_dim']}**\n"
        f"- graph_input_dim: **{report['feature_dims']['graph_dim']}**\n"
        f"- hidden_dim: **{report['model']['hidden_dim']}**\n"
        f"- num_labels: **{report['feature_dims']['target_dim']}**\n"
        f"- parameters: **{report['model']['parameter_count']}**\n\n"
        f"## {validation_heading}\n\n"
        "- dataset load\n"
        "- forward pass\n"
        "- masked BCE loss\n"
        "- backward + optimizer step\n"
        "- checkpoint save/load roundtrip\n"
        "- tiny masked metrics on smoke batch\n"
    )


def run_phase4_smoke(config_path: Path = DEFAULT_PHASE4_CONFIG_PATH) -> Dict[str, Any]:
    run_started = datetime.now(timezone.utc)
    resolved_config_path = config_path.resolve()
    train_cfg = _load_train_config(resolved_config_path)
    _set_seed(train_cfg.seed)

    device = _resolve_device(train_cfg.device)
    dataset_all = Phase4Dataset(resolved_config_path)
    dataset_train = Phase4Dataset(resolved_config_path, split="train")

    if len(dataset_train) == 0:
        raise ValueError("Train split is empty; smoke training cannot proceed.")

    feature_dims = dataset_all.feature_shapes()
    target_dim = int(feature_dims["target_dim"])
    if target_dim != len(train_cfg.swc_ids):
        raise ValueError(
            "Target dimension mismatch between dataset and config SWCs. "
            f"dataset={target_dim}, config={len(train_cfg.swc_ids)}"
        )

    model_config = OpcodeGTConfig(
        mode=train_cfg.mode,
        opcode_input_dim=int(feature_dims["opcode_dim"]),
        graph_input_dim=int(feature_dims["graph_dim"]),
        hidden_dim=train_cfg.hidden_dim,
        num_labels=target_dim,
        dropout=train_cfg.dropout,
        use_gnn=dataset_train.has_graph_artifacts and train_cfg.mode in {"graph_only", "fused"},
        num_gnn_layers=3,
    )
    model = build_opcodegt(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    train_loader = DataLoader(
        dataset_train,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        collate_fn=phase4_collate_fn,
    )
    train_iter = iter(train_loader)

    # Compute class weights for imbalanced labels
    from src.training.losses import compute_pos_weight
    _all_targets = []
    _all_masks = []
    for i in range(len(dataset_train)):
        item = dataset_train[i]
        _all_targets.append(item["targets"])
        _all_masks.append(item["target_mask"])
    _t = torch.stack(_all_targets)
    _m = torch.stack(_all_masks)
    pos_weight = compute_pos_weight(_t, _m).to(device)

    # LR scheduler (cosine annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg.max_steps, eta_min=1e-6
    )

    step_metrics: List[Dict[str, float]] = []
    last_batch_tensors: Dict[str, torch.Tensor] = {}
    best_val_loss = float("inf")
    patience_counter = 0
    patience_limit = max(50, train_cfg.max_steps // 5)

    for step in range(1, train_cfg.max_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        opcode_features = batch["opcode_features"].to(device=device, dtype=torch.float32)
        graph_features = batch["graph_features"].to(device=device, dtype=torch.float32)
        targets = batch["targets"].to(device=device, dtype=torch.float32)
        target_mask = batch["target_mask"].to(device=device, dtype=torch.bool)

        # GNN inputs
        graph_x = batch.get("graph_x")
        graph_edge_index = batch.get("graph_edge_index")
        graph_batch_vec = batch.get("graph_batch")
        graph_edge_type = batch.get("graph_edge_type")
        if graph_x is not None:
            graph_x = graph_x.to(device)
            graph_edge_index = graph_edge_index.to(device)
            graph_batch_vec = graph_batch_vec.to(device)
        if graph_edge_type is not None:
            graph_edge_type = graph_edge_type.to(device)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            opcode_features=opcode_features,
            graph_features=graph_features,
            graph_x=graph_x,
            graph_edge_index=graph_edge_index,
            graph_batch=graph_batch_vec,
            graph_edge_type=graph_edge_type,
        )
        loss = masked_bce_with_logits(
            logits=logits, targets=targets, target_mask=target_mask,
            reduction="mean", pos_weight=pos_weight,
        )
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite smoke loss at step {step}: {float(loss.item())}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        metrics = masked_batch_metrics(logits=logits.detach(), targets=targets, target_mask=target_mask)
        metrics["step"] = float(step)
        metrics["loss"] = float(loss.item())
        metrics["lr"] = float(scheduler.get_last_lr()[0])
        step_metrics.append(metrics)

        last_batch_tensors = {
            "opcode_features": opcode_features.detach().clone(),
            "graph_features": graph_features.detach().clone(),
        }
        if graph_x is not None:
            last_batch_tensors["graph_x"] = graph_x.detach().clone()
            last_batch_tensors["graph_edge_index"] = graph_edge_index.detach().clone()
            last_batch_tensors["graph_batch"] = graph_batch_vec.detach().clone()
        if graph_edge_type is not None:
            last_batch_tensors["graph_edge_type"] = graph_edge_type.detach().clone()

    output_notes: List[Dict[str, str]] = []
    resolved_checkpoint_dir = _resolve_output_dir(train_cfg.checkpoint_dir, output_notes)
    resolved_model_spec_path = _resolve_output_path(train_cfg.model_spec_md_path, output_notes)
    resolved_smoke_report_path = _resolve_output_path(train_cfg.smoke_report_json_path, output_notes)
    resolved_run_manifest_path = _resolve_output_path(train_cfg.run_manifest_json_path, output_notes)

    resolved_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (resolved_checkpoint_dir / _checkpoint_filename(train_cfg.run_mode)).resolve()
    checkpoint_payload = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "step_count": int(train_cfg.max_steps),
        "model_config": {
            "mode": model_config.mode,
            "opcode_input_dim": model_config.opcode_input_dim,
            "graph_input_dim": model_config.graph_input_dim,
            "hidden_dim": model_config.hidden_dim,
            "num_labels": model_config.num_labels,
            "dropout": model_config.dropout,
            "use_gnn": model_config.use_gnn,
            "num_gnn_layers": model_config.num_gnn_layers,
            "architecture": model_config.architecture,
            "use_cross_attention": model_config.use_cross_attention,
            "use_hmpgt": model_config.use_hmpgt,
            "use_label_attention": model_config.use_label_attention,
            "cfg_only": model_config.cfg_only,
            "num_heads": model_config.num_heads,
            "num_edge_types": model_config.num_edge_types,
        },
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "feature_dims": dict(feature_dims),
    }
    torch.save(checkpoint_payload, checkpoint_path)

    loaded_payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    loaded_model_cfg = OpcodeGTConfig(
        mode=str(loaded_payload["model_config"]["mode"]),
        opcode_input_dim=int(loaded_payload["model_config"]["opcode_input_dim"]),
        graph_input_dim=int(loaded_payload["model_config"]["graph_input_dim"]),
        hidden_dim=int(loaded_payload["model_config"]["hidden_dim"]),
        num_labels=int(loaded_payload["model_config"]["num_labels"]),
        dropout=float(loaded_payload["model_config"]["dropout"]),
        use_gnn=bool(loaded_payload["model_config"].get("use_gnn", False)),
        num_gnn_layers=int(loaded_payload["model_config"].get("num_gnn_layers", 3)),
        architecture=str(loaded_payload["model_config"].get("architecture", "simple_gin_fusion")),
        use_cross_attention=bool(loaded_payload["model_config"].get("use_cross_attention", True)),
        use_hmpgt=bool(loaded_payload["model_config"].get("use_hmpgt", True)),
        use_label_attention=bool(loaded_payload["model_config"].get("use_label_attention", True)),
        cfg_only=bool(loaded_payload["model_config"].get("cfg_only", False)),
        num_heads=int(loaded_payload["model_config"].get("num_heads", 8)),
        num_edge_types=int(loaded_payload["model_config"].get("num_edge_types", 9)),
    )
    reloaded_model = build_opcodegt(loaded_model_cfg).to(device)
    reloaded_model.load_state_dict(loaded_payload["model_state_dict"])

    model.eval()
    reloaded_model.eval()
    with torch.no_grad():
        forward_kwargs = {
            "opcode_features": last_batch_tensors["opcode_features"],
            "graph_features": last_batch_tensors["graph_features"],
        }
        if "graph_x" in last_batch_tensors:
            forward_kwargs["graph_x"] = last_batch_tensors["graph_x"]
            forward_kwargs["graph_edge_index"] = last_batch_tensors["graph_edge_index"]
            forward_kwargs["graph_batch"] = last_batch_tensors["graph_batch"]
        if "graph_edge_type" in last_batch_tensors:
            forward_kwargs["graph_edge_type"] = last_batch_tensors["graph_edge_type"]
        original_logits = model(**forward_kwargs)
        reloaded_logits = reloaded_model(**forward_kwargs)
    checkpoint_roundtrip_match = bool(torch.allclose(original_logits, reloaded_logits, atol=1e-6, rtol=1e-6))
    if not checkpoint_roundtrip_match:
        raise ValueError("Checkpoint roundtrip logits mismatch.")

    report: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "phase4_smoke_training" if train_cfg.run_mode == "smoke" else "phase4_training",
        "run_mode": train_cfg.run_mode,
        "main_benchmark_swcs": list(train_cfg.swc_ids),
        "inputs": {
            "phase4_config_yaml": _rel(resolved_config_path),
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
            "checkpoint_path": _rel(checkpoint_path),
            "model_spec_md": _rel(resolved_model_spec_path),
            "report_json": _rel(resolved_smoke_report_path),
            "smoke_report_json": _rel(resolved_smoke_report_path),
            "run_manifest_json": _rel(resolved_run_manifest_path),
        },
        "configured_outputs": {
            "checkpoint_dir": _rel(train_cfg.checkpoint_dir),
            "report_json": _rel(train_cfg.smoke_report_json_path),
            "model_spec_md": _rel(train_cfg.model_spec_md_path),
            "smoke_report_json": _rel(train_cfg.smoke_report_json_path),
            "run_manifest_json": _rel(train_cfg.run_manifest_json_path),
        },
        "output_resolution_notes": output_notes,
        "model": {
            "mode": train_cfg.mode,
            "hidden_dim": int(train_cfg.hidden_dim),
            "dropout": float(train_cfg.dropout),
            "parameter_count": int(model.parameter_count()),
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
        "smoke_step_metrics": step_metrics,
        "checkpoint_roundtrip_match": checkpoint_roundtrip_match,
        "smoke_batch_contract_count": int(len(last_batch_tensors["opcode_features"])),
    }

    model_spec_md = _build_model_spec(report)
    _write_json(report, resolved_smoke_report_path)
    _write_text(model_spec_md, resolved_model_spec_path)

    run_manifest = {
        "run_started_utc": run_started.isoformat(),
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "entrypoint": _entrypoint_for_config(resolved_config_path),
        "config_path": _rel(resolved_config_path),
        "inputs": report["inputs"],
        "outputs": report["outputs"],
        "summary": {
            "mode": report["model"]["mode"],
            "parameter_count": report["model"]["parameter_count"],
            "steps_executed": int(len(step_metrics)),
            "checkpoint_roundtrip_match": bool(report["checkpoint_roundtrip_match"]),
            "final_loss": float(step_metrics[-1]["loss"]),
        },
    }
    _write_json(run_manifest, resolved_run_manifest_path)
    return report


def run_phase4_training(config_path: Path = DEFAULT_PHASE4_CONFIG_PATH) -> Dict[str, Any]:
    return run_phase4_smoke(config_path=config_path)


def _print_summary(report: Mapping[str, Any]) -> None:
    final_metrics = report["smoke_step_metrics"][-1]
    print(f"mode: {report['model']['mode']}")
    print(f"parameter_count: {report['model']['parameter_count']}")
    print(f"final_loss: {final_metrics['loss']:.6f}")
    print(f"masked_accuracy: {final_metrics['masked_accuracy']:.6f}")
    print(f"checkpoint_roundtrip_match: {report['checkpoint_roundtrip_match']}")
    print(f"report_json: {report['outputs']['report_json']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 4 OpcodeGT training (smoke/full configured in YAML).")
    parser.add_argument("--config", type=Path, default=DEFAULT_PHASE4_CONFIG_PATH)
    args = parser.parse_args()
    report = run_phase4_training(args.config.resolve())
    _print_summary(report)


if __name__ == "__main__":
    main()
