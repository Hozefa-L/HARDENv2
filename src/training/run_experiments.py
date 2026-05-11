from __future__ import annotations

import argparse
import json
import random
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Subset

import pandas as pd

import gc


from src.baselines import (
    BaselineMLP, BaselineMLPConfig,
    BiLSTMBaseline, BiLSTMBaselineConfig,
    ClassicalBaselineConfig, MaskedOneVsRestLogisticRegression,
    CodeBERTClassifier, CodeBERTClassifierConfig,
    GATBaseline, GATBaselineConfig,
    GCNBaseline, GCNBaselineConfig,
)
from src.baselines.classical import MaskedOneVsRestXGBoost, MaskedOneVsRestRandomForest, MaskedOneVsRestLightGBM
from src.baselines.bilstm_baseline import tokenize_opcode_text, PAD_IDX
from src.evaluation.metrics import compute_masked_multilabel_metrics, optimize_per_swc_thresholds, sigmoid
from src.models.opcodegt import OpcodeGT, OpcodeGTConfig, SimpleGINFusion, build_opcodegt, VALID_MODES
from src.training.losses import masked_bce_with_logits, compute_pos_weight
from src.training.balanced_sampler import build_weighted_sampler
from src.training.phase4_dataset import Phase4Dataset, REQUIRED_SPLITS, phase4_collate_fn
from src.training.train_baseline import BASELINE_REGISTRY, DEFAULT_BASELINE_ORDER

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHASE6_CONFIG_PATH = PROJECT_ROOT / "configs/phase6.yaml"


def _extract_gnn_inputs(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Optional[torch.Tensor]]:
    """Extract GNN graph inputs from a batch, returning None if not present."""
    gx = batch.get("graph_x")
    if gx is None:
        return {"graph_x": None, "graph_edge_index": None, "graph_batch": None, "graph_edge_type": None}
    result: Dict[str, Optional[torch.Tensor]] = {
        "graph_x": gx.to(device),
        "graph_edge_index": batch["graph_edge_index"].to(device),
        "graph_batch": batch["graph_batch"].to(device),
    }
    edge_type = batch.get("graph_edge_type")
    result["graph_edge_type"] = edge_type.to(device) if edge_type is not None else None
    return result


class RunUnavailableError(RuntimeError):
    """Raised when a matrix run cannot execute due to missing split rows or artifacts."""


class EarlyStopper:
    """Tracks validation metric and decides when to stop training."""

    def __init__(self, patience: int = 15, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best_score: Optional[float] = None
        self.best_step: int = 0
        self.counter: int = 0
        self.best_state: Optional[Dict[str, Any]] = None

    def step(self, score: float, step: int, model_state: Optional[Dict[str, Any]] = None) -> bool:
        """Returns True if training should stop."""
        improved = False
        if self.best_score is None:
            improved = True
        elif self.mode == "max" and score > self.best_score:
            improved = True
        elif self.mode == "min" and score < self.best_score:
            improved = True

        if improved:
            self.best_score = score
            self.best_step = step
            self.counter = 0
            if model_state is not None:
                self.best_state = {k: v.cpu().clone() for k, v in model_state.items()}
        else:
            self.counter += 1

        return self.counter >= self.patience

    def restore_best(self, model: torch.nn.Module) -> None:
        """Restore the best model state if available."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def _cleanup_gpu() -> None:
    """Free GPU memory between runs."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


@dataclass(frozen=True)
class NeuralHyperparams:
    hidden_dim: int
    dropout: float


@dataclass(frozen=True)
class OpcodeGTHyperparams(NeuralHyperparams):
    """Extended hyperparams for OpcodeGT with v2 architecture fields."""
    architecture: str = "opcodegt_v2"
    use_hmpgt: bool = True
    use_cross_attention: bool = False
    use_label_attention: bool = True
    cfg_only: bool = False
    num_heads: int = 8
    num_edge_types: int = 9
    num_gnn_layers: int = 3


@dataclass(frozen=True)
class Phase6RunConfig:
    swc_ids: List[int]
    phase4_config_path: Path
    phase5_config_path: Path
    variant_manifests_dir: Path
    opcodegt_modes: List[str]
    baseline_ids: List[str]
    dataset_variants: List[str]
    seeds: List[int]
    batch_size: int
    lr: float
    weight_decay: float
    max_steps: int
    num_workers: int
    device: str
    decision_threshold: float
    resume: bool
    rerun_failed: bool
    fail_fast: bool
    opcodegt: OpcodeGTHyperparams
    mlp: NeuralHyperparams
    classical: ClassicalBaselineConfig
    checkpoint_root: Path
    run_manifest_json_path: Path
    metrics_dir: Path
    models_dir: Path
    tfidf_features_path: Optional[Path] = None
    pattern_features_path: Optional[Path] = None
    feature_index_path_override: Optional[Path] = None  # e.g. 10-SWC parquet for Phase 7
    # Early stopping and ablation config
    eval_every: int = 100
    patience: int = 15
    early_stopping: bool = False
    run_timeout_seconds: int = 3600
    ablation_variants: Optional[List[Dict[str, Any]]] = None
    # Class balancing config (Phase 7+)
    use_weighted_sampler: bool = False
    max_oversample_ratio: float = 20.0
    pos_weight_cap: float = 100.0
    use_per_swc_threshold: bool = False


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
        if path.suffix in {".yaml", ".yml"}:
            payload = yaml.safe_load(fp)
        else:
            payload = json.load(fp)
    if not isinstance(payload, dict):
        raise ValueError(f"{context} must be a mapping: {path}")
    return payload


def _entrypoint_for_config(config_path: Path) -> str:
    return f"python -m src.training.run_experiments --config {_rel(config_path.resolve())}"


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


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
    # Support explicit device index: cuda:0, cuda:1, etc.
    if normalized.startswith("cuda:"):
        if not torch.cuda.is_available():
            return torch.device("cpu")
        idx_str = normalized.split(":", 1)[1]
        if not idx_str.isdigit():
            raise ValueError(f"`training.device` cuda index must be an integer, got: {configured!r}")
        idx = int(idx_str)
        n = torch.cuda.device_count()
        if idx >= n:
            raise ValueError(f"`training.device` cuda:{idx} requested but only {n} GPU(s) available.")
        return torch.device(f"cuda:{idx}")
    raise ValueError(f"`training.device` must be 'cpu', 'cuda', or 'cuda:N', got: {configured!r}")



def _normalize_unique_strings(values: Sequence[Any], context: str) -> List[str]:
    if not values:
        return []
    normalized: List[str] = []
    seen: Set[str] = set()
    for raw in values:
        value = str(raw).strip()
        if value and value not in seen:
            normalized.append(value)
            seen.add(value)
    return normalized


def _normalize_swc_ids(values: Sequence[Any]) -> List[int]:
    if not values:
        raise ValueError("`main_benchmark.swc_ids` must be a non-empty list.")
    normalized: List[int] = []
    seen: Set[int] = set()
    for raw in values:
        swc_id = int(raw)
        if swc_id not in seen:
            normalized.append(swc_id)
            seen.add(swc_id)
    return normalized


def _normalize_seeds(values: Sequence[Any]) -> List[int]:
    if not values:
        raise ValueError("`matrix.seeds` must be a non-empty list.")
    normalized: List[int] = []
    seen: Set[int] = set()
    for raw in values:
        seed = int(raw)
        if seed not in seen:
            normalized.append(seed)
            seen.add(seed)
    return normalized


def _load_phase5_baseline_ids(phase5_config_path: Path) -> List[str]:
    phase5 = _safe_read_mapping(phase5_config_path, "Phase 5 config")
    baselines_cfg = phase5.get("baselines", {})
    if baselines_cfg is None:
        baselines_cfg = {}
    if not isinstance(baselines_cfg, dict):
        raise ValueError("`baselines` in Phase 5 config must be a mapping when provided.")
    selected = baselines_cfg.get("selected", DEFAULT_BASELINE_ORDER)
    baseline_ids = _normalize_unique_strings(selected, "phase5.baselines.selected")
    unknown = sorted(set(baseline_ids) - set(BASELINE_REGISTRY.keys()))
    if unknown:
        raise ValueError(f"Phase 5 selected unsupported baselines: {unknown}")
    return baseline_ids


def _discover_variant_manifests_dir(phase4_config_path: Path) -> Path:
    phase4_cfg = _safe_read_mapping(phase4_config_path, "Phase 4 config")
    inputs = phase4_cfg.get("inputs", {})
    if inputs is None:
        inputs = {}
    if not isinstance(inputs, dict):
        raise ValueError("`inputs` in Phase 4 config must be a mapping when provided.")

    phase3_manifest_path = _resolve_path(inputs.get("phase3_run_manifest_json") or "reports/phase3/phase3_run_manifest.json")
    phase3_manifest = _safe_read_mapping(phase3_manifest_path, "Phase 3 run manifest")
    phase3_inputs = phase3_manifest.get("inputs", {})
    if not isinstance(phase3_inputs, dict):
        raise ValueError("Phase 3 run manifest must contain mapping `inputs`.")

    phase2_manifest_path = _resolve_path(
        phase3_inputs.get("phase2_graph_builder_run_manifest_json") or "reports/phase2/graph_builder_run_manifest.json"
    )
    phase2_manifest = _safe_read_mapping(phase2_manifest_path, "Phase 2 graph builder run manifest")
    phase2_outputs = phase2_manifest.get("outputs", {})
    if not isinstance(phase2_outputs, dict):
        raise ValueError("Phase 2 graph builder run manifest must contain mapping `outputs`.")

    discovered = phase2_outputs.get("variant_manifests_dir")
    if not discovered:
        raise ValueError(
            "Unable to discover Phase 2 variant manifests directory from Phase 4/3/2 manifests. "
            "Set `inputs.variant_manifests_dir` in Phase 6 config explicitly."
        )
    return _resolve_path(discovered)


def _load_config(config_path: Path) -> Phase6RunConfig:
    raw = _safe_read_mapping(config_path, "Phase 6 config")

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
    phase4_config_path = _resolve_path(inputs.get("phase4_config_yaml") or "configs/phase4.yaml")
    phase5_config_path = _resolve_path(inputs.get("phase5_config_yaml") or "configs/phase5.yaml")

    explicit_variant_dir = inputs.get("variant_manifests_dir")
    if explicit_variant_dir:
        variant_manifests_dir = _resolve_path(explicit_variant_dir)
    else:
        variant_manifests_dir = _discover_variant_manifests_dir(phase4_config_path)

    # Optional: override the feature index parquet (e.g. versioned 10-SWC file)
    feature_index_raw = inputs.get("feature_index_parquet")
    feature_index_path_override = _resolve_path(feature_index_raw) if feature_index_raw else None

    matrix = raw.get("matrix", {})
    if matrix is None:
        matrix = {}
    if not isinstance(matrix, dict):
        raise ValueError("`matrix` must be a mapping when provided.")

    opcodegt_modes = _normalize_unique_strings(matrix.get("opcodegt_modes", ["opcode_only", "graph_only", "fused"]), "matrix.opcodegt_modes")
    invalid_modes = sorted(set(opcodegt_modes) - set(VALID_MODES))
    if invalid_modes:
        raise ValueError(f"Unsupported OpcodeGT modes in `matrix.opcodegt_modes`: {invalid_modes}. Supported: {sorted(VALID_MODES)}")

    baseline_values = matrix.get("baselines")
    if baseline_values is None:
        baseline_values = _load_phase5_baseline_ids(phase5_config_path)
    baseline_ids = _normalize_unique_strings(baseline_values, "matrix.baselines")
    unknown_baselines = sorted(set(baseline_ids) - set(BASELINE_REGISTRY.keys()))
    if unknown_baselines:
        raise ValueError(f"Unsupported baselines in `matrix.baselines`: {unknown_baselines}. Supported: {sorted(BASELINE_REGISTRY.keys())}")

    dataset_variants = _normalize_unique_strings(
        matrix.get("dataset_variants", ["clean_default", "no_proxy", "cgt_only", "combined_posaug"]),
        "matrix.dataset_variants",
    )
    seeds = _normalize_seeds(matrix.get("seeds", [42, 123, 456, 789, 2024]))

    training = raw.get("training", {})
    if training is None:
        training = {}
    if not isinstance(training, dict):
        raise ValueError("`training` must be a mapping when provided.")
    batch_size = int(training.get("batch_size", 32))
    lr = float(training.get("lr", 5e-4))
    weight_decay = float(training.get("weight_decay", 0.0))
    max_steps = int(training.get("max_steps", 50))
    num_workers = int(training.get("num_workers", 0))
    device = str(training.get("device", "cpu")).strip().lower()
    threshold = float(training.get("decision_threshold", 0.5))

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
    if not (0.0 < threshold < 1.0):
        raise ValueError("`training.decision_threshold` must be in (0.0, 1.0).")

    execution = raw.get("execution", {})
    if execution is None:
        execution = {}
    if not isinstance(execution, dict):
        raise ValueError("`execution` must be a mapping when provided.")
    resume = bool(execution.get("resume", True))
    rerun_failed = bool(execution.get("rerun_failed", False))
    fail_fast = bool(execution.get("fail_fast", False))

    model_cfg = raw.get("models", {})
    if model_cfg is None:
        model_cfg = {}
    if not isinstance(model_cfg, dict):
        raise ValueError("`models` must be a mapping when provided.")
    opcodegt_cfg = model_cfg.get("opcodegt", {})
    mlp_cfg = model_cfg.get("mlp", {})
    classical_cfg = model_cfg.get("classical", {})
    if opcodegt_cfg is None:
        opcodegt_cfg = {}
    if mlp_cfg is None:
        mlp_cfg = {}
    if classical_cfg is None:
        classical_cfg = {}
    if not isinstance(opcodegt_cfg, dict) or not isinstance(mlp_cfg, dict) or not isinstance(classical_cfg, dict):
        raise ValueError("`models.opcodegt`, `models.mlp`, and `models.classical` must be mappings.")

    opcodegt_hparams = OpcodeGTHyperparams(
        hidden_dim=int(opcodegt_cfg.get("hidden_dim", 256)),
        dropout=float(opcodegt_cfg.get("dropout", 0.1)),
        architecture=str(opcodegt_cfg.get("architecture", "simple_gin_fusion")),
        use_hmpgt=bool(opcodegt_cfg.get("use_hmpgt", True)),
        use_cross_attention=bool(opcodegt_cfg.get("use_cross_attention", True)),
        use_label_attention=bool(opcodegt_cfg.get("use_label_attention", True)),
        cfg_only=bool(opcodegt_cfg.get("cfg_only", False)),
        num_heads=int(opcodegt_cfg.get("num_heads", 8)),
        num_edge_types=int(opcodegt_cfg.get("num_edge_types", 9)),
        num_gnn_layers=int(opcodegt_cfg.get("num_gnn_layers", 3)),
    )
    mlp_hparams = NeuralHyperparams(
        hidden_dim=int(mlp_cfg.get("hidden_dim", 256)),
        dropout=float(mlp_cfg.get("dropout", 0.1)),
    )
    for name, hparams in [("models.opcodegt", opcodegt_hparams), ("models.mlp", mlp_hparams)]:
        if hparams.hidden_dim <= 0:
            raise ValueError(f"`{name}.hidden_dim` must be positive.")
        if hparams.dropout < 0.0 or hparams.dropout >= 1.0:
            raise ValueError(f"`{name}.dropout` must be in [0.0, 1.0).")

    classical = ClassicalBaselineConfig(
        max_iter=int(classical_cfg.get("max_iter", 200)),
        C=float(classical_cfg.get("C", 1.0)),
        solver=str(classical_cfg.get("solver", "liblinear")),
        random_state=int(classical_cfg.get("random_state", 42)),
    )
    if classical.max_iter <= 0:
        raise ValueError("`models.classical.max_iter` must be positive.")
    if classical.C <= 0.0:
        raise ValueError("`models.classical.C` must be positive.")

    outputs = raw.get("outputs", {})
    if outputs is None:
        outputs = {}
    if not isinstance(outputs, dict):
        raise ValueError("`outputs` must be a mapping when provided.")
    checkpoint_root = _resolve_path(outputs.get("checkpoint_root") or "checkpoints/phase6")
    run_manifest_json_path = _resolve_path(outputs.get("run_manifest_json") or "reports/phase6/phase6_run_manifest.json")
    metrics_dir = _resolve_path(outputs.get("metrics_dir") or str(checkpoint_root / "metrics"))
    models_dir = _resolve_path(outputs.get("models_dir") or str(checkpoint_root / "models"))

    # Optional enriched feature paths for XGBoost/RF baselines
    enriched_cfg = raw.get("enriched_features", {})
    if enriched_cfg is None:
        enriched_cfg = {}
    if not isinstance(enriched_cfg, dict):
        raise ValueError("`enriched_features` must be a mapping when provided.")
    tfidf_raw = enriched_cfg.get("tfidf_features_parquet")
    pattern_raw = enriched_cfg.get("pattern_features_parquet")
    tfidf_features_path = _resolve_path(tfidf_raw) if tfidf_raw else None
    pattern_features_path = _resolve_path(pattern_raw) if pattern_raw else None

    # Early stopping config
    eval_every = int(training.get("eval_every", 50))
    patience = int(training.get("patience", 7))
    early_stopping = bool(training.get("early_stopping", True))
    run_timeout_seconds = int(training.get("run_timeout_seconds", 3600))

    # Ablation variants (optional list of OpcodeGT config overrides)
    ablation_cfg = model_cfg.get("opcodegt_ablation", {})
    if ablation_cfg is None:
        ablation_cfg = {}
    ablation_variants_raw = ablation_cfg.get("variants", None) if isinstance(ablation_cfg, dict) else None
    ablation_variants: Optional[List[Dict[str, Any]]] = None
    if ablation_variants_raw and isinstance(ablation_variants_raw, list):
        ablation_variants = [dict(v) for v in ablation_variants_raw]

    # Class balancing options (optional; backward-compatible defaults match Phase 6 behavior)
    balancing_cfg = raw.get("balancing", {})
    if balancing_cfg is None:
        balancing_cfg = {}
    if not isinstance(balancing_cfg, dict):
        raise ValueError("`balancing` must be a mapping when provided.")
    use_weighted_sampler = bool(balancing_cfg.get("use_weighted_sampler", False))
    max_oversample_ratio = float(balancing_cfg.get("max_oversample_ratio", 20.0))
    pos_weight_cap = float(balancing_cfg.get("pos_weight_cap", 100.0))
    use_per_swc_threshold = bool(balancing_cfg.get("use_per_swc_threshold", False))

    return Phase6RunConfig(
        swc_ids=swc_ids,
        phase4_config_path=phase4_config_path,
        phase5_config_path=phase5_config_path,
        variant_manifests_dir=variant_manifests_dir,
        opcodegt_modes=opcodegt_modes,
        baseline_ids=baseline_ids,
        dataset_variants=dataset_variants,
        seeds=seeds,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        max_steps=max_steps,
        num_workers=num_workers,
        device=device,
        decision_threshold=threshold,
        resume=resume,
        rerun_failed=rerun_failed,
        fail_fast=fail_fast,
        opcodegt=opcodegt_hparams,
        mlp=mlp_hparams,
        classical=classical,
        checkpoint_root=checkpoint_root,
        run_manifest_json_path=run_manifest_json_path,
        metrics_dir=metrics_dir,
        models_dir=models_dir,
        tfidf_features_path=tfidf_features_path,
        pattern_features_path=pattern_features_path,
        feature_index_path_override=feature_index_path_override,
        eval_every=eval_every,
        patience=patience,
        early_stopping=early_stopping,
        run_timeout_seconds=run_timeout_seconds,
        ablation_variants=ablation_variants,
        use_weighted_sampler=use_weighted_sampler,
        max_oversample_ratio=max_oversample_ratio,
        pos_weight_cap=pos_weight_cap,
        use_per_swc_threshold=use_per_swc_threshold,
    )


def _load_variant_manifest(path: Path) -> Dict[str, Any]:
    payload = _safe_read_mapping(path, f"Variant manifest `{path.name}`")
    graph_ids_raw = payload.get("graph_ids", [])
    if not isinstance(graph_ids_raw, list):
        raise ValueError(f"Variant manifest `{path}` has non-list `graph_ids`.")
    split_counts_raw = payload.get("split_counts", {})
    if split_counts_raw is None:
        split_counts_raw = {}
    if not isinstance(split_counts_raw, dict):
        raise ValueError(f"Variant manifest `{path}` has non-mapping `split_counts`.")
    split_counts = {str(split): int(split_counts_raw.get(split, 0)) for split in REQUIRED_SPLITS}
    return {
        "variant_name": str(payload.get("variant_name") or path.stem),
        "description": str(payload.get("description") or ""),
        "graph_ids": [str(x) for x in graph_ids_raw],
        "graph_id_set": set(str(x) for x in graph_ids_raw),
        "graph_count": int(payload.get("graph_count", len(graph_ids_raw))),
        "split_counts": split_counts,
        "path": path,
    }


def _load_variant_manifests(config: Phase6RunConfig) -> Dict[str, Dict[str, Any]]:
    manifests: Dict[str, Dict[str, Any]] = {}
    for variant in config.dataset_variants:
        path = (config.variant_manifests_dir / f"{variant}.json").resolve()
        if not path.exists():
            raise FileNotFoundError(f"Missing variant manifest for `{variant}`: {path}")
        payload = _load_variant_manifest(path)
        manifests[variant] = payload
    return manifests


def _build_model_specs(config: Phase6RunConfig) -> List[Dict[str, str]]:
    specs: List[Dict[str, str]] = []
    for mode in config.opcodegt_modes:
        specs.append(
            {
                "model_group": "opcodegt",
                "model_variant": mode,
                "model_id": f"opcodegt_{mode}",
                "mode": mode,
                "family": "neural",
            }
        )
    # Ablation variants: each gets its own model_variant entry
    if config.ablation_variants:
        for ablation in config.ablation_variants:
            name = str(ablation.get("name", "unnamed_ablation"))
            # Ablation variants use fused mode by default
            ablation_mode = str(ablation.get("mode", "fused"))
            specs.append(
                {
                    "model_group": "opcodegt",
                    "model_variant": name,
                    "model_id": f"opcodegt_{name}",
                    "mode": ablation_mode,
                    "family": "neural",
                    "is_ablation": "true",
                }
            )
    for baseline_id in config.baseline_ids:
        baseline_spec = BASELINE_REGISTRY[baseline_id]
        family = str(baseline_spec.get("family", "mlp"))
        mode = str(baseline_spec.get("mode", ""))
        specs.append(
            {
                "model_group": "baseline",
                "model_variant": baseline_id,
                "model_id": baseline_id,
                "mode": mode,
                "family": family,
            }
        )
    return specs


def _sanitize_token(value: str) -> str:
    allowed = []
    for ch in str(value):
        if ch.isalnum() or ch in {"_", "-"}:
            allowed.append(ch)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_")


def _run_id(model_spec: Mapping[str, str], dataset_variant: str, seed: int) -> str:
    return (
        f"{_sanitize_token(model_spec['model_group'])}"
        f"-{_sanitize_token(model_spec['model_variant'])}"
        f"__{_sanitize_token(dataset_variant)}"
        f"__seed{int(seed)}"
    )


def _build_matrix_runs(config: Phase6RunConfig, model_specs: Sequence[Mapping[str, str]]) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for dataset_variant in config.dataset_variants:
        for seed in config.seeds:
            for model_spec in model_specs:
                runs.append(
                    {
                        "run_id": _run_id(model_spec, dataset_variant=dataset_variant, seed=seed),
                        "model_group": str(model_spec["model_group"]),
                        "model_variant": str(model_spec["model_variant"]),
                        "model_id": str(model_spec["model_id"]),
                        "model_family": str(model_spec["family"]),
                        "mode": str(model_spec.get("mode", "")),
                        "is_ablation": str(model_spec.get("is_ablation", "false")),
                        "dataset_variant": str(dataset_variant),
                        "seed": int(seed),
                        "status": "pending",
                        "attempts": 0,
                        "started_utc": None,
                        "finished_utc": None,
                        "error": None,
                        "traceback": None,
                        "metrics_json": None,
                        "checkpoint_path": None,
                        "summary": {},
                    }
                )
    return runs


def _summarize_run_statuses(runs: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for run in runs:
        status = str(run.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _initialize_or_resume_manifest(
    *,
    config: Phase6RunConfig,
    config_path: Path,
    matrix_runs: Sequence[Mapping[str, Any]],
    variant_manifests: Mapping[str, Mapping[str, Any]],
    resume: bool,
) -> Dict[str, Any]:
    manifest_path = config.run_manifest_json_path.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    existing_runs: Dict[str, Dict[str, Any]] = {}
    if resume and manifest_path.exists():
        loaded = _safe_read_mapping(manifest_path, "Phase 6 run manifest")
        loaded_runs = loaded.get("runs", [])
        if isinstance(loaded_runs, list):
            for item in loaded_runs:
                if isinstance(item, dict):
                    run_id = str(item.get("run_id", "")).strip()
                    if run_id:
                        existing_runs[run_id] = dict(item)

    runs: List[Dict[str, Any]] = []
    for base in matrix_runs:
        run_id = str(base["run_id"])
        existing = existing_runs.get(run_id)
        if existing is None:
            runs.append(dict(base))
            continue
        merged = dict(base)
        merged.update(existing)
        if str(merged.get("status", "")) == "running":
            merged["status"] = "pending"
            merged["error"] = "Previous execution interrupted while status was `running`; reset to `pending`."
            merged["traceback"] = None
        runs.append(merged)

    manifest: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "phase6_full_experiments",
        "entrypoint": _entrypoint_for_config(config_path),
        "config_path": _rel(config_path.resolve()),
        "inputs": {
            "phase4_config_yaml": _rel(config.phase4_config_path),
            "phase5_config_yaml": _rel(config.phase5_config_path),
            "variant_manifests_dir": _rel(config.variant_manifests_dir),
            "variant_manifests": {
                name: _rel(Path(info["path"]).resolve()) for name, info in variant_manifests.items()
            },
        },
        "outputs": {
            "checkpoint_root": _rel(config.checkpoint_root),
            "models_dir": _rel(config.models_dir),
            "metrics_dir": _rel(config.metrics_dir),
            "run_manifest_json": _rel(config.run_manifest_json_path),
        },
        "matrix": {
            "opcodegt_modes": list(config.opcodegt_modes),
            "baselines": list(config.baseline_ids),
            "dataset_variants": list(config.dataset_variants),
            "seeds": list(config.seeds),
            "expected_run_count": int(len(runs)),
        },
        "runs": runs,
        "summary": _summarize_run_statuses(runs),
    }
    _write_json(manifest, manifest_path)
    return manifest


def _persist_manifest(manifest: Mapping[str, Any], path: Path) -> None:
    payload = dict(manifest)
    payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["summary"] = _summarize_run_statuses(payload.get("runs", []))
    _write_json(payload, path)


def _subset_indices_for_ids(dataset: Phase4Dataset, allowed_contract_ids: Set[str]) -> List[int]:
    contract_ids = dataset.frame["fp_runtime_unified"].astype(str).tolist()
    return [idx for idx, contract_id in enumerate(contract_ids) if contract_id in allowed_contract_ids]


def _metadata_summary_for_indices(dataset: Phase4Dataset, indices: Sequence[int]) -> Dict[str, Any]:
    frame = dataset.frame.iloc[list(indices)]
    source_counts = frame["source_group"].fillna("").astype(str).value_counts().to_dict()
    return {
        "rows": int(len(frame)),
        "proxy_like_count": int(frame["is_proxy_like"].astype(bool).sum()),
        "stub_like_count": int(frame["is_stub_like"].astype(bool).sum()),
        "graph_unavailable_count": int(frame["graph_unavailable"].astype(bool).sum()),
        "source_group_counts": {str(k): int(v) for k, v in source_counts.items()},
    }


def _to_subset(dataset: Dataset, indices: Sequence[int]) -> Subset:
    if not indices:
        raise RunUnavailableError("Split subset is empty after variant filtering.")
    return Subset(dataset, [int(i) for i in indices])


def _evaluate_neural_model(
    *,
    model: torch.nn.Module,
    dataset: Phase4Dataset,
    indices: Sequence[int],
    split_name: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    swc_ids: Sequence[int],
    threshold: float,
    per_swc_thresholds: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    subset = _to_subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=min(batch_size, len(subset)),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=phase4_collate_fn,
    )

    logits_chunks: List[np.ndarray] = []
    targets_chunks: List[np.ndarray] = []
    mask_chunks: List[np.ndarray] = []
    contract_ids: List[str] = []

    import inspect
    accepts_edge_type = "graph_edge_type" in inspect.signature(model.forward).parameters

    model.eval()
    with torch.no_grad():
        for batch in loader:
            opcode_features = batch["opcode_features"].to(device=device, dtype=torch.float32)
            graph_features = batch["graph_features"].to(device=device, dtype=torch.float32)
            targets = batch["targets"].to(dtype=torch.float32).cpu().numpy()
            target_mask = batch["target_mask"].to(dtype=torch.bool).cpu().numpy()

            gnn_inputs = _extract_gnn_inputs(batch, device)
            if not accepts_edge_type and "graph_edge_type" in gnn_inputs:
                del gnn_inputs["graph_edge_type"]

            logits = model(opcode_features=opcode_features, graph_features=graph_features,
                           **gnn_inputs)
            logits_np = logits.detach().cpu().numpy().astype(np.float64)

            logits_chunks.append(logits_np)
            targets_chunks.append(targets.astype(np.float64))
            mask_chunks.append(target_mask.astype(bool))
            contract_ids.extend([str(cid) for cid in batch["contract_id"]])

    logits_all = np.concatenate(logits_chunks, axis=0)
    targets_all = np.concatenate(targets_chunks, axis=0)
    mask_all = np.concatenate(mask_chunks, axis=0)
    probs_all = sigmoid(logits_all)
    metrics = compute_masked_multilabel_metrics(
        probabilities=probs_all,
        logits=logits_all,
        targets=targets_all,
        target_mask=mask_all,
        swc_ids=swc_ids,
        threshold=threshold,
        per_swc_thresholds=per_swc_thresholds,
        split_name=split_name,
    )
    return {
        "metrics": metrics,
        "rows": int(logits_all.shape[0]),
        "contract_ids": contract_ids,
        "probabilities": probs_all,
        "targets": targets_all,
        "target_mask": mask_all,
    }


def _collect_arrays_for_classical(dataset: Phase4Dataset, indices: Sequence[int]) -> Dict[str, np.ndarray]:
    subset = _to_subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=len(subset),
        shuffle=False,
        num_workers=0,
        collate_fn=phase4_collate_fn,
    )
    batch = next(iter(loader))
    return {
        "graph_features": batch["graph_features"].to(dtype=torch.float32).cpu().numpy().astype(np.float64),
        "targets": batch["targets"].to(dtype=torch.float32).cpu().numpy().astype(np.float64),
        "target_mask": batch["target_mask"].to(dtype=torch.bool).cpu().numpy().astype(bool),
        "contract_ids": np.asarray(batch["contract_id"], dtype=object),
    }


def _load_enriched_feature_map(config: Phase6RunConfig) -> Optional[Dict[str, np.ndarray]]:
    """Load TF-IDF and pattern features into a contract_id → feature_vector mapping.

    Returns None if no enriched feature paths are configured.
    """
    frames = []
    for path in [config.tfidf_features_path, config.pattern_features_path]:
        if path is None:
            continue
        if not path.exists():
            raise FileNotFoundError(f"Enriched feature file not found: {path}")
        df = pd.read_parquet(path)
        if "fp_runtime_unified" not in df.columns:
            raise ValueError(f"Enriched feature file missing 'fp_runtime_unified': {path}")
        df["fp_runtime_unified"] = df["fp_runtime_unified"].astype(str).str.strip()
        feat_cols = [c for c in df.columns if c not in {"fp_runtime_unified", "split"}]
        frames.append(df.set_index("fp_runtime_unified")[feat_cols])

    if not frames:
        return None

    merged = pd.concat(frames, axis=1)
    result: Dict[str, np.ndarray] = {}
    for contract_id in merged.index:
        result[str(contract_id)] = merged.loc[contract_id].values.astype(np.float64)
    return result


def _enrich_classical_arrays(
    arrays: Dict[str, np.ndarray],
    enriched_map: Optional[Dict[str, np.ndarray]],
) -> np.ndarray:
    """Concatenate graph features with enriched features for classical models."""
    graph_features = arrays["graph_features"]
    if enriched_map is None:
        return graph_features

    contract_ids = arrays["contract_ids"]
    extra_list = []
    for cid in contract_ids:
        cid_str = str(cid)
        if cid_str in enriched_map:
            extra_list.append(enriched_map[cid_str])
        else:
            # Use zero vector for missing contracts
            dim = next(iter(enriched_map.values())).shape[0]
            extra_list.append(np.zeros(dim, dtype=np.float64))

    extra = np.stack(extra_list, axis=0)
    return np.concatenate([graph_features, extra], axis=1)


def _evaluate_classical_model(
    *,
    model,
    split_arrays: Mapping[str, np.ndarray],
    split_name: str,
    swc_ids: Sequence[int],
    threshold: float,
    per_swc_thresholds: Optional[Sequence[float]] = None,
    feature_key: str = "graph_features",
) -> Dict[str, Any]:
    features = np.asarray(split_arrays[feature_key], dtype=np.float64)
    targets = np.asarray(split_arrays["targets"], dtype=np.float64)
    target_mask = np.asarray(split_arrays["target_mask"], dtype=bool)
    logits = model.predict_logits(features)
    probs = model.predict_proba(features)
    metrics = compute_masked_multilabel_metrics(
        probabilities=probs,
        logits=logits,
        targets=targets,
        target_mask=target_mask,
        swc_ids=swc_ids,
        threshold=threshold,
        per_swc_thresholds=per_swc_thresholds,
        split_name=split_name,
    )
    return {
        "metrics": metrics,
        "rows": int(features.shape[0]),
        "contract_ids": [str(v) for v in split_arrays["contract_ids"].tolist()],
        "probabilities": probs,
        "targets": targets,
        "target_mask": target_mask,
    }


def _run_opcodegt_experiment(
    *,
    run: Mapping[str, Any],
    config: Phase6RunConfig,
    datasets: Mapping[str, Phase4Dataset],
    split_indices: Mapping[str, Sequence[int]],
    device: torch.device,
) -> Dict[str, Any]:
    _set_seed(int(run["seed"]))
    mode = str(run.get("mode", run["model_variant"]))
    dataset_train = datasets["train"]
    feature_dims = dataset_train.feature_shapes()
    train_subset = _to_subset(dataset_train, split_indices["train"])

    if config.use_weighted_sampler:
        _wsampler = build_weighted_sampler(
            dataset_train,
            swc_ids=config.swc_ids,
            indices=split_indices["train"],
            max_oversample_ratio=config.max_oversample_ratio,
            seed=int(run["seed"]),
        )
        train_loader = DataLoader(
            train_subset,
            batch_size=min(config.batch_size, len(train_subset)),
            sampler=_wsampler,
            num_workers=config.num_workers,
            collate_fn=phase4_collate_fn,
        )
    else:
        train_loader = DataLoader(
            train_subset,
            batch_size=min(config.batch_size, len(train_subset)),
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=phase4_collate_fn,
        )
    train_iter = iter(train_loader)

    model_cfg = OpcodeGTConfig(
        mode=mode,
        opcode_input_dim=int(feature_dims["opcode_dim"]),
        graph_input_dim=int(feature_dims["graph_dim"]),
        hidden_dim=int(config.opcodegt.hidden_dim),
        num_labels=int(feature_dims["target_dim"]),
        dropout=float(config.opcodegt.dropout),
        use_gnn=dataset_train.has_graph_artifacts and mode in {"graph_only", "fused"},
        num_gnn_layers=int(config.opcodegt.num_gnn_layers),
        architecture=str(config.opcodegt.architecture),
        use_cross_attention=bool(config.opcodegt.use_cross_attention),
        use_hmpgt=bool(config.opcodegt.use_hmpgt),
        use_label_attention=bool(config.opcodegt.use_label_attention),
        cfg_only=bool(config.opcodegt.cfg_only),
        num_heads=int(config.opcodegt.num_heads),
        num_edge_types=int(config.opcodegt.num_edge_types),
    )
    # Apply ablation variant overrides if present
    if run.get("is_ablation") == "true" and config.ablation_variants:
        variant_name = str(run["model_variant"])
        for ablation in config.ablation_variants:
            if str(ablation.get("name")) == variant_name:
                overrides: Dict[str, Any] = {}
                if "architecture" in ablation:
                    overrides["architecture"] = str(ablation["architecture"])
                if "use_hmpgt" in ablation:
                    overrides["use_hmpgt"] = bool(ablation["use_hmpgt"])
                if "use_cross_attention" in ablation:
                    overrides["use_cross_attention"] = bool(ablation["use_cross_attention"])
                if "use_label_attention" in ablation:
                    overrides["use_label_attention"] = bool(ablation["use_label_attention"])
                if "cfg_only" in ablation:
                    is_cfg_only = bool(ablation["cfg_only"])
                    overrides["cfg_only"] = is_cfg_only
                    if is_cfg_only:
                        overrides["use_gnn"] = True
                if overrides:
                    from dataclasses import replace as _dc_replace
                    model_cfg = _dc_replace(model_cfg, **overrides)
                break
    model = build_opcodegt(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_steps, eta_min=1e-6
    )

    # Compute per-SWC pos_weight from training subset
    _targets_list, _masks_list = [], []
    for idx in split_indices["train"]:
        item = dataset_train[idx]
        _targets_list.append(item["targets"])
        _masks_list.append(item["target_mask"])
    pos_weight = compute_pos_weight(
        torch.stack(_targets_list),
        torch.stack(_masks_list),
        max_pos_weight=config.pos_weight_cap,
    ).to(device)

    losses: List[float] = []

    # Early stopping setup
    stopper = EarlyStopper(patience=config.patience, mode="max") if config.early_stopping else None
    actual_steps = 0
    import time as _time
    run_start_time = _time.monotonic()

    for step_idx in range(1, config.max_steps + 1):
        # Timeout check
        if _time.monotonic() - run_start_time > config.run_timeout_seconds:
            break

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
        
        gnn_inputs = _extract_gnn_inputs(batch, device)
        if getattr(model, "_accepts_edge_type_cache", None) is None:
            import inspect
            model._accepts_edge_type_cache = "graph_edge_type" in inspect.signature(model.forward).parameters
        
        if not model._accepts_edge_type_cache and "graph_edge_type" in gnn_inputs:
            del gnn_inputs["graph_edge_type"]

        logits = model(opcode_features=opcode_features, graph_features=graph_features,
                       **gnn_inputs)
        loss = masked_bce_with_logits(logits=logits, targets=targets, target_mask=target_mask,
                                      reduction="mean", pos_weight=pos_weight)
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss for run `{run['run_id']}` at step {step_idx}.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.item()))
        actual_steps = step_idx

        # Periodic validation for early stopping
        if stopper is not None and step_idx % config.eval_every == 0:
            model.eval()
            with torch.no_grad():
                val_eval = _evaluate_neural_model(
                    model=model,
                    dataset=datasets["val"],
                    indices=split_indices["val"],
                    split_name="val",
                    device=device,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    swc_ids=config.swc_ids,
                    threshold=config.decision_threshold,
                )
            val_f1 = float(val_eval["metrics"]["macro_f1"])
            should_stop = stopper.step(val_f1, step_idx, model.state_dict())
            if should_stop:
                break

    # Restore best model if early stopping was used
    if stopper is not None and stopper.best_state is not None:
        stopper.restore_best(model)

    metrics_by_split: Dict[str, Any] = {}
    tuned_thresholds: Optional[List[float]] = None

    # First pass: evaluate val with default threshold to get raw arrays for tuning
    val_result = _evaluate_neural_model(
        model=model,
        dataset=datasets["val"],
        indices=split_indices["val"],
        split_name="val",
        device=device,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        swc_ids=config.swc_ids,
        threshold=config.decision_threshold,
    )
    # Optimize per-SWC thresholds on validation set
    thr_info = optimize_per_swc_thresholds(
        probabilities=val_result["probabilities"],
        targets=val_result["targets"],
        target_mask=val_result["target_mask"],
        swc_ids=config.swc_ids,
    )
    tuned_thresholds = thr_info["thresholds"]

    # Second pass: evaluate all splits with tuned thresholds
    for split in REQUIRED_SPLITS:
        eval_result = _evaluate_neural_model(
            model=model,
            dataset=datasets[split],
            indices=split_indices[split],
            split_name=split,
            device=device,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            swc_ids=config.swc_ids,
            threshold=config.decision_threshold,
            per_swc_thresholds=tuned_thresholds,
        )
        metrics_by_split[split] = eval_result["metrics"]

    checkpoint_path = (config.models_dir / f"{run['run_id']}.pt").resolve()
    checkpoint_payload = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": str(run["run_id"]),
        "seed": int(run["seed"]),
        "model_group": "opcodegt",
        "model_variant": mode,
        "config": {
            "mode": mode,
            "opcode_input_dim": int(model_cfg.opcode_input_dim),
            "graph_input_dim": int(model_cfg.graph_input_dim),
            "hidden_dim": int(model_cfg.hidden_dim),
            "num_labels": int(model_cfg.num_labels),
            "dropout": float(model_cfg.dropout),
        },
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_losses": losses,
        "tuned_thresholds": tuned_thresholds,
        "threshold_info": thr_info["per_swc"],
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload, checkpoint_path)

    return {
        "checkpoint_path": checkpoint_path,
        "training_losses": losses,
        "parameter_count": int(model.parameter_count()),
        "split_metrics": metrics_by_split,
        "tuned_thresholds": tuned_thresholds,
    }


def _run_mlp_baseline_experiment(
    *,
    run: Mapping[str, Any],
    config: Phase6RunConfig,
    datasets: Mapping[str, Phase4Dataset],
    split_indices: Mapping[str, Sequence[int]],
    device: torch.device,
) -> Dict[str, Any]:
    _set_seed(int(run["seed"]))
    baseline_id = str(run["model_variant"])
    mode = str(BASELINE_REGISTRY[baseline_id]["mode"])
    dataset_train = datasets["train"]
    feature_dims = dataset_train.feature_shapes()
    train_subset = _to_subset(dataset_train, split_indices["train"])

    train_loader = DataLoader(
        train_subset,
        batch_size=min(config.batch_size, len(train_subset)),
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=phase4_collate_fn,
    )
    train_iter = iter(train_loader)

    model_cfg = BaselineMLPConfig(
        mode=mode,
        opcode_input_dim=int(feature_dims["opcode_dim"]),
        graph_input_dim=int(feature_dims["graph_dim"]),
        hidden_dim=int(config.mlp.hidden_dim),
        num_labels=int(feature_dims["target_dim"]),
        dropout=float(config.mlp.dropout),
    )
    model = BaselineMLP(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Compute per-SWC pos_weight from training subset (same weighting as OpcodeGT for fair comparison)
    _targets_list, _masks_list = [], []
    for idx in split_indices["train"]:
        item = dataset_train[idx]
        _targets_list.append(item["targets"])
        _masks_list.append(item["target_mask"])
    pos_weight = compute_pos_weight(torch.stack(_targets_list), torch.stack(_masks_list)).to(device)

    losses: List[float] = []

    stopper = EarlyStopper(patience=config.patience, mode="max") if config.early_stopping else None
    import time as _time
    run_start_time = _time.monotonic()

    for step_idx in range(1, config.max_steps + 1):
        if _time.monotonic() - run_start_time > config.run_timeout_seconds:
            break
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
        loss = masked_bce_with_logits(logits=logits, targets=targets, target_mask=target_mask,
                                      reduction="mean", pos_weight=pos_weight)
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss for run `{run['run_id']}`.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.item()))

        if stopper is not None and step_idx % config.eval_every == 0:
            model.eval()
            with torch.no_grad():
                val_eval = _evaluate_neural_model(
                    model=model, dataset=datasets["val"], indices=split_indices["val"],
                    split_name="val", device=device, batch_size=config.batch_size,
                    num_workers=config.num_workers, swc_ids=config.swc_ids,
                    threshold=config.decision_threshold,
                )
            if stopper.step(float(val_eval["metrics"]["macro_f1"]), step_idx, model.state_dict()):
                break

    if stopper is not None and stopper.best_state is not None:
        stopper.restore_best(model)

    metrics_by_split: Dict[str, Any] = {}
    tuned_thresholds: Optional[List[float]] = None

    # Optimize per-SWC thresholds on validation set
    val_result = _evaluate_neural_model(
        model=model,
        dataset=datasets["val"],
        indices=split_indices["val"],
        split_name="val",
        device=device,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        swc_ids=config.swc_ids,
        threshold=config.decision_threshold,
    )
    thr_info = optimize_per_swc_thresholds(
        probabilities=val_result["probabilities"],
        targets=val_result["targets"],
        target_mask=val_result["target_mask"],
        swc_ids=config.swc_ids,
    )
    tuned_thresholds = thr_info["thresholds"]

    for split in REQUIRED_SPLITS:
        eval_result = _evaluate_neural_model(
            model=model,
            dataset=datasets[split],
            indices=split_indices[split],
            split_name=split,
            device=device,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            swc_ids=config.swc_ids,
            threshold=config.decision_threshold,
            per_swc_thresholds=tuned_thresholds,
        )
        metrics_by_split[split] = eval_result["metrics"]

    checkpoint_path = (config.models_dir / f"{run['run_id']}.pt").resolve()
    checkpoint_payload = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": str(run["run_id"]),
        "seed": int(run["seed"]),
        "model_group": "baseline",
        "model_variant": baseline_id,
        "config": {
            "mode": mode,
            "opcode_input_dim": int(model_cfg.opcode_input_dim),
            "graph_input_dim": int(model_cfg.graph_input_dim),
            "hidden_dim": int(model_cfg.hidden_dim),
            "num_labels": int(model_cfg.num_labels),
            "dropout": float(model_cfg.dropout),
        },
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_losses": losses,
        "tuned_thresholds": tuned_thresholds,
        "threshold_info": thr_info["per_swc"],
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload, checkpoint_path)

    return {
        "checkpoint_path": checkpoint_path,
        "training_losses": losses,
        "parameter_count": int(model.parameter_count()),
        "split_metrics": metrics_by_split,
        "tuned_thresholds": tuned_thresholds,
    }


def _run_codebert_baseline_experiment(
    *,
    run: Mapping[str, Any],
    config: Phase6RunConfig,
    datasets: Mapping[str, Phase4Dataset],
    split_indices: Mapping[str, Sequence[int]],
    device: torch.device,
) -> Dict[str, Any]:
    _set_seed(int(run["seed"]))
    dataset_train = datasets["train"]
    feature_dims = dataset_train.feature_shapes()
    train_subset = _to_subset(dataset_train, split_indices["train"])

    train_loader = DataLoader(
        train_subset,
        batch_size=min(config.batch_size, len(train_subset)),
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=phase4_collate_fn,
    )
    train_iter = iter(train_loader)

    model_cfg = CodeBERTClassifierConfig(
        input_dim=int(feature_dims["opcode_dim"]),
        hidden1=512,
        hidden2=256,
        num_labels=int(feature_dims["target_dim"]),
        dropout=float(config.mlp.dropout),
    )
    model = CodeBERTClassifier(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    _targets_list, _masks_list = [], []
    for idx in split_indices["train"]:
        item = dataset_train[idx]
        _targets_list.append(item["targets"])
        _masks_list.append(item["target_mask"])
    pos_weight = compute_pos_weight(torch.stack(_targets_list), torch.stack(_masks_list)).to(device)

    losses: List[float] = []

    stopper = EarlyStopper(patience=config.patience, mode="max") if config.early_stopping else None
    import time as _time
    run_start_time = _time.monotonic()

    for step_idx in range(1, config.max_steps + 1):
        if _time.monotonic() - run_start_time > config.run_timeout_seconds:
            break
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        opcode_features = batch["opcode_features"].to(device=device, dtype=torch.float32)
        targets = batch["targets"].to(device=device, dtype=torch.float32)
        target_mask = batch["target_mask"].to(device=device, dtype=torch.bool)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(opcode_features=opcode_features)
        loss = masked_bce_with_logits(logits=logits, targets=targets, target_mask=target_mask,
                                      reduction="mean", pos_weight=pos_weight)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite loss for run `{}`.".format(run['run_id']))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.item()))

        if stopper is not None and step_idx % config.eval_every == 0:
            model.eval()
            with torch.no_grad():
                val_eval = _evaluate_neural_model(
                    model=model, dataset=datasets["val"], indices=split_indices["val"],
                    split_name="val", device=device, batch_size=config.batch_size,
                    num_workers=config.num_workers, swc_ids=config.swc_ids,
                    threshold=config.decision_threshold,
                )
            if stopper.step(float(val_eval["metrics"]["macro_f1"]), step_idx, model.state_dict()):
                break

    if stopper is not None and stopper.best_state is not None:
        stopper.restore_best(model)

    metrics_by_split: Dict[str, Any] = {}
    tuned_thresholds: Optional[List[float]] = None

    val_result = _evaluate_neural_model(
        model=model,
        dataset=datasets["val"],
        indices=split_indices["val"],
        split_name="val",
        device=device,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        swc_ids=config.swc_ids,
        threshold=config.decision_threshold,
    )
    thr_info = optimize_per_swc_thresholds(
        probabilities=val_result["probabilities"],
        targets=val_result["targets"],
        target_mask=val_result["target_mask"],
        swc_ids=config.swc_ids,
    )
    tuned_thresholds = thr_info["thresholds"]

    for split in REQUIRED_SPLITS:
        eval_result = _evaluate_neural_model(
            model=model,
            dataset=datasets[split],
            indices=split_indices[split],
            split_name=split,
            device=device,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            swc_ids=config.swc_ids,
            threshold=config.decision_threshold,
            per_swc_thresholds=tuned_thresholds,
        )
        metrics_by_split[split] = eval_result["metrics"]

    checkpoint_path = (config.models_dir / "{}.pt".format(run['run_id'])).resolve()
    checkpoint_payload = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": str(run["run_id"]),
        "seed": int(run["seed"]),
        "model_group": "baseline",
        "model_variant": "codebert_classifier",
        "config": {
            "input_dim": int(model_cfg.input_dim),
            "hidden1": int(model_cfg.hidden1),
            "hidden2": int(model_cfg.hidden2),
            "num_labels": int(model_cfg.num_labels),
            "dropout": float(model_cfg.dropout),
        },
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_losses": losses,
        "tuned_thresholds": tuned_thresholds,
        "threshold_info": thr_info["per_swc"],
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload, checkpoint_path)

    return {
        "checkpoint_path": checkpoint_path,
        "training_losses": losses,
        "parameter_count": int(model.parameter_count()),
        "split_metrics": metrics_by_split,
        "tuned_thresholds": tuned_thresholds,
    }


def _run_gcn_baseline_experiment(
    *,
    run: Mapping[str, Any],
    config: Phase6RunConfig,
    datasets: Mapping[str, Phase4Dataset],
    split_indices: Mapping[str, Sequence[int]],
    device: torch.device,
) -> Dict[str, Any]:
    _set_seed(int(run["seed"]))
    dataset_train = datasets["train"]
    feature_dims = dataset_train.feature_shapes()
    train_subset = _to_subset(dataset_train, split_indices["train"])

    train_loader = DataLoader(
        train_subset,
        batch_size=min(config.batch_size, len(train_subset)),
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=phase4_collate_fn,
    )
    train_iter = iter(train_loader)

    model_cfg = GCNBaselineConfig(
        hidden_dim=int(config.mlp.hidden_dim),
        num_labels=int(feature_dims["target_dim"]),
        num_gcn_layers=3,
        dropout=float(config.mlp.dropout),
    )
    model = GCNBaseline(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    _targets_list, _masks_list = [], []
    for idx in split_indices["train"]:
        item = dataset_train[idx]
        _targets_list.append(item["targets"])
        _masks_list.append(item["target_mask"])
    pos_weight = compute_pos_weight(torch.stack(_targets_list), torch.stack(_masks_list)).to(device)

    losses: List[float] = []

    stopper = EarlyStopper(patience=config.patience, mode="max") if config.early_stopping else None
    import time as _time
    run_start_time = _time.monotonic()

    for step_idx in range(1, config.max_steps + 1):
        if _time.monotonic() - run_start_time > config.run_timeout_seconds:
            break
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        gnn_inputs = _extract_gnn_inputs(batch, device)
        targets = batch["targets"].to(device=device, dtype=torch.float32)
        target_mask = batch["target_mask"].to(device=device, dtype=torch.bool)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            graph_x=gnn_inputs["graph_x"],
            graph_edge_index=gnn_inputs["graph_edge_index"],
            graph_batch=gnn_inputs["graph_batch"],
        )
        loss = masked_bce_with_logits(logits=logits, targets=targets, target_mask=target_mask,
                                      reduction="mean", pos_weight=pos_weight)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite loss for run `{}`.".format(run['run_id']))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.item()))

        if stopper is not None and step_idx % config.eval_every == 0:
            model.eval()
            with torch.no_grad():
                val_eval = _evaluate_neural_model(
                    model=model, dataset=datasets["val"], indices=split_indices["val"],
                    split_name="val", device=device, batch_size=config.batch_size,
                    num_workers=config.num_workers, swc_ids=config.swc_ids,
                    threshold=config.decision_threshold,
                )
            if stopper.step(float(val_eval["metrics"]["macro_f1"]), step_idx, model.state_dict()):
                break

    if stopper is not None and stopper.best_state is not None:
        stopper.restore_best(model)

    metrics_by_split: Dict[str, Any] = {}
    tuned_thresholds: Optional[List[float]] = None

    val_result = _evaluate_neural_model(
        model=model,
        dataset=datasets["val"],
        indices=split_indices["val"],
        split_name="val",
        device=device,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        swc_ids=config.swc_ids,
        threshold=config.decision_threshold,
    )
    thr_info = optimize_per_swc_thresholds(
        probabilities=val_result["probabilities"],
        targets=val_result["targets"],
        target_mask=val_result["target_mask"],
        swc_ids=config.swc_ids,
    )
    tuned_thresholds = thr_info["thresholds"]

    for split in REQUIRED_SPLITS:
        eval_result = _evaluate_neural_model(
            model=model,
            dataset=datasets[split],
            indices=split_indices[split],
            split_name=split,
            device=device,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            swc_ids=config.swc_ids,
            threshold=config.decision_threshold,
            per_swc_thresholds=tuned_thresholds,
        )
        metrics_by_split[split] = eval_result["metrics"]

    checkpoint_path = (config.models_dir / "{}.pt".format(run['run_id'])).resolve()
    checkpoint_payload = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": str(run["run_id"]),
        "seed": int(run["seed"]),
        "model_group": "baseline",
        "model_variant": "gcn_baseline",
        "config": {
            "hidden_dim": int(model_cfg.hidden_dim),
            "num_labels": int(model_cfg.num_labels),
            "num_gcn_layers": int(model_cfg.num_gcn_layers),
            "dropout": float(model_cfg.dropout),
        },
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_losses": losses,
        "tuned_thresholds": tuned_thresholds,
        "threshold_info": thr_info["per_swc"],
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload, checkpoint_path)

    return {
        "checkpoint_path": checkpoint_path,
        "training_losses": losses,
        "parameter_count": int(model.parameter_count()),
        "split_metrics": metrics_by_split,
        "tuned_thresholds": tuned_thresholds,
    }


def _run_classical_baseline_experiment(
    *,
    run: Mapping[str, Any],
    config: Phase6RunConfig,
    datasets: Mapping[str, Phase4Dataset],
    split_indices: Mapping[str, Sequence[int]],
) -> Dict[str, Any]:
    seed = int(run["seed"])
    _set_seed(seed)
    classical_cfg = ClassicalBaselineConfig(
        max_iter=int(config.classical.max_iter),
        C=float(config.classical.C),
        solver=str(config.classical.solver),
        random_state=seed,
    )
    model = MaskedOneVsRestLogisticRegression(classical_cfg)
    train_arrays = _collect_arrays_for_classical(datasets["train"], split_indices["train"])
    model.fit(
        features=train_arrays["graph_features"],
        targets=train_arrays["targets"].astype(np.int64),
        target_mask=train_arrays["target_mask"].astype(bool),
    )

    metrics_by_split: Dict[str, Any] = {}
    tuned_thresholds: Optional[List[float]] = None

    # Optimize per-SWC thresholds on validation set
    val_arrays = _collect_arrays_for_classical(datasets["val"], split_indices["val"])
    val_result = _evaluate_classical_model(
        model=model,
        split_arrays=val_arrays,
        split_name="val",
        swc_ids=config.swc_ids,
        threshold=config.decision_threshold,
    )
    thr_info = optimize_per_swc_thresholds(
        probabilities=val_result["probabilities"],
        targets=val_result["targets"],
        target_mask=val_result["target_mask"],
        swc_ids=config.swc_ids,
    )
    tuned_thresholds = thr_info["thresholds"]

    for split in REQUIRED_SPLITS:
        split_arrays = _collect_arrays_for_classical(datasets[split], split_indices[split])
        eval_result = _evaluate_classical_model(
            model=model,
            split_arrays=split_arrays,
            split_name=split,
            swc_ids=config.swc_ids,
            threshold=config.decision_threshold,
            per_swc_thresholds=tuned_thresholds,
        )
        metrics_by_split[split] = eval_result["metrics"]

    checkpoint_path = (config.models_dir / f"{run['run_id']}.pkl").resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint_path)

    return {
        "checkpoint_path": checkpoint_path,
        "training_losses": [],
        "parameter_count": int(model.parameter_count()),
        "split_metrics": metrics_by_split,
        "fit_info": model.fit_info,
        "tuned_thresholds": tuned_thresholds,
    }


def _run_enriched_classical_experiment(
    *,
    run: Mapping[str, Any],
    config: Phase6RunConfig,
    datasets: Mapping[str, Phase4Dataset],
    split_indices: Mapping[str, Sequence[int]],
    enriched_map: Optional[Dict[str, np.ndarray]],
) -> Dict[str, Any]:
    """Run XGBoost or RF baseline with enriched features (graph + TF-IDF + pattern)."""
    seed = int(run["seed"])
    _set_seed(seed)

    variant_id = str(run["model_variant"])
    if variant_id == "classical_xgboost":
        model = MaskedOneVsRestXGBoost(random_state=seed)
    elif variant_id == "classical_rf":
        model = MaskedOneVsRestRandomForest(random_state=seed)
    elif variant_id == "classical_lgbm":
        model = MaskedOneVsRestLightGBM(random_state=seed)
    else:
        raise ValueError(f"Unknown enriched classical variant: {variant_id}")

    train_arrays = _collect_arrays_for_classical(datasets["train"], split_indices["train"])
    train_features = _enrich_classical_arrays(train_arrays, enriched_map)

    model.fit(
        features=train_features,
        targets=train_arrays["targets"].astype(np.int64),
        target_mask=train_arrays["target_mask"].astype(bool),
    )

    metrics_by_split: Dict[str, Any] = {}
    tuned_thresholds: Optional[List[float]] = None

    val_arrays = _collect_arrays_for_classical(datasets["val"], split_indices["val"])
    val_arrays["enriched_features"] = _enrich_classical_arrays(val_arrays, enriched_map)
    val_result = _evaluate_classical_model(
        model=model,
        split_arrays=val_arrays,
        split_name="val",
        swc_ids=config.swc_ids,
        threshold=config.decision_threshold,
        feature_key="enriched_features",
    )
    thr_info = optimize_per_swc_thresholds(
        probabilities=val_result["probabilities"],
        targets=val_result["targets"],
        target_mask=val_result["target_mask"],
        swc_ids=config.swc_ids,
    )
    tuned_thresholds = thr_info["thresholds"]

    for split in REQUIRED_SPLITS:
        split_arrays = _collect_arrays_for_classical(datasets[split], split_indices[split])
        split_arrays["enriched_features"] = _enrich_classical_arrays(split_arrays, enriched_map)
        eval_result = _evaluate_classical_model(
            model=model,
            split_arrays=split_arrays,
            split_name=split,
            swc_ids=config.swc_ids,
            threshold=config.decision_threshold,
            per_swc_thresholds=tuned_thresholds,
            feature_key="enriched_features",
        )
        metrics_by_split[split] = eval_result["metrics"]

    checkpoint_path = (config.models_dir / f"{run['run_id']}.pkl").resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint_path)

    return {
        "checkpoint_path": checkpoint_path,
        "training_losses": [],
        "parameter_count": int(model.parameter_count()),
        "split_metrics": metrics_by_split,
        "fit_info": model.fit_info,
        "tuned_thresholds": tuned_thresholds,
    }


def _run_gat_baseline_experiment(
    *,
    run: Mapping[str, Any],
    config: Phase6RunConfig,
    datasets: Mapping[str, Phase4Dataset],
    split_indices: Mapping[str, Sequence[int]],
    device: torch.device,
) -> Dict[str, Any]:
    """Run GAT baseline (same graph pipeline as GCN, with attention heads)."""
    _set_seed(int(run["seed"]))
    dataset_train = datasets["train"]
    feature_dims = dataset_train.feature_shapes()
    train_subset = _to_subset(dataset_train, split_indices["train"])

    train_loader = DataLoader(
        train_subset,
        batch_size=min(config.batch_size, len(train_subset)),
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=phase4_collate_fn,
    )
    train_iter = iter(train_loader)

    model_cfg = GATBaselineConfig(
        hidden_dim=int(config.mlp.hidden_dim),
        num_labels=int(feature_dims["target_dim"]),
        num_gat_layers=3,
        heads=8,
        dropout=float(config.mlp.dropout),
    )
    model = GATBaseline(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    _targets_list, _masks_list = [], []
    for idx in split_indices["train"]:
        item = dataset_train[idx]
        _targets_list.append(item["targets"])
        _masks_list.append(item["target_mask"])
    pos_weight = compute_pos_weight(torch.stack(_targets_list), torch.stack(_masks_list)).to(device)

    losses: List[float] = []

    stopper = EarlyStopper(patience=config.patience, mode="max") if config.early_stopping else None
    import time as _time
    run_start_time = _time.monotonic()

    for step_idx in range(1, config.max_steps + 1):
        if _time.monotonic() - run_start_time > config.run_timeout_seconds:
            break
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        gnn_inputs = _extract_gnn_inputs(batch, device)
        targets = batch["targets"].to(device=device, dtype=torch.float32)
        target_mask = batch["target_mask"].to(device=device, dtype=torch.bool)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            graph_x=gnn_inputs["graph_x"],
            graph_edge_index=gnn_inputs["graph_edge_index"],
            graph_batch=gnn_inputs["graph_batch"],
        )
        loss = masked_bce_with_logits(logits=logits, targets=targets, target_mask=target_mask,
                                      reduction="mean", pos_weight=pos_weight)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite loss for run `{}`.".format(run['run_id']))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.item()))

        if stopper is not None and step_idx % config.eval_every == 0:
            model.eval()
            with torch.no_grad():
                val_eval = _evaluate_neural_model(
                    model=model, dataset=datasets["val"], indices=split_indices["val"],
                    split_name="val", device=device, batch_size=config.batch_size,
                    num_workers=config.num_workers, swc_ids=config.swc_ids,
                    threshold=config.decision_threshold,
                )
            if stopper.step(float(val_eval["metrics"]["macro_f1"]), step_idx, model.state_dict()):
                break

    if stopper is not None and stopper.best_state is not None:
        stopper.restore_best(model)

    metrics_by_split: Dict[str, Any] = {}
    tuned_thresholds: Optional[List[float]] = None

    val_result = _evaluate_neural_model(
        model=model,
        dataset=datasets["val"],
        indices=split_indices["val"],
        split_name="val",
        device=device,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        swc_ids=config.swc_ids,
        threshold=config.decision_threshold,
    )
    thr_info = optimize_per_swc_thresholds(
        probabilities=val_result["probabilities"],
        targets=val_result["targets"],
        target_mask=val_result["target_mask"],
        swc_ids=config.swc_ids,
    )
    tuned_thresholds = thr_info["thresholds"]

    for split in REQUIRED_SPLITS:
        eval_result = _evaluate_neural_model(
            model=model,
            dataset=datasets[split],
            indices=split_indices[split],
            split_name=split,
            device=device,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            swc_ids=config.swc_ids,
            threshold=config.decision_threshold,
            per_swc_thresholds=tuned_thresholds,
        )
        metrics_by_split[split] = eval_result["metrics"]

    checkpoint_path = (config.models_dir / "{}.pt".format(run['run_id'])).resolve()
    checkpoint_payload = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": str(run["run_id"]),
        "seed": int(run["seed"]),
        "model_group": "baseline",
        "model_variant": "gat_baseline",
        "config": {
            "hidden_dim": int(model_cfg.hidden_dim),
            "num_labels": int(model_cfg.num_labels),
            "num_gat_layers": int(model_cfg.num_gat_layers),
            "heads": int(model_cfg.heads),
            "dropout": float(model_cfg.dropout),
        },
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_losses": losses,
        "tuned_thresholds": tuned_thresholds,
        "threshold_info": thr_info["per_swc"],
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload, checkpoint_path)

    return {
        "checkpoint_path": checkpoint_path,
        "training_losses": losses,
        "parameter_count": int(model.parameter_count()),
        "split_metrics": metrics_by_split,
        "tuned_thresholds": tuned_thresholds,
    }


def _load_opcode_token_map(max_seq_len: int = 1024) -> Dict[str, List[int]]:
    """Pre-tokenize opcode text corpus for BiLSTM baseline."""
    corpus_path = PROJECT_ROOT / "data/features/main_benchmark/opcode_text_corpus.parquet"
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Opcode text corpus not found at {corpus_path}. "
            "Run Phase 3 feature extraction first."
        )
    df = pd.read_parquet(corpus_path, columns=["fp_runtime_unified", "opcode_text"])
    token_map: Dict[str, List[int]] = {}
    for _, row in df.iterrows():
        contract_id = str(row["fp_runtime_unified"])
        text = str(row["opcode_text"]) if pd.notna(row["opcode_text"]) else ""
        token_map[contract_id] = tokenize_opcode_text(text, max_len=max_seq_len)
    return token_map


def _batch_token_ids_from_map(
    contract_ids: Sequence[str],
    token_map: Dict[str, List[int]],
    max_seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Create padded token ID tensor for a batch of contracts."""
    batch_ids = []
    for cid in contract_ids:
        ids = token_map.get(str(cid), [])
        ids = ids[:max_seq_len]
        padded = ids + [PAD_IDX] * (max_seq_len - len(ids))
        batch_ids.append(padded)
    return torch.tensor(batch_ids, dtype=torch.long, device=device)


def _evaluate_bilstm_model(
    *,
    model: torch.nn.Module,
    dataset: Phase4Dataset,
    indices: Sequence[int],
    split_name: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    swc_ids: Sequence[int],
    threshold: float,
    token_map: Dict[str, List[int]],
    max_seq_len: int,
    per_swc_thresholds: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Evaluate BiLSTM model using opcode token sequences."""
    subset = _to_subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=min(batch_size, len(subset)),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=phase4_collate_fn,
    )

    logits_chunks: List[np.ndarray] = []
    targets_chunks: List[np.ndarray] = []
    mask_chunks: List[np.ndarray] = []
    contract_ids_all: List[str] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            contract_ids = [str(cid) for cid in batch["contract_id"]]
            token_ids = _batch_token_ids_from_map(contract_ids, token_map, max_seq_len, device)
            targets = batch["targets"].to(dtype=torch.float32).cpu().numpy()
            target_mask = batch["target_mask"].to(dtype=torch.bool).cpu().numpy()

            logits = model(opcode_token_ids=token_ids)
            logits_np = logits.detach().cpu().numpy().astype(np.float64)

            logits_chunks.append(logits_np)
            targets_chunks.append(targets.astype(np.float64))
            mask_chunks.append(target_mask.astype(bool))
            contract_ids_all.extend(contract_ids)

    logits_all = np.concatenate(logits_chunks, axis=0)
    targets_all = np.concatenate(targets_chunks, axis=0)
    mask_all = np.concatenate(mask_chunks, axis=0)
    probs_all = sigmoid(logits_all)
    metrics = compute_masked_multilabel_metrics(
        probabilities=probs_all,
        logits=logits_all,
        targets=targets_all,
        target_mask=mask_all,
        swc_ids=swc_ids,
        threshold=threshold,
        per_swc_thresholds=per_swc_thresholds,
        split_name=split_name,
    )
    return {
        "metrics": metrics,
        "rows": int(logits_all.shape[0]),
        "contract_ids": contract_ids_all,
        "probabilities": probs_all,
        "targets": targets_all,
        "target_mask": mask_all,
    }


def _run_bilstm_baseline_experiment(
    *,
    run: Mapping[str, Any],
    config: Phase6RunConfig,
    datasets: Mapping[str, Phase4Dataset],
    split_indices: Mapping[str, Sequence[int]],
    device: torch.device,
    token_map: Dict[str, List[int]],
) -> Dict[str, Any]:
    """Run BiLSTM baseline on opcode token sequences."""
    _set_seed(int(run["seed"]))
    dataset_train = datasets["train"]
    feature_dims = dataset_train.feature_shapes()
    train_subset = _to_subset(dataset_train, split_indices["train"])
    max_seq_len = 1024

    train_loader = DataLoader(
        train_subset,
        batch_size=min(config.batch_size, len(train_subset)),
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=phase4_collate_fn,
    )
    train_iter = iter(train_loader)

    model_cfg = BiLSTMBaselineConfig(
        num_labels=int(feature_dims["target_dim"]),
        hidden_dim=int(config.mlp.hidden_dim),
        dropout=float(config.mlp.dropout),
        max_seq_len=max_seq_len,
    )
    model = BiLSTMBaseline(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    _targets_list, _masks_list = [], []
    for idx in split_indices["train"]:
        item = dataset_train[idx]
        _targets_list.append(item["targets"])
        _masks_list.append(item["target_mask"])
    pos_weight = compute_pos_weight(torch.stack(_targets_list), torch.stack(_masks_list)).to(device)

    losses: List[float] = []

    stopper = EarlyStopper(patience=config.patience, mode="max") if config.early_stopping else None
    import time as _time
    run_start_time = _time.monotonic()

    for step_idx in range(1, config.max_steps + 1):
        if _time.monotonic() - run_start_time > config.run_timeout_seconds:
            break
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        contract_ids = [str(cid) for cid in batch["contract_id"]]
        token_ids = _batch_token_ids_from_map(contract_ids, token_map, max_seq_len, device)
        targets = batch["targets"].to(device=device, dtype=torch.float32)
        target_mask = batch["target_mask"].to(device=device, dtype=torch.bool)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(opcode_token_ids=token_ids)
        loss = masked_bce_with_logits(logits=logits, targets=targets, target_mask=target_mask,
                                      reduction="mean", pos_weight=pos_weight)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite loss for run `{}`.".format(run['run_id']))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.item()))

        if stopper is not None and step_idx % config.eval_every == 0:
            model.eval()
            with torch.no_grad():
                val_eval = _evaluate_bilstm_model(
                    model=model, dataset=datasets["val"], indices=split_indices["val"],
                    split_name="val", device=device, batch_size=config.batch_size,
                    num_workers=config.num_workers, swc_ids=config.swc_ids,
                    threshold=config.decision_threshold,
                    token_map=token_map, max_seq_len=max_seq_len,
                )
            if stopper.step(float(val_eval["metrics"]["macro_f1"]), step_idx, model.state_dict()):
                break

    if stopper is not None and stopper.best_state is not None:
        stopper.restore_best(model)

    metrics_by_split: Dict[str, Any] = {}
    tuned_thresholds: Optional[List[float]] = None

    val_result = _evaluate_bilstm_model(
        model=model,
        dataset=datasets["val"],
        indices=split_indices["val"],
        split_name="val",
        device=device,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        swc_ids=config.swc_ids,
        threshold=config.decision_threshold,
        token_map=token_map,
        max_seq_len=max_seq_len,
    )
    thr_info = optimize_per_swc_thresholds(
        probabilities=val_result["probabilities"],
        targets=val_result["targets"],
        target_mask=val_result["target_mask"],
        swc_ids=config.swc_ids,
    )
    tuned_thresholds = thr_info["thresholds"]

    for split in REQUIRED_SPLITS:
        eval_result = _evaluate_bilstm_model(
            model=model,
            dataset=datasets[split],
            indices=split_indices[split],
            split_name=split,
            device=device,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            swc_ids=config.swc_ids,
            threshold=config.decision_threshold,
            per_swc_thresholds=tuned_thresholds,
            token_map=token_map,
            max_seq_len=max_seq_len,
        )
        metrics_by_split[split] = eval_result["metrics"]

    checkpoint_path = (config.models_dir / "{}.pt".format(run['run_id'])).resolve()
    checkpoint_payload = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": str(run["run_id"]),
        "seed": int(run["seed"]),
        "model_group": "baseline",
        "model_variant": "bilstm_baseline",
        "config": {
            "hidden_dim": int(model_cfg.hidden_dim),
            "num_labels": int(model_cfg.num_labels),
            "num_layers": int(model_cfg.num_layers),
            "embed_dim": int(model_cfg.embed_dim),
            "vocab_size": int(model_cfg.vocab_size),
            "max_seq_len": int(model_cfg.max_seq_len),
            "dropout": float(model_cfg.dropout),
        },
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_losses": losses,
        "tuned_thresholds": tuned_thresholds,
        "threshold_info": thr_info["per_swc"],
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload, checkpoint_path)

    return {
        "checkpoint_path": checkpoint_path,
        "training_losses": losses,
        "parameter_count": int(model.parameter_count()),
        "split_metrics": metrics_by_split,
        "tuned_thresholds": tuned_thresholds,
    }


def _execute_single_run(
    *,
    run: Mapping[str, Any],
    config: Phase6RunConfig,
    datasets: Mapping[str, Phase4Dataset],
    variant_manifest: Mapping[str, Any],
    device: torch.device,
    enriched_map: Optional[Dict[str, np.ndarray]] = None,
    token_map: Optional[Dict[str, List[int]]] = None,
) -> Dict[str, Any]:
    allowed_ids = set(str(x) for x in variant_manifest["graph_id_set"])
    split_indices = {split: _subset_indices_for_ids(datasets[split], allowed_ids) for split in REQUIRED_SPLITS}
    for split in REQUIRED_SPLITS:
        expected_count = int(variant_manifest["split_counts"].get(split, 0))
        observed_count = int(len(split_indices[split]))
        if observed_count != expected_count:
            raise ValueError(
                f"Variant `{run['dataset_variant']}` split count mismatch for `{split}`: "
                f"manifest={expected_count}, observed={observed_count}"
            )

    if len(split_indices["train"]) == 0:
        raise RunUnavailableError(
            f"Variant `{run['dataset_variant']}` has zero training rows for run `{run['run_id']}`."
        )
    if len(split_indices["val"]) == 0 or len(split_indices["test"]) == 0:
        raise RunUnavailableError(
            f"Variant `{run['dataset_variant']}` has zero val/test rows for run `{run['run_id']}`."
        )

    if str(run["model_group"]) == "opcodegt":
        output = _run_opcodegt_experiment(
            run=run,
            config=config,
            datasets=datasets,
            split_indices=split_indices,
            device=device,
        )
    elif str(run["model_variant"]) == "classical_graph_lr":
        output = _run_classical_baseline_experiment(
            run=run,
            config=config,
            datasets=datasets,
            split_indices=split_indices,
        )
    elif str(run["model_variant"]) in ("classical_xgboost", "classical_rf", "classical_lgbm"):
        output = _run_enriched_classical_experiment(
            run=run,
            config=config,
            datasets=datasets,
            split_indices=split_indices,
            enriched_map=enriched_map,
        )
    elif str(run["model_variant"]) == "codebert_classifier":
        output = _run_codebert_baseline_experiment(
            run=run,
            config=config,
            datasets=datasets,
            split_indices=split_indices,
            device=device,
        )
    elif str(run["model_variant"]) == "gcn_baseline":
        output = _run_gcn_baseline_experiment(
            run=run,
            config=config,
            datasets=datasets,
            split_indices=split_indices,
            device=device,
        )
    elif str(run["model_variant"]) == "gat_baseline":
        output = _run_gat_baseline_experiment(
            run=run,
            config=config,
            datasets=datasets,
            split_indices=split_indices,
            device=device,
        )
    elif str(run["model_variant"]) == "bilstm_baseline":
        if token_map is None:
            token_map = _load_opcode_token_map(max_seq_len=1024)
        output = _run_bilstm_baseline_experiment(
            run=run,
            config=config,
            datasets=datasets,
            split_indices=split_indices,
            device=device,
            token_map=token_map,
        )
    else:
        output = _run_mlp_baseline_experiment(
            run=run,
            config=config,
            datasets=datasets,
            split_indices=split_indices,
            device=device,
        )

    metadata_summary = {
        split: _metadata_summary_for_indices(datasets[split], split_indices[split]) for split in REQUIRED_SPLITS
    }
    return {
        "split_indices": {split: list(map(int, idxs)) for split, idxs in split_indices.items()},
        "split_counts": {split: int(len(split_indices[split])) for split in REQUIRED_SPLITS},
        "metadata_summary": metadata_summary,
        **output,
    }


def _write_run_metrics(
    *,
    run: Mapping[str, Any],
    run_output: Mapping[str, Any],
    config: Phase6RunConfig,
    variant_manifest: Mapping[str, Any],
) -> Path:
    metrics_path = (config.metrics_dir / f"{run['run_id']}.json").resolve()
    metrics_payload: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": str(run["run_id"]),
        "model_group": str(run["model_group"]),
        "model_variant": str(run["model_variant"]),
        "dataset_variant": str(run["dataset_variant"]),
        "seed": int(run["seed"]),
        "main_benchmark_swcs": list(config.swc_ids),
        "variant_manifest": {
            "path": _rel(Path(variant_manifest["path"]).resolve()),
            "description": str(variant_manifest["description"]),
            "graph_count": int(variant_manifest["graph_count"]),
            "split_counts": {split: int(variant_manifest["split_counts"].get(split, 0)) for split in REQUIRED_SPLITS},
        },
        "split_counts": dict(run_output["split_counts"]),
        "metadata_summary": dict(run_output["metadata_summary"]),
        "parameter_count": int(run_output.get("parameter_count", 0)),
        "training_losses": [float(x) for x in run_output.get("training_losses", [])],
        "fit_info": run_output.get("fit_info", []),
        "split_metrics": dict(run_output["split_metrics"]),
        "checkpoint_path": _rel(Path(run_output["checkpoint_path"]).resolve()),
    }
    _write_json(metrics_payload, metrics_path)
    return metrics_path


def _run_matches_filters(
    run: Mapping[str, Any],
    *,
    filter_variants: Optional[Set[str]],
    filter_model_groups: Optional[Set[str]],
    filter_model_variants: Optional[Set[str]],
    filter_seeds: Optional[Set[int]],
) -> bool:
    if filter_variants is not None and str(run["dataset_variant"]) not in filter_variants:
        return False
    if filter_model_groups is not None and str(run["model_group"]) not in filter_model_groups:
        return False
    if filter_model_variants is not None and str(run["model_variant"]) not in filter_model_variants:
        return False
    if filter_seeds is not None and int(run["seed"]) not in filter_seeds:
        return False
    return True


def run_experiment_matrix(
    config_path: Path = DEFAULT_PHASE6_CONFIG_PATH,
    *,
    selected_variants: Optional[Sequence[str]] = None,
    selected_model_groups: Optional[Sequence[str]] = None,
    selected_model_variants: Optional[Sequence[str]] = None,
    selected_seeds: Optional[Sequence[int]] = None,
    max_runs: Optional[int] = None,
    resume_override: Optional[bool] = None,
    rerun_failed_override: Optional[bool] = None,
) -> Dict[str, Any]:
    resolved_config_path = config_path.resolve()
    config = _load_config(resolved_config_path)
    device = _resolve_device(config.device)

    variant_manifests = _load_variant_manifests(config)
    model_specs = _build_model_specs(config)
    matrix_runs = _build_matrix_runs(config, model_specs)

    use_resume = config.resume if resume_override is None else bool(resume_override)
    rerun_failed = config.rerun_failed if rerun_failed_override is None else bool(rerun_failed_override)

    manifest = _initialize_or_resume_manifest(
        config=config,
        config_path=resolved_config_path,
        matrix_runs=matrix_runs,
        variant_manifests=variant_manifests,
        resume=use_resume,
    )
    manifest_path = config.run_manifest_json_path.resolve()

    dataset_all = Phase4Dataset(
        config.phase4_config_path,
        feature_index_path_override=config.feature_index_path_override,
    )
    dataset_swc_ids = [int(column.split("_")[1]) for column in dataset_all.target_columns]
    if dataset_swc_ids != config.swc_ids:
        raise ValueError(
            "SWC order mismatch between Phase 6 config and Phase 4 dataset. "
            f"config={config.swc_ids}, dataset={dataset_swc_ids}"
        )
    datasets = {
        split: Phase4Dataset(
            config.phase4_config_path,
            split=split,
            feature_index_path_override=config.feature_index_path_override,
        ) for split in REQUIRED_SPLITS
    }

    config.metrics_dir.mkdir(parents=True, exist_ok=True)
    config.models_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_root.mkdir(parents=True, exist_ok=True)

    # Load enriched features for XGBoost/RF/LightGBM baselines (if configured)
    enriched_map = _load_enriched_feature_map(config)

    # Lazily load opcode token map for BiLSTM (only if BiLSTM is in the baseline list)
    token_map: Optional[Dict[str, List[int]]] = None
    if "bilstm_baseline" in config.baseline_ids:
        token_map = _load_opcode_token_map(max_seq_len=1024)

    filter_variants = set(str(v) for v in selected_variants) if selected_variants else None
    filter_groups = set(str(v) for v in selected_model_groups) if selected_model_groups else None
    filter_models = set(str(v) for v in selected_model_variants) if selected_model_variants else None
    filter_seed_values = set(int(v) for v in selected_seeds) if selected_seeds else None

    executed = 0
    skipped_completed = 0
    skipped_filtered = 0
    skipped_failed = 0
    failed = 0
    unavailable = 0

    runs = manifest["runs"]
    for idx, run in enumerate(runs):
        if not _run_matches_filters(
            run,
            filter_variants=filter_variants,
            filter_model_groups=filter_groups,
            filter_model_variants=filter_models,
            filter_seeds=filter_seed_values,
        ):
            skipped_filtered += 1
            continue
        if max_runs is not None and executed >= int(max_runs):
            break

        status = str(run.get("status", "pending"))
        if use_resume and status == "completed":
            skipped_completed += 1
            continue
        if status == "failed" and not rerun_failed:
            skipped_failed += 1
            continue
        if status == "unavailable" and not rerun_failed:
            skipped_failed += 1
            continue

        run["status"] = "running"
        run["attempts"] = int(run.get("attempts", 0)) + 1
        run["started_utc"] = datetime.now(timezone.utc).isoformat()
        run["finished_utc"] = None
        run["error"] = None
        run["traceback"] = None
        _persist_manifest(manifest, manifest_path)

        variant_name = str(run["dataset_variant"])
        variant_manifest = variant_manifests[variant_name]
        try:
            run_output = _execute_single_run(
                run=run,
                config=config,
                datasets=datasets,
                variant_manifest=variant_manifest,
                device=device,
                enriched_map=enriched_map,
                token_map=token_map,
            )
            metrics_path = _write_run_metrics(
                run=run,
                run_output=run_output,
                config=config,
                variant_manifest=variant_manifest,
            )
            test_metrics = run_output["split_metrics"]["test"]
            run["status"] = "completed"
            run["finished_utc"] = datetime.now(timezone.utc).isoformat()
            run["metrics_json"] = _rel(metrics_path)
            run["checkpoint_path"] = _rel(Path(run_output["checkpoint_path"]).resolve())
            run["summary"] = {
                "test_macro_f1": float(test_metrics["macro_f1"]),
                "test_micro_f1": float(test_metrics["micro_f1"]),
                "test_multilabel_loss": float(test_metrics["multilabel_loss"]),
                "test_macro_precision": float(test_metrics.get("macro_precision", 0.0)),
                "test_macro_recall": float(test_metrics.get("macro_recall", 0.0)),
                "test_macro_mcc": float(test_metrics.get("macro_mcc", 0.0)),
                "test_macro_accuracy": float(test_metrics.get("macro_accuracy", 0.0)),
                "test_subset_accuracy": float(test_metrics.get("subset_accuracy", 0.0)),
                "test_macro_average_precision": (
                    float(test_metrics["macro_average_precision"])
                    if test_metrics["macro_average_precision"] is not None
                    else None
                ),
                "test_micro_average_precision": (
                    float(test_metrics["micro_average_precision"])
                    if test_metrics["micro_average_precision"] is not None
                    else None
                ),
                "split_counts": dict(run_output["split_counts"]),
            }
            executed += 1
        except RunUnavailableError as exc:
            run["status"] = "unavailable"
            run["finished_utc"] = datetime.now(timezone.utc).isoformat()
            run["error"] = str(exc)
            run["traceback"] = None
            unavailable += 1
            executed += 1
        except (torch.cuda.OutOfMemoryError if hasattr(torch.cuda, 'OutOfMemoryError') else RuntimeError) as exc:
            if "out of memory" in str(exc).lower() or isinstance(exc, getattr(torch.cuda, 'OutOfMemoryError', type(None))):
                run["status"] = "failed"
                run["finished_utc"] = datetime.now(timezone.utc).isoformat()
                run["error"] = f"CUDA OOM: {exc}"
                run["traceback"] = traceback.format_exc(limit=10)
                failed += 1
                executed += 1
                _cleanup_gpu()
            else:
                raise
        except Exception as exc:  # noqa: BLE001 - explicit run-level failure capture
            run["status"] = "failed"
            run["finished_utc"] = datetime.now(timezone.utc).isoformat()
            run["error"] = f"{type(exc).__name__}: {exc}"
            run["traceback"] = traceback.format_exc(limit=20)
            failed += 1
            executed += 1
            if config.fail_fast:
                _persist_manifest(manifest, manifest_path)
                raise
        finally:
            runs[idx] = run
            _persist_manifest(manifest, manifest_path)
            _cleanup_gpu()

    status_counts = _summarize_run_statuses(runs)
    summary = {
        "executed_runs": int(executed),
        "skipped_completed": int(skipped_completed),
        "skipped_filtered": int(skipped_filtered),
        "skipped_failed_or_unavailable": int(skipped_failed),
        "new_failed_runs": int(failed),
        "new_unavailable_runs": int(unavailable),
        "status_counts": status_counts,
        "run_manifest_json": _rel(manifest_path),
        "matrix_expected_runs": int(len(runs)),
    }
    return summary


def _parse_optional_csv(values: Optional[Sequence[str]]) -> Optional[List[str]]:
    if values is None:
        return None
    flattened: List[str] = []
    for raw in values:
        for piece in str(raw).split(","):
            value = piece.strip()
            if value:
                flattened.append(value)
    return flattened or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 6 experiment matrix with resumable manifests.")
    parser.add_argument("--config", type=Path, default=DEFAULT_PHASE6_CONFIG_PATH)
    parser.add_argument("--dataset-variant", action="append", dest="dataset_variants")
    parser.add_argument("--model-group", action="append", dest="model_groups")
    parser.add_argument("--model-variant", action="append", dest="model_variants")
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true", help="Ignore previous completed runs.")
    parser.add_argument("--rerun-failed", action="store_true", help="Rerun runs currently marked failed/unavailable.")
    args = parser.parse_args()

    dataset_variants = _parse_optional_csv(args.dataset_variants)
    model_groups = _parse_optional_csv(args.model_groups)
    model_variants = _parse_optional_csv(args.model_variants)

    summary = run_experiment_matrix(
        config_path=args.config.resolve(),
        selected_variants=dataset_variants,
        selected_model_groups=model_groups,
        selected_model_variants=model_variants,
        selected_seeds=args.seeds,
        max_runs=args.max_runs,
        resume_override=not args.no_resume,
        rerun_failed_override=True if args.rerun_failed else None,
    )

    print(f"executed_runs: {summary['executed_runs']}")
    print(f"status_counts: {summary['status_counts']}")
    print(f"run_manifest: {summary['run_manifest_json']}")


if __name__ == "__main__":
    main()
