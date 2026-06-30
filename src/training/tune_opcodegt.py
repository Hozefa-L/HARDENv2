"""
Level 6: Optuna Hyperparameter Tuning for OpcodeGT v2 (graph_only mode)

In-process tuning — no subprocess overhead. Each trial trains directly
in the current process, reusing the loaded dataset. Much faster than
spawning a new process per trial.

Usage:
    PYTHONPATH=. python -m src.training.tune_opcodegt --n-trials 30
"""
import argparse
import json
import gc
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import optuna
    from optuna.exceptions import TrialPruned
except ImportError:
    print("Please install optuna to run tuning: `pip install optuna`")
    exit(1)

from src.evaluation.metrics import compute_masked_multilabel_metrics, sigmoid
from src.models.opcodegt import OpcodeGT, OpcodeGTConfig, build_opcodegt
from src.training.losses import masked_bce_with_logits, compute_pos_weight
from src.training.phase4_dataset import Phase4Dataset, phase4_collate_fn

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _extract_gnn_inputs(batch: Dict[str, Any], device: torch.device) -> Dict[str, Optional[torch.Tensor]]:
    """Extract GNN graph inputs from a batch."""
    gx = batch.get("graph_x")
    if gx is None:
        return {"graph_x": None, "graph_edge_index": None, "graph_batch": None, "graph_edge_type": None}
    return {
        "graph_x": gx.to(device),
        "graph_edge_index": batch["graph_edge_index"].to(device),
        "graph_batch": batch["graph_batch"].to(device),
        "graph_edge_type": batch.get("graph_edge_type", torch.zeros(1)).to(device),
    }


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_labels: int,
) -> Dict[str, float]:
    """Evaluate model on a DataLoader, return metrics dict."""
    model.eval()
    logits_all, targets_all, mask_all = [], [], []

    with torch.no_grad():
        for batch in loader:
            opcode = batch["opcode_features"].to(device, dtype=torch.float32)
            graph = batch["graph_features"].to(device, dtype=torch.float32)
            gnn = _extract_gnn_inputs(batch, device)
            logits = model(opcode_features=opcode, graph_features=graph, **gnn)
            logits_all.append(logits.cpu().numpy())
            targets_all.append(batch["targets"].numpy())
            mask_all.append(batch["target_mask"].numpy())

    logits_np = np.concatenate(logits_all, axis=0)
    targets_np = np.concatenate(targets_all, axis=0)
    mask_np = np.concatenate(mask_all, axis=0)
    probs_np = sigmoid(logits_np)

    metrics = compute_masked_multilabel_metrics(
        probabilities=probs_np,
        targets=targets_np,
        target_mask=mask_np,
        threshold=0.5,
        swc_ids=list(range(num_labels)),
    )
    return metrics


def objective(trial: optuna.Trial, dataset_train: Phase4Dataset, dataset_val: Phase4Dataset,
              feature_dims: Dict[str, int], device: torch.device, max_steps: int) -> float:
    """Single tuning trial — trains in-process, no subprocess."""

    # Suggest hyperparameters
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256])
    num_gnn_layers = trial.suggest_int("num_gnn_layers", 2, 4)
    num_heads = trial.suggest_categorical("num_heads", [4, 8])
    dropout = trial.suggest_float("dropout", 0.05, 0.3)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64])

    # Build model (graph_only mode)
    config = OpcodeGTConfig(
        mode="graph_only",
        opcode_input_dim=int(feature_dims["opcode_dim"]),
        graph_input_dim=int(feature_dims["graph_dim"]),
        hidden_dim=hidden_dim,
        num_labels=int(feature_dims["target_dim"]),
        dropout=dropout,
        use_gnn=True,
        num_gnn_layers=num_gnn_layers,
        architecture="opcodegt_v2",
        use_hmpgt=True,
        use_cross_attention=False,
        use_label_attention=True,
        num_heads=num_heads,
    )
    model = build_opcodegt(config).to(device)

    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=phase4_collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=phase4_collate_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop with early stopping and Optuna pruning
    best_val_f1 = 0.0
    patience_counter = 0
    patience = 7
    eval_every = 50
    step = 0

    try:
        while step < max_steps:
            model.train()
            for batch in train_loader:
                if step >= max_steps:
                    break

                opcode = batch["opcode_features"].to(device, dtype=torch.float32)
                graph = batch["graph_features"].to(device, dtype=torch.float32)
                targets = batch["targets"].to(device, dtype=torch.float32)
                target_mask = batch["target_mask"].to(device, dtype=torch.bool)
                gnn = _extract_gnn_inputs(batch, device)

                optimizer.zero_grad(set_to_none=True)
                logits = model(opcode_features=opcode, graph_features=graph, **gnn)
                loss = masked_bce_with_logits(logits=logits, targets=targets,
                                              target_mask=target_mask, reduction="mean")
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite loss")

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                step += 1

                # Periodic validation
                if step % eval_every == 0:
                    val_metrics = _evaluate(model, val_loader, device, int(feature_dims["target_dim"]))
                    val_f1 = val_metrics.get("macro_f1", 0.0)

                    # Report to Optuna for pruning
                    trial.report(val_f1, step)
                    if trial.should_prune():
                        raise TrialPruned()

                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        break  # Early stop

                    model.train()

            if patience_counter >= patience:
                break

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower():
            print(f"Trial {trial.number}: OOM — skipping (params too large for GPU)")
            del model, optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return 0.0
        raise  # Re-raise non-OOM RuntimeErrors
    except ValueError:
        # Non-finite loss = diverged
        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return 0.0

    # Final evaluation
    if best_val_f1 == 0.0:
        val_metrics = _evaluate(model, val_loader, device, int(feature_dims["target_dim"]))
        best_val_f1 = val_metrics.get("macro_f1", 0.0)

    # Cleanup
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print(f"Trial {trial.number}: val_macro_f1={best_val_f1:.4f} "
          f"(hd={hidden_dim}, layers={num_gnn_layers}, heads={num_heads}, "
          f"lr={lr:.2e}, wd={weight_decay:.2e}, do={dropout:.2f}, bs={batch_size}, "
          f"stopped at step {step})")

    return best_val_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser("HARDEN-V2 Level 6 Optuna Tuner (in-process)")
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Number of hyperparameter trials")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Max training steps per trial")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to train on (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for data loading")
    args = parser.parse_args()

    # Resolve device
    device_str = args.device.strip().lower()
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device_str = "cpu"
    device = torch.device(device_str)

    # Load datasets ONCE (reused across all trials)
    print("Loading datasets...")
    dataset_train = Phase4Dataset(split="train")
    dataset_val = Phase4Dataset(split="val")
    feature_dims = dataset_train.feature_shapes()
    print(f"  Train: {len(dataset_train)} samples")
    print(f"  Val:   {len(dataset_val)} samples")
    print(f"  Opcode dim: {feature_dims['opcode_dim']}")
    print(f"  Graph dim:  {feature_dims['graph_dim']}")
    print(f"  Labels:     {feature_dims['target_dim']}")
    print(f"  Max steps:  {args.max_steps}")
    print(f"  Device:     {device}")
    print()

    # SQLite persistence — survives crashes
    db_path = PROJECT_ROOT / "checkpoints" / "optuna_tuning.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        direction="maximize",
        study_name="opcodegt_v2_graph_only",
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=200,
        ),
    )

    try:
        study.optimize(
            lambda trial: objective(
                trial, dataset_train, dataset_val,
                feature_dims, device, args.max_steps,
            ),
            n_trials=args.n_trials,
        )
    except KeyboardInterrupt:
        print("\nTuning interrupted. Progress saved to SQLite.")

    print("\n" + "=" * 50)
    if len(study.trials) > 0 and study.best_trial is not None:
        print(f"Best Macro-F1: {study.best_value:.4f}")
        print("Best Hyperparameters:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

        # Save best params
        out_path = PROJECT_ROOT / "reports/level6_best_hyperparams.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"best_value": study.best_value, **study.best_params}, f, indent=2)
        print(f"\nSaved to {out_path}")
    else:
        print("No completed trials yet.")
    print("=" * 50)
