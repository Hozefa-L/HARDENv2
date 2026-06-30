"""Per-SWC weighted random sampler for class-imbalanced multi-label training.

Builds a WeightedRandomSampler that upsamples training contracts holding
rare positive labels (e.g. SWC-115, SWC-128) relative to their
inverse-frequency weight. Contracts with no positives across all assessed
SWCs receive weight 1.0 (base rate).

Usage::

    from src.training.balanced_sampler import build_weighted_sampler
    from torch.utils.data import DataLoader

    sampler = build_weighted_sampler(dataset_train, swc_ids)
    loader = DataLoader(dataset_train, batch_size=32, sampler=sampler)
"""

import logging
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

logger = logging.getLogger(__name__)


def build_weighted_sampler(
    dataset,
    swc_ids: Sequence[int],
    *,
    indices: Optional[Sequence[int]] = None,
    max_oversample_ratio: float = 20.0,
    seed: Optional[int] = None,
) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler for multi-label class imbalance correction.

    For each training sample the sample weight is the maximum across all
    positive, assessed SWC labels of that label's inverse class frequency
    (negatives / positives across the full training set).  Contracts that
    carry no positive labels get weight 1.0 so they are still drawn but at
    the base rate.

    Args:
        dataset: A Phase4Dataset instance (must be the training split).
        swc_ids: Ordered list of SWC IDs matching dataset.target_columns.
        indices: Optional list of indices within the dataset to use.
            Required for data efficiency experiments (subsets).
        max_oversample_ratio: Upper cap on any single label's weight to
            prevent extreme oversampling. Default 20.0.
        seed: Optional RNG seed for reproducibility of the sampler.

    Returns:
        WeightedRandomSampler that can replace shuffle=True in DataLoader.
    """
    target_columns = [f"swc_{sid}" for sid in swc_ids]
    mask_columns = [f"swc_{sid}_assessed" for sid in swc_ids]

    n = len(indices) if indices is not None else len(dataset)
    if n == 0:
        raise ValueError("Dataset/Subset is empty; cannot build weighted sampler.")

    frame = dataset.frame
    if indices is not None:
        frame = frame.iloc[list(indices)]
    missing = [c for c in target_columns + mask_columns if c not in frame.columns]
    if missing:
        raise ValueError(f"Dataset frame missing columns: {missing}")

    # ---- Compute per-label inverse-frequency weights from training data ----
    label_weights: List[float] = []
    for lbl_col, msk_col in zip(target_columns, mask_columns):
        is_assessed = frame[msk_col].fillna(False).astype(bool)
        labels_assessed = frame.loc[is_assessed, lbl_col].fillna(0).astype(int)
        n_pos = int((labels_assessed == 1).sum())
        n_neg = int((labels_assessed == 0).sum())
        if n_pos == 0:
            # No positives at all → no upsampling needed for this label
            label_weights.append(1.0)
        else:
            w = float(n_neg) / float(n_pos)
            w = min(w, max_oversample_ratio)
            label_weights.append(w)

    label_weight_arr = np.array(label_weights, dtype=np.float64)
    logger.info(
        "Per-SWC label inverse-freq weights (capped at %.1f): %s",
        max_oversample_ratio,
        dict(zip(swc_ids, label_weight_arr.round(2).tolist())),
    )

    # ---- Compute per-sample weight = max weight over its positive labels ----
    sample_weights = np.ones(n, dtype=np.float64)
    for i, (lbl_col, msk_col) in enumerate(zip(target_columns, mask_columns)):
        is_pos = (
            frame[lbl_col].fillna(0).astype(int) == 1
        ) & frame[msk_col].fillna(False).astype(bool)
        sample_weights[is_pos.values] = np.maximum(
            sample_weights[is_pos.values], label_weight_arr[i]
        )

    n_upsampled = int((sample_weights > 1.0).sum())
    logger.info(
        "Weighted sampler: %d / %d training samples will be upsampled "
        "(weight > 1.0). Effective epoch length = %d draws.",
        n_upsampled,
        n,
        n,
    )

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=n,
        replacement=True,
        generator=generator,
    )
