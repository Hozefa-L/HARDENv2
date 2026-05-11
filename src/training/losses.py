from typing import Dict, Optional

import torch
import torch.nn.functional as F


def compute_pos_weight(
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    max_pos_weight: float = 100.0,
) -> torch.Tensor:
    """Compute per-label positive weight for class-imbalance correction.

    Returns a weight vector of shape [num_labels] where each element is
    (num_negatives / num_positives) for that label, clamped to [1, max_pos_weight].

    Args:
        targets: Float tensor [N, num_labels] with 0/1 values.
        target_mask: Bool tensor [N, num_labels]; True = assessed.
        max_pos_weight: Upper bound on any single label's weight (default 100).
            Raised from the original 50 to allow stronger upweighting for
            extremely sparse classes (e.g. SWC-115 at 1:60, SWC-128 at 1:10).
    """
    mask = target_mask.float()
    assessed = mask.sum(dim=0).clamp(min=1.0)
    pos = (targets * mask).sum(dim=0)
    neg = assessed - pos
    weight = (neg / pos.clamp(min=1.0)).clamp(min=1.0, max=float(max_pos_weight))
    return weight


def masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    reduction: str = "mean",
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if logits.shape != targets.shape:
        raise ValueError(f"`logits` and `targets` must share shape. got {tuple(logits.shape)} vs {tuple(targets.shape)}")
    if logits.shape != target_mask.shape:
        raise ValueError(
            f"`target_mask` shape must match logits shape. got {tuple(target_mask.shape)} vs {tuple(logits.shape)}"
        )
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError("`reduction` must be one of: mean, sum, none.")

    mask = target_mask.to(dtype=logits.dtype)
    elementwise = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight,
    )
    masked = elementwise * mask

    if reduction == "none":
        return masked

    denom = mask.sum()
    if float(denom.item()) <= 0.0:
        raise ValueError("No assessed labels in `target_mask`; masked loss denominator is zero.")

    if reduction == "sum":
        return masked.sum()
    return masked.sum() / denom


def masked_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Focal loss with masking for class-imbalanced multilabel classification.

    Focuses training on hard, misclassified examples by down-weighting
    easy examples. Especially useful when most SWC labels are rare.
    """
    if logits.shape != targets.shape or logits.shape != target_mask.shape:
        raise ValueError("Shape mismatch between logits, targets, and target_mask.")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError("`reduction` must be one of: mean, sum, none.")

    mask = target_mask.to(dtype=logits.dtype)
    p = torch.sigmoid(logits)
    # Binary cross-entropy per element
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    # Focal modulating factor
    p_t = targets * p + (1 - targets) * (1 - p)
    focal_weight = (1 - p_t) ** gamma

    # Alpha balancing
    alpha_t = targets * alpha + (1 - targets) * (1 - alpha)

    loss = alpha_t * focal_weight * bce
    masked_loss = loss * mask

    if reduction == "none":
        return masked_loss

    denom = mask.sum()
    if float(denom.item()) <= 0.0:
        raise ValueError("No assessed labels in `target_mask`; masked loss denominator is zero.")

    if reduction == "sum":
        return masked_loss.sum()
    return masked_loss.sum() / denom


def masked_batch_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    if threshold <= 0.0 or threshold >= 1.0:
        raise ValueError("`threshold` must be in (0.0, 1.0).")

    if logits.shape != targets.shape or logits.shape != target_mask.shape:
        raise ValueError("`logits`, `targets`, and `target_mask` must have identical shapes.")

    mask = target_mask.to(dtype=logits.dtype)
    denom = mask.sum()
    if float(denom.item()) <= 0.0:
        raise ValueError("No assessed labels in `target_mask`; cannot compute masked metrics.")

    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).to(dtype=targets.dtype)
    correctness = (predictions == targets).to(dtype=logits.dtype)

    accuracy = float(((correctness * mask).sum() / denom).item())
    positive_rate = float(((predictions.to(dtype=logits.dtype) * mask).sum() / denom).item())
    target_positive_rate = float(((targets.to(dtype=logits.dtype) * mask).sum() / denom).item())

    return {
        "masked_accuracy": accuracy,
        "masked_pred_positive_rate": positive_rate,
        "masked_target_positive_rate": target_positive_rate,
    }

