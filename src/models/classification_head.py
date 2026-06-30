"""Label-aware attention classification head for multi-label vulnerability detection.

Provides two classification heads:
- LabelAwareHead: Each SWC vulnerability type gets a learned query vector that
  attends to the fused contract embedding, producing specialized representations.
- SimpleLinearHead: Standard linear projection (ablation baseline).

Factory function ``build_classification_head`` selects the head based on config.
"""

import math
from dataclasses import dataclass
from typing import Union

import torch
from torch import nn

__all__ = [
    "LabelAwareHead",
    "SimpleLinearHead",
    "build_classification_head",
    "ClassificationHeadConfig",
]


@dataclass
class ClassificationHeadConfig:
    """Minimal config consumed by the factory."""

    hidden_dim: int
    num_labels: int
    dropout: float = 0.1
    use_label_attention: bool = True


class LabelAwareHead(nn.Module):
    """Label-aware attention classification head.

    Each vulnerability type owns a learned query vector that attends to the
    projected contract embedding.  The per-label representation is then
    classified independently, enabling label-specific specialization.

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of the input contract embedding.
    num_labels : int
        Number of binary labels (SWC vulnerability types).
    dropout : float
        Dropout probability applied after layer-norm.
    """

    def __init__(self, hidden_dim: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("`hidden_dim` must be positive.")
        if num_labels <= 0:
            raise ValueError("`num_labels` must be positive.")

        self.hidden_dim = hidden_dim
        self.num_labels = num_labels

        # Learned query per vulnerability type — xavier init for stable gradients
        self.label_queries = nn.Parameter(torch.empty(num_labels, hidden_dim))
        nn.init.xavier_uniform_(self.label_queries.unsqueeze(0))  # treat as [1, L, H]

        self.attention_proj = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, contract_embedding: torch.Tensor) -> torch.Tensor:
        """Produce per-label logits via label-aware attention.

        Parameters
        ----------
        contract_embedding : Tensor [B, H]
            Fused (or single-branch) contract representation.

        Returns
        -------
        logits : Tensor [B, L]
            Raw logits (pre-sigmoid) — one per label per sample.
        """
        if contract_embedding.ndim != 2:
            raise ValueError(
                f"Expected rank-2 input [batch, hidden_dim], got shape {tuple(contract_embedding.shape)}"
            )

        # Project embedding for attention computation  [B, H]
        proj = self.attention_proj(contract_embedding)

        # Attention scores: each label query attends to the contract
        # label_queries [L, H] @ proj^T [H, B] → [L, B]
        scale = math.sqrt(self.hidden_dim)
        attn_scores = torch.matmul(self.label_queries, proj.t()) / scale

        # Normalize across labels (each position competes)
        attn_weights = torch.softmax(attn_scores, dim=0)  # [L, B]

        # Label-specific representations:
        # query context  +  attention-weighted contract embedding
        # label_queries [L, H] → [1, L, H]
        # attn_weights^T [B, L] → [B, L, 1]
        # contract_embedding [B, H] → [B, 1, H]
        label_features = (
            self.label_queries.unsqueeze(0)
            + attn_weights.t().unsqueeze(-1) * contract_embedding.unsqueeze(1)
        )  # [B, L, H]

        label_features = self.layer_norm(label_features)
        label_features = self.dropout(label_features)

        # Per-label binary logit  [B, L, 1] → [B, L]
        logits = self.classifier(label_features).squeeze(-1)
        return logits


class SimpleLinearHead(nn.Module):
    """Plain linear classification head (ablation baseline).

    Mirrors the original ``nn.Linear(hidden_dim, num_labels)`` used in OpcodeGT
    with an optional dropout layer.

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of the input contract embedding.
    num_labels : int
        Number of binary labels.
    dropout : float
        Dropout probability applied before the linear layer.
    """

    def __init__(self, hidden_dim: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("`hidden_dim` must be positive.")
        if num_labels <= 0:
            raise ValueError("`num_labels` must be positive.")

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, num_labels)

    def forward(self, contract_embedding: torch.Tensor) -> torch.Tensor:
        """Forward pass — dropout then linear projection.

        Parameters
        ----------
        contract_embedding : Tensor [B, H]

        Returns
        -------
        logits : Tensor [B, num_labels]
        """
        if contract_embedding.ndim != 2:
            raise ValueError(
                f"Expected rank-2 input [batch, hidden_dim], got shape {tuple(contract_embedding.shape)}"
            )
        return self.linear(self.dropout(contract_embedding))


def build_classification_head(
    config: Union[ClassificationHeadConfig, object],
) -> nn.Module:
    """Factory: build classification head from config.

    Parameters
    ----------
    config
        Any object with ``hidden_dim``, ``num_labels``, ``dropout``, and
        ``use_label_attention`` attributes.

    Returns
    -------
    nn.Module
        A ``LabelAwareHead`` if ``config.use_label_attention`` is True,
        otherwise a ``SimpleLinearHead``.
    """
    hidden_dim = int(getattr(config, "hidden_dim"))
    num_labels = int(getattr(config, "num_labels"))
    dropout = float(getattr(config, "dropout", 0.1))
    use_label_attention = bool(getattr(config, "use_label_attention", True))

    if use_label_attention:
        return LabelAwareHead(hidden_dim, num_labels, dropout)
    return SimpleLinearHead(hidden_dim, num_labels, dropout)
