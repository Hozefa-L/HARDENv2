"""Vanilla CodeBERT classifier baseline.

Represents the 'pretrained LM + linear probe' approach common in
vulnerability detection literature (e.g. SmartBERT, CodeBERT-based detectors).
Takes 768-d CodeBERT CLS embeddings and applies a 3-layer MLP head.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn


@dataclass(frozen=True)
class CodeBERTClassifierConfig:
    input_dim: int
    hidden1: int
    hidden2: int
    num_labels: int
    dropout: float = 0.3


class CodeBERTClassifier(nn.Module):
    """3-layer MLP head over CodeBERT CLS embeddings for multi-label classification."""

    def __init__(self, config: CodeBERTClassifierConfig):
        super().__init__()
        if config.input_dim <= 0:
            raise ValueError("`input_dim` must be positive.")
        if config.hidden1 <= 0:
            raise ValueError("`hidden1` must be positive.")
        if config.hidden2 <= 0:
            raise ValueError("`hidden2` must be positive.")
        if config.num_labels <= 0:
            raise ValueError("`num_labels` must be positive.")
        if config.dropout < 0.0 or config.dropout >= 1.0:
            raise ValueError("`dropout` must be in [0.0, 1.0).")

        self.config = CodeBERTClassifierConfig(
            input_dim=int(config.input_dim),
            hidden1=int(config.hidden1),
            hidden2=int(config.hidden2),
            num_labels=int(config.num_labels),
            dropout=float(config.dropout),
        )

        self.encoder = nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.hidden1),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden1, self.config.hidden2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )
        self.classifier = nn.Linear(self.config.hidden2, self.config.num_labels)

    def forward(
        self,
        opcode_features: Optional[torch.Tensor] = None,
        graph_features: Optional[torch.Tensor] = None,
        graph_x: Optional[torch.Tensor] = None,
        graph_edge_index: Optional[torch.Tensor] = None,
        graph_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass using only opcode (CodeBERT CLS) features."""
        if opcode_features is None:
            raise ValueError("`opcode_features` (CodeBERT CLS embeddings) is required.")
        if opcode_features.ndim != 2:
            raise ValueError(
                "Expected rank-2 opcode_features tensor, got shape {}.".format(
                    tuple(opcode_features.shape)
                )
            )
        hidden = self.encoder(opcode_features)
        logits = self.classifier(hidden)
        return logits

    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def branch_dims(self) -> Dict[str, int]:
        return {
            "input_dim": int(self.config.input_dim),
            "hidden1": int(self.config.hidden1),
            "hidden2": int(self.config.hidden2),
            "num_labels": int(self.config.num_labels),
        }
