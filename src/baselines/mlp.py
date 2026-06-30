from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn

VALID_MLP_BASELINE_MODES = {"opcode_only", "graph_only", "fusion_concat"}


@dataclass(frozen=True)
class BaselineMLPConfig:
    mode: str
    opcode_input_dim: int
    graph_input_dim: int
    hidden_dim: int
    num_labels: int
    dropout: float = 0.1


class BaselineMLP(nn.Module):
    """Simple MLP baseline over opcode, graph, or concatenated flat features."""

    def __init__(self, config: BaselineMLPConfig):
        super().__init__()
        mode = str(config.mode).strip().lower()
        if mode not in VALID_MLP_BASELINE_MODES:
            raise ValueError(f"`mode` must be one of {sorted(VALID_MLP_BASELINE_MODES)}.")
        if config.opcode_input_dim <= 0:
            raise ValueError("`opcode_input_dim` must be positive.")
        if config.graph_input_dim <= 0:
            raise ValueError("`graph_input_dim` must be positive.")
        if config.hidden_dim <= 0:
            raise ValueError("`hidden_dim` must be positive.")
        if config.num_labels <= 0:
            raise ValueError("`num_labels` must be positive.")
        if config.dropout < 0.0 or config.dropout >= 1.0:
            raise ValueError("`dropout` must be in [0.0, 1.0).")

        self.config = BaselineMLPConfig(
            mode=mode,
            opcode_input_dim=int(config.opcode_input_dim),
            graph_input_dim=int(config.graph_input_dim),
            hidden_dim=int(config.hidden_dim),
            num_labels=int(config.num_labels),
            dropout=float(config.dropout),
        )

        if self.config.mode == "opcode_only":
            input_dim = int(self.config.opcode_input_dim)
        elif self.config.mode == "graph_only":
            input_dim = int(self.config.graph_input_dim)
        else:
            input_dim = int(self.config.opcode_input_dim + self.config.graph_input_dim)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )
        self.classifier = nn.Linear(self.config.hidden_dim, self.config.num_labels)

    def forward(
        self,
        opcode_features: Optional[torch.Tensor],
        graph_features: Optional[torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        if self.config.mode == "opcode_only":
            if opcode_features is None:
                raise ValueError("`opcode_features` is required for mode `opcode_only`.")
            features = opcode_features
        elif self.config.mode == "graph_only":
            if graph_features is None:
                raise ValueError("`graph_features` is required for mode `graph_only`.")
            features = graph_features
        else:
            if opcode_features is None or graph_features is None:
                raise ValueError("Both `opcode_features` and `graph_features` are required for mode `fusion_concat`.")
            features = torch.cat([opcode_features, graph_features], dim=-1)

        if features.ndim != 2:
            raise ValueError(f"Expected rank-2 feature tensor, got shape {tuple(features.shape)}.")

        hidden = self.encoder(features)
        logits = self.classifier(hidden)
        if logits.ndim != 2:
            raise RuntimeError(f"Classifier produced invalid output shape: {tuple(logits.shape)}.")
        return logits

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters()))

    def branch_dims(self) -> Dict[str, int]:
        return {
            "opcode_input_dim": int(self.config.opcode_input_dim),
            "graph_input_dim": int(self.config.graph_input_dim),
            "hidden_dim": int(self.config.hidden_dim),
            "num_labels": int(self.config.num_labels),
        }

