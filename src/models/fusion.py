from typing import Tuple

import torch
from torch import nn


class GatedFusion(nn.Module):
    """Simple, ablation-friendly gated fusion over two branch embeddings."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("`hidden_dim` must be positive.")
        self.hidden_dim = int(hidden_dim)
        self.gate = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, opcode_embedding: torch.Tensor, graph_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if opcode_embedding.ndim != 2 or graph_embedding.ndim != 2:
            raise ValueError("Fusion inputs must be rank-2 tensors shaped [batch, hidden_dim].")
        if opcode_embedding.shape != graph_embedding.shape:
            raise ValueError(
                "Fusion input shape mismatch: "
                f"opcode={tuple(opcode_embedding.shape)}, graph={tuple(graph_embedding.shape)}"
            )

        joint = torch.cat([opcode_embedding, graph_embedding], dim=-1)
        gate = torch.sigmoid(self.gate(joint))
        fused = (gate * opcode_embedding) + ((1.0 - gate) * graph_embedding)
        return self.norm(fused), gate

