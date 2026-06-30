"""GCN baseline for smart contract vulnerability detection.

Represents GCN-based approaches from the literature (e.g. DR-GCN, Peculiar,
and other graph-based smart contract vulnerability detectors).
Operates on actual graph structure via GCNConv layers.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool

from src.models.gnn_branch import OpcodeEmbedding


@dataclass(frozen=True)
class GCNBaselineConfig:
    hidden_dim: int
    num_labels: int
    num_gcn_layers: int = 3
    dropout: float = 0.3
    num_opcodes: int = 256


class GCNBaseline(nn.Module):
    """Standard GCN baseline over contract CFG graph structure.

    Architecture:
        node features → OpcodeEmbedding → 3 × GCNConv + ReLU + Dropout
        → global_mean_pool → Linear classifier
    """

    def __init__(self, config: GCNBaselineConfig):
        super().__init__()
        if config.hidden_dim <= 0:
            raise ValueError("`hidden_dim` must be positive.")
        if config.num_labels <= 0:
            raise ValueError("`num_labels` must be positive.")
        if config.num_gcn_layers <= 0:
            raise ValueError("`num_gcn_layers` must be positive.")
        if config.dropout < 0.0 or config.dropout >= 1.0:
            raise ValueError("`dropout` must be in [0.0, 1.0).")

        self.config = GCNBaselineConfig(
            hidden_dim=int(config.hidden_dim),
            num_labels=int(config.num_labels),
            num_gcn_layers=int(config.num_gcn_layers),
            dropout=float(config.dropout),
            num_opcodes=int(config.num_opcodes),
        )

        self.node_embed = OpcodeEmbedding(
            num_opcodes=self.config.num_opcodes,
            opcode_embed_dim=64,
            pc_embed_dim=32,
            output_dim=self.config.hidden_dim,
        )

        self.gcn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(self.config.num_gcn_layers):
            self.gcn_layers.append(GCNConv(self.config.hidden_dim, self.config.hidden_dim))
            self.norms.append(nn.BatchNorm1d(self.config.hidden_dim))

        self.dropout = nn.Dropout(self.config.dropout)
        self.classifier = nn.Linear(self.config.hidden_dim, self.config.num_labels)

    def forward(
        self,
        opcode_features: Optional[torch.Tensor] = None,
        graph_features: Optional[torch.Tensor] = None,
        graph_x: Optional[torch.Tensor] = None,
        graph_edge_index: Optional[torch.Tensor] = None,
        graph_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass using graph structure (graph_x, graph_edge_index, graph_batch)."""
        if graph_x is None or graph_edge_index is None:
            raise ValueError(
                "GCN baseline requires `graph_x` and `graph_edge_index`. "
                "Ensure the dataset has graph artifacts available."
            )
        if graph_batch is None:
            graph_batch = torch.zeros(graph_x.size(0), dtype=torch.long, device=graph_x.device)

        h = self.node_embed(graph_x)

        for conv, norm in zip(self.gcn_layers, self.norms):
            h = conv(h, graph_edge_index)
            h = norm(h)
            h = torch.relu(h)
            h = self.dropout(h)

        h = global_mean_pool(h, graph_batch)
        logits = self.classifier(h)
        return logits

    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def branch_dims(self) -> Dict[str, int]:
        return {
            "hidden_dim": int(self.config.hidden_dim),
            "num_labels": int(self.config.num_labels),
            "num_gcn_layers": int(self.config.num_gcn_layers),
        }
