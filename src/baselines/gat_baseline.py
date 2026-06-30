"""GAT baseline for smart contract vulnerability detection.

Graph Attention Network baseline using GATConv layers.
Provides a standard attention-based GNN comparison point alongside
the GCN baseline (message-passing) and GIN branch (in OpcodeGT).
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
from torch_geometric.nn import GATConv, global_mean_pool

from src.models.gnn_branch import OpcodeEmbedding


@dataclass(frozen=True)
class GATBaselineConfig:
    hidden_dim: int
    num_labels: int
    num_gat_layers: int = 3
    heads: int = 8
    dropout: float = 0.3
    num_opcodes: int = 256


class GATBaseline(nn.Module):
    """Standard GAT baseline over contract CFG graph structure.

    Architecture:
        node features → OpcodeEmbedding → 3 × [GATConv + ELU + Dropout]
        → global_mean_pool → Linear classifier

    Uses multi-head attention in intermediate layers (concatenated outputs),
    and single-head attention in the final layer (mean aggregation) to produce
    hidden_dim-sized node representations for pooling.
    """

    def __init__(self, config: GATBaselineConfig):
        super().__init__()
        if config.hidden_dim <= 0:
            raise ValueError("`hidden_dim` must be positive.")
        if config.num_labels <= 0:
            raise ValueError("`num_labels` must be positive.")
        if config.num_gat_layers <= 0:
            raise ValueError("`num_gat_layers` must be positive.")
        if config.heads <= 0:
            raise ValueError("`heads` must be positive.")
        if config.dropout < 0.0 or config.dropout >= 1.0:
            raise ValueError("`dropout` must be in [0.0, 1.0).")

        self.config = GATBaselineConfig(
            hidden_dim=int(config.hidden_dim),
            num_labels=int(config.num_labels),
            num_gat_layers=int(config.num_gat_layers),
            heads=int(config.heads),
            dropout=float(config.dropout),
            num_opcodes=int(config.num_opcodes),
        )

        self.node_embed = OpcodeEmbedding(
            num_opcodes=self.config.num_opcodes,
            opcode_embed_dim=64,
            pc_embed_dim=32,
            output_dim=self.config.hidden_dim,
        )

        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(self.config.num_gat_layers):
            if i < self.config.num_gat_layers - 1:
                # Intermediate layers: multi-head with concat
                self.gat_layers.append(
                    GATConv(
                        self.config.hidden_dim,
                        self.config.hidden_dim // self.config.heads,
                        heads=self.config.heads,
                        concat=True,
                        dropout=self.config.dropout,
                    )
                )
            else:
                # Final layer: single head with mean aggregation
                self.gat_layers.append(
                    GATConv(
                        self.config.hidden_dim,
                        self.config.hidden_dim,
                        heads=1,
                        concat=False,
                        dropout=self.config.dropout,
                    )
                )
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
                "GAT baseline requires `graph_x` and `graph_edge_index`. "
                "Ensure the dataset has graph artifacts available."
            )
        if graph_batch is None:
            graph_batch = torch.zeros(graph_x.size(0), dtype=torch.long, device=graph_x.device)

        h = self.node_embed(graph_x)

        for conv, norm in zip(self.gat_layers, self.norms):
            h = conv(h, graph_edge_index)
            h = norm(h)
            h = torch.nn.functional.elu(h)
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
            "num_gat_layers": int(self.config.num_gat_layers),
            "heads": int(self.config.heads),
        }
