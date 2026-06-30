"""GNN branch encoder for OpcodeGT using actual graph structure.

Processes PyTorch Geometric graph data (node features + edge_index) through
GIN layers with global pooling, replacing the flat-feature MLP graph branch.
"""

from typing import Optional, Tuple

import torch
from torch import nn
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool


class OpcodeEmbedding(nn.Module):
    """Learnable embedding for integer-encoded node features (opcode_id, pc, bb_id)."""

    def __init__(
        self,
        num_opcodes: int = 256,
        pc_embed_dim: int = 32,
        opcode_embed_dim: int = 64,
        output_dim: int = 128,
    ):
        super().__init__()
        self.opcode_embed = nn.Embedding(num_opcodes + 1, opcode_embed_dim, padding_idx=0)
        self.pc_linear = nn.Linear(1, pc_embed_dim)
        self.project = nn.Linear(opcode_embed_dim + pc_embed_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [num_nodes, 3] with columns (opcode_id, pc, basic_block_id)."""
        opcode_ids = x[:, 0].long().clamp(0, 256)
        pc_values = x[:, 1:2].float()
        opcode_emb = self.opcode_embed(opcode_ids)
        pc_emb = self.pc_linear(pc_values)
        combined = torch.cat([opcode_emb, pc_emb], dim=-1)
        return self.project(combined)


class GINBlock(nn.Module):
    """Single GIN layer with batch norm and residual connection."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv = GINConv(mlp, train_eps=True)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        out = self.conv(x, edge_index)
        out = self.norm(out)
        out = self.dropout(out)
        return out + x  # residual


class GNNBranch(nn.Module):
    """GNN encoder using GIN layers with global pooling.

    Takes actual PyG graph data and produces a fixed-size graph embedding.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_gnn_layers: int = 3,
        dropout: float = 0.1,
        num_opcodes: int = 256,
        node_input_features: int = 3,
    ):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("`hidden_dim` must be positive.")
        if num_gnn_layers <= 0:
            raise ValueError("`num_gnn_layers` must be positive.")

        self.hidden_dim = hidden_dim
        self.node_embed = OpcodeEmbedding(
            num_opcodes=num_opcodes,
            opcode_embed_dim=64,
            pc_embed_dim=32,
            output_dim=hidden_dim,
        )
        self.gnn_layers = nn.ModuleList(
            [GINBlock(hidden_dim, dropout) for _ in range(num_gnn_layers)]
        )
        # mean + max pooling → 2*hidden_dim → project to hidden_dim
        self.pool_project = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [total_nodes, node_input_features]
            edge_index: Edge connectivity [2, total_edges]
            batch: Batch assignment vector [total_nodes] (for batched graphs)

        Returns:
            Graph-level embedding [batch_size, hidden_dim]
        """
        h = self.node_embed(x)

        for layer in self.gnn_layers:
            h = layer(h, edge_index)

        # global pooling: concatenate mean and max
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_pooled = torch.cat([h_mean, h_max], dim=-1)

        return self.pool_project(h_pooled)
