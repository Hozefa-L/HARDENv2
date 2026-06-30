"""Cross-Attention Fusion module for HARDEN-V2.

Performs bidirectional cross-attention between contract-level token
embeddings (from the CodeBERT opcode branch) and per-node graph
embeddings (from the HMPGT / GNN graph branch).

Directions
----------
T→G  Token embedding queries graph node embeddings.
     "Which graph nodes matter for this contract's opcode representation?"

G→T  Graph node embeddings query the token embedding.
     "Which token-level information informs each node?"

The two attended representations are pooled and projected to produce a
single fused contract embedding of shape ``[batch_size, hidden_dim]``.

Implementation notes
--------------------
Because the current pipeline uses frozen CodeBERT CLS features, the
token side is a **single vector per contract** (not a full sequence).
The graph side has a **variable number of nodes per contract** due to
batched PyTorch Geometric graphs.

To run batched ``nn.MultiheadAttention`` efficiently we:

1. Pad graph node sequences to ``max_nodes_in_batch``.
2. Build a boolean key-padding mask for the padded positions.
3. Run both attention directions in one batched call each.
4. Pool the G→T attended output (mean over valid nodes).
5. Concatenate the T→G and G→T pooled outputs and linearly project.
"""

from typing import Optional

import torch
from torch import nn

__all__ = ["CrossAttentionFusion"]


class CrossAttentionFusion(nn.Module):
    """Bidirectional cross-attention fusion for token and graph embeddings.

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of both the token and graph node embeddings.
    num_heads : int
        Number of attention heads (must divide ``hidden_dim``).
    dropout : float
        Dropout probability applied inside the attention layers and the
        output projection.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError("`hidden_dim` must be positive.")
        if num_heads <= 0:
            raise ValueError("`num_heads` must be positive.")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"`hidden_dim` ({hidden_dim}) must be divisible by "
                f"`num_heads` ({num_heads})."
            )
        if not (0.0 <= dropout < 1.0):
            raise ValueError("`dropout` must be in [0.0, 1.0).")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # T→G: token queries, graph keys/values
        self.t2g_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # G→T: graph queries, token keys/values
        self.g2t_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Merge bidirectional outputs: [2*hidden_dim] → [hidden_dim]
        self.projection = nn.Linear(2 * hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def forward(
        self,
        token_embeddings: torch.Tensor,
        graph_node_embeddings: torch.Tensor,
        graph_batch: Optional[torch.Tensor] = None,
        token_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse token and graph embeddings via bidirectional cross-attention.

        Parameters
        ----------
        token_embeddings : Tensor [batch_size, hidden_dim]
            Contract-level opcode representations (e.g. frozen CodeBERT CLS).
        graph_node_embeddings : Tensor [total_nodes, hidden_dim]
            All node embeddings from a batched PyG graph.
        graph_batch : Tensor [total_nodes], optional
            Batch assignment vector from PyG (maps each node to its graph
            index).  Required when ``total_nodes > 0``.
        token_lengths : Tensor, optional
            Reserved for future use when the token side carries a full
            sequence.  Currently unused.

        Returns
        -------
        Tensor [batch_size, hidden_dim]
            Fused contract embedding.
        """
        batch_size = token_embeddings.size(0)
        device = token_embeddings.device

        # --- handle empty / missing graph nodes ---
        if graph_node_embeddings.numel() == 0:
            return self._fallback_token_only(token_embeddings)

        if graph_batch is None:
            if batch_size == 1:
                graph_batch = torch.zeros(
                    graph_node_embeddings.size(0),
                    dtype=torch.long,
                    device=device,
                )
            else:
                raise ValueError(
                    "`graph_batch` is required when batch_size > 1."
                )

        # --- per-graph empty check ---
        nodes_per_graph = _count_per_graph(graph_batch, batch_size)
        all_empty = (nodes_per_graph == 0).all()
        if all_empty:
            return self._fallback_token_only(token_embeddings)

        # --- pad graph nodes into [B, max_nodes, D] ---
        padded_nodes, key_padding_mask = self._pad_graph_nodes(
            graph_node_embeddings, graph_batch, batch_size, nodes_per_graph,
        )

        # Token side: [B, 1, D]
        token_seq = token_embeddings.unsqueeze(1)

        # --- T→G attention: token queries graph nodes ---
        t2g_out, _ = self.t2g_attention(
            query=token_seq,
            key=padded_nodes,
            value=padded_nodes,
            key_padding_mask=key_padding_mask,
        )
        # t2g_out: [B, 1, D] → squeeze to [B, D]
        t2g_pooled = t2g_out.squeeze(1)

        # --- G→T attention: graph nodes query token ---
        # No key_padding_mask needed: the key side is a single token.
        g2t_out, _ = self.g2t_attention(
            query=padded_nodes,
            key=token_seq,
            value=token_seq,
        )
        # g2t_out: [B, max_nodes, D] → masked mean pool over valid nodes
        g2t_pooled = self._masked_mean_pool(
            g2t_out, key_padding_mask, nodes_per_graph,
        )

        # --- For graphs with 0 nodes, fall back to token embedding ---
        empty_mask = (nodes_per_graph == 0)
        if empty_mask.any():
            t2g_pooled = torch.where(
                empty_mask.unsqueeze(-1), token_embeddings, t2g_pooled,
            )
            g2t_pooled = torch.where(
                empty_mask.unsqueeze(-1), token_embeddings, g2t_pooled,
            )

        # --- merge & project ---
        merged = torch.cat([t2g_pooled, g2t_pooled], dim=-1)  # [B, 2D]
        fused = self.projection(merged)  # [B, D]
        fused = self.dropout(fused)
        fused = self.layer_norm(fused + token_embeddings)  # residual
        return fused

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _fallback_token_only(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """When no graph nodes are available, pass token through projection."""
        batch_size = token_embeddings.size(0)
        dummy = token_embeddings.clone()
        merged = torch.cat([dummy, dummy], dim=-1)
        fused = self.projection(merged)
        fused = self.dropout(fused)
        return self.layer_norm(fused + token_embeddings)

    @staticmethod
    def _pad_graph_nodes(
        node_embeddings: torch.Tensor,
        graph_batch: torch.Tensor,
        batch_size: int,
        nodes_per_graph: torch.Tensor,
    ) -> tuple:
        """Pad variable-length node sequences into a dense [B, max_N, D] tensor.

        Returns
        -------
        padded : Tensor [B, max_nodes, D]
        key_padding_mask : BoolTensor [B, max_nodes]
            ``True`` at padded (invalid) positions.
        """
        max_nodes = int(nodes_per_graph.max().item())
        # Ensure at least 1 to avoid zero-size tensors
        max_nodes = max(max_nodes, 1)
        hidden_dim = node_embeddings.size(-1)
        device = node_embeddings.device

        padded = node_embeddings.new_zeros(batch_size, max_nodes, hidden_dim)
        key_padding_mask = torch.ones(
            batch_size, max_nodes, dtype=torch.bool, device=device,
        )

        for i in range(batch_size):
            n = int(nodes_per_graph[i].item())
            if n == 0:
                continue
            mask_i = graph_batch == i
            padded[i, :n] = node_embeddings[mask_i]
            key_padding_mask[i, :n] = False

        return padded, key_padding_mask

    @staticmethod
    def _masked_mean_pool(
        attended: torch.Tensor,
        key_padding_mask: torch.Tensor,
        nodes_per_graph: torch.Tensor,
    ) -> torch.Tensor:
        """Mean-pool over valid (non-padded) positions.

        Parameters
        ----------
        attended : [B, max_nodes, D]
        key_padding_mask : [B, max_nodes]  (True = pad)
        nodes_per_graph : [B]

        Returns
        -------
        pooled : [B, D]
        """
        # Zero out padded positions
        valid_mask = (~key_padding_mask).unsqueeze(-1).float()  # [B, N, 1]
        summed = (attended * valid_mask).sum(dim=1)  # [B, D]
        counts = nodes_per_graph.clamp(min=1).unsqueeze(-1).float()  # [B, 1]
        return summed / counts


def _count_per_graph(
    graph_batch: torch.Tensor, batch_size: int,
) -> torch.Tensor:
    """Count the number of nodes assigned to each graph in the batch."""
    counts = torch.zeros(batch_size, dtype=torch.long, device=graph_batch.device)
    ones = torch.ones_like(graph_batch)
    counts.scatter_add_(0, graph_batch, ones)
    return counts
