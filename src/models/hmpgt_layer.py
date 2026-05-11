"""Heterogeneous Message-Passing Graph Transformer (HMPGT) layer and branch.

Novel contribution:
    Unlike standard GAT or homogeneous graph transformers that apply the same
    learned projections to every edge, HMPGT maintains **type-specific Q/K/V
    linear projections** for each of the 9 heterogeneous edge types present in
    EVM control-flow and data-flow graphs (5 CFG + 4 DFG).  This allows the
    model to learn distinct attention semantics for, e.g., conditional jumps
    (cfg_jumpi_true/false) vs. storage data-flow (dfg_storage_flow).

    Additionally, an optional **cross-modal attention bias** lets the graph
    attention scores be conditioned on sequential token context from a
    pretrained CodeBERT encoder.  This is computed as a lightweight
    cross-attention (graph-node queries × token keys) projected to a scalar
    bias that is added to the sparse attention logits before softmax, enabling
    the graph branch to integrate information from the sequential branch
    without requiring full feature concatenation.

Edge type vocabulary (from ``src.preprocessing.graph_builder``):
    0: cfg_fallthrough   1: cfg_jump          2: cfg_jumpi_true
    3: cfg_jumpi_false   4: cfg_terminal      5: dfg_stack_flow
    6: dfg_storage_flow  7: dfg_memory_flow   8: dfg_control_dependency
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.utils.checkpoint as checkpoint
from torch import Tensor, nn
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch_geometric.utils import softmax as pyg_softmax

from src.models.gnn_branch import OpcodeEmbedding


# ---------------------------------------------------------------------------
# Cross-modal attention bias module
# ---------------------------------------------------------------------------

class CrossModalBias(nn.Module):
    """Compute a per-node scalar attention bias from sequential token context.

    Mechanism:
        1. A small multi-head cross-attention where graph nodes are queries and
           token embeddings are keys/values.
        2. The resulting per-node vector is projected to a scalar.

    This scalar is broadcast to every outgoing edge of the node and added to
    the attention logits in the HMPGT layer.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.proj = nn.Linear(hidden_dim, 1)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x_nodes: Tensor,
        token_context: Tensor,
        batch: Optional[Tensor] = None,
        token_batch: Optional[Tensor] = None,
    ) -> Tensor:
        """Return per-node scalar bias ``[num_nodes, 1]``.

        For batched graphs the function groups nodes and tokens by their batch
        index, pads them into a dense (B, max_len, D) tensor, runs batched
        multi-head attention, and then un-pads back to the sparse layout.
        """
        if batch is None and token_batch is None:
            # Single graph — treat as batch-size 1.
            q = x_nodes.unsqueeze(0)          # [1, N, D]
            kv = token_context.unsqueeze(0)    # [1, T, D]
            out, _ = self.cross_attn(q, kv, kv)  # [1, N, D]
            out = self.norm(out.squeeze(0))    # [N, D]
            return self.proj(out)              # [N, 1]

        # Batched mode — pad to dense and use key_padding_mask.
        device = x_nodes.device
        if batch is None:
            batch = torch.zeros(x_nodes.size(0), dtype=torch.long, device=device)
        if token_batch is None:
            token_batch = torch.zeros(token_context.size(0), dtype=torch.long, device=device)

        bs = max(batch.max().item(), token_batch.max().item()) + 1

        # Pad nodes.
        node_counts = torch.zeros(bs, dtype=torch.long, device=device)
        node_counts.scatter_add_(0, batch, torch.ones_like(batch))
        max_n = int(node_counts.max().item())

        q_padded = x_nodes.new_zeros(bs, max_n, x_nodes.size(-1))
        q_mask = torch.ones(bs, max_n, dtype=torch.bool, device=device)
        offsets_n = torch.zeros(bs, dtype=torch.long, device=device)
        for i in range(bs):
            sel = (batch == i)
            n_i = int(sel.sum().item())
            q_padded[i, :n_i] = x_nodes[sel]
            q_mask[i, :n_i] = False
            offsets_n[i] = n_i

        # Pad tokens.
        tok_counts = torch.zeros(bs, dtype=torch.long, device=device)
        tok_counts.scatter_add_(0, token_batch, torch.ones_like(token_batch))
        max_t = int(tok_counts.max().item())

        kv_padded = token_context.new_zeros(bs, max_t, token_context.size(-1))
        kv_mask = torch.ones(bs, max_t, dtype=torch.bool, device=device)
        for i in range(bs):
            sel = (token_batch == i)
            t_i = int(sel.sum().item())
            kv_padded[i, :t_i] = token_context[sel]
            kv_mask[i, :t_i] = False

        out, _ = self.cross_attn(
            q_padded, kv_padded, kv_padded,
            key_padding_mask=kv_mask,
        )  # [B, max_n, D]

        # Un-pad back to sparse layout.
        parts = []
        for i in range(bs):
            n_i = int(offsets_n[i].item())
            parts.append(out[i, :n_i])
        out_sparse = torch.cat(parts, dim=0)       # [total_nodes, D]
        out_sparse = self.norm(out_sparse)
        return self.proj(out_sparse)                # [total_nodes, 1]


# ---------------------------------------------------------------------------
# HMPGT Layer
# ---------------------------------------------------------------------------

class HMPGTLayer(nn.Module):
    """Single Heterogeneous Message-Passing Graph Transformer layer.

    Architecture per forward pass:
        1. **Type-specific message computation** — separate learned V
           projections per edge type transform source node features.
        2. **Heterogeneous multi-head attention** — separate Q/K projections
           per edge type compute attention logits, normalized with sparse
           softmax over each destination node's neighborhood.
        3. **Cross-modal attention bias (optional)** — a scalar bias derived
           from sequential token context (CodeBERT) is added to attention
           logits before softmax.
        4. **Aggregation + residual + LayerNorm**.

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of input/output node features.
    num_heads : int
        Number of attention heads.
    num_edge_types : int
        Number of distinct edge types (default 9 for HARDEN-V2).
    dropout : float
        Dropout rate for attention weights and feed-forward.
    use_cross_modal_bias : bool
        Whether to include the cross-modal attention bias mechanism.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        num_edge_types: int = 9,
        dropout: float = 0.1,
        use_cross_modal_bias: bool = True,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_edge_types = num_edge_types
        self.scale = math.sqrt(self.head_dim)
        self.use_cross_modal_bias = use_cross_modal_bias

        # Type-specific Q / K / V projections  (ModuleList indexed by etype).
        self.q_projs = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_edge_types)]
        )
        self.k_projs = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_edge_types)]
        )
        self.v_projs = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_edge_types)]
        )

        # Output projection after multi-head aggregation.
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Cross-modal bias (optional).
        if use_cross_modal_bias:
            self.cross_modal = CrossModalBias(hidden_dim, num_heads=min(num_heads, 4), dropout=dropout)
        else:
            self.cross_modal = None

        # Feed-forward network (post-attention).
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attn_drop = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    def forward(
        self,
        x_nodes: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        batch: Optional[Tensor] = None,
        token_context: Optional[Tensor] = None,
        token_batch: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        x_nodes : Tensor [total_nodes, hidden_dim]
            Node feature matrix.
        edge_index : Tensor [2, num_edges]
            Source-to-destination edge connectivity.
        edge_type : Tensor [num_edges]
            Integer edge-type label for every edge.
        batch : Tensor [total_nodes], optional
            Batch assignment for batched graphs.
        token_context : Tensor [total_tokens, hidden_dim], optional
            Sequential token context from CodeBERT.
        token_batch : Tensor [total_tokens], optional
            Batch assignment for token sequences.

        Returns
        -------
        Tensor [total_nodes, hidden_dim]
        """
        num_nodes = x_nodes.size(0)
        num_edges = edge_index.size(1)
        device = x_nodes.device

        src, dst = edge_index[0], edge_index[1]  # [E]

        # -- Guard: handle empty graphs (no edges) gracefully ------
        if num_edges == 0:
            # No message passing possible — just apply FFN with residual.
            x_nodes = self.norm1(x_nodes)
            x_nodes = self.norm2(x_nodes + self.ffn(x_nodes))
            return x_nodes

        # -- Step 1 & 3: Type-specific Q, K, V and attention scores ----------
        # Collect per-type results and reorder to original edge ordering.
        # This avoids in-place indexed assignment which breaks autograd.
        q_parts, k_parts, v_parts, idx_parts = [], [], [], []

        for etype in range(self.num_edge_types):
            mask = edge_type == etype
            if not mask.any():
                continue
            idx = mask.nonzero(as_tuple=False).squeeze(-1)
            idx_parts.append(idx)
            src_e = src[idx]
            dst_e = dst[idx]

            q_parts.append(self.q_projs[etype](x_nodes[dst_e]))  # [E_t, D]
            k_parts.append(self.k_projs[etype](x_nodes[src_e]))
            v_parts.append(self.v_projs[etype](x_nodes[src_e]))

        # Concatenate in type-grouped order then scatter back to edge order.
        order = torch.cat(idx_parts)
        inv_order = torch.empty_like(order)
        inv_order[order] = torch.arange(order.size(0), device=device)

        q_cat = torch.cat(q_parts, dim=0)[inv_order]  # [E, D]
        k_cat = torch.cat(k_parts, dim=0)[inv_order]
        v_cat = torch.cat(v_parts, dim=0)[inv_order]

        # Reshape to multi-head: [E, H, d_k]
        q_all = q_cat.view(num_edges, self.num_heads, self.head_dim)
        k_all = k_cat.view(num_edges, self.num_heads, self.head_dim)
        v_all = v_cat.view(num_edges, self.num_heads, self.head_dim)

        # Attention logits: sum over head_dim → [E, H]
        attn_logits = (q_all * k_all).sum(dim=-1) / self.scale  # [E, H]

        # -- Step 2: Cross-modal bias (optional) ----------------------------
        if (
            self.cross_modal is not None
            and token_context is not None
        ):
            bias = self.cross_modal(
                x_nodes, token_context, batch, token_batch
            )  # [N, 1]
            # Broadcast per-destination-node bias to edges.
            attn_logits = attn_logits + bias[dst]  # [E, 1] broadcasts to [E, H]

        # -- Sparse softmax per destination node ----------------------------
        # pyg_softmax normalizes over edges sharing the same dst node.
        attn_weights = pyg_softmax(attn_logits, dst, num_nodes=num_nodes)  # [E, H]
        attn_weights = self.attn_drop(attn_weights)

        # -- Step 4: Weighted aggregation -----------------------------------
        # weighted messages: [E, H, d_k]
        messages = attn_weights.unsqueeze(-1) * v_all  # [E, H, d_k]
        messages_flat = messages.view(num_edges, self.hidden_dim)  # [E, D]

        # Scatter-add into destination nodes.
        agg = x_nodes.new_zeros(num_nodes, self.hidden_dim)
        agg.scatter_add_(
            0,
            dst.unsqueeze(1).expand_as(messages_flat),
            messages_flat,
        )

        # Output projection.
        agg = self.out_proj(agg)

        # Residual + LayerNorm.
        x_nodes = self.norm1(x_nodes + agg)

        # Feed-forward with residual + LayerNorm.
        x_nodes = self.norm2(x_nodes + self.ffn(x_nodes))

        return x_nodes


# ---------------------------------------------------------------------------
# HMPGT Branch  (stacks layers + embedding + global pooling)
# ---------------------------------------------------------------------------

class HMPGTBranch(nn.Module):
    """Full graph-branch encoder using stacked HMPGT layers.

    Pipeline:
        1. ``OpcodeEmbedding`` maps raw integer node features (opcode_id, pc,
           bb_id) to a dense ``hidden_dim``-dimensional representation.
        2. A stack of ``HMPGTLayer`` layers performs heterogeneous message
           passing with type-specific attention.
        3. Global pooling (concatenated mean + max) followed by a linear
           projection yields a fixed-size graph-level embedding.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension throughout the network.
    num_layers : int
        Number of stacked HMPGT layers.
    num_heads : int
        Attention heads per HMPGT layer.
    num_edge_types : int
        Distinct edge types in the heterogeneous graph.
    dropout : float
        Dropout probability.
    use_cross_modal_bias : bool
        Enable cross-modal attention bias in every layer.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        num_edge_types: int = 9,
        dropout: float = 0.1,
        use_cross_modal_bias: bool = False,
        use_checkpointing: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_checkpointing = use_checkpointing

        # Node feature encoder (shared with GNNBranch for consistency).
        self.node_embed = OpcodeEmbedding(
            num_opcodes=256,
            opcode_embed_dim=64,
            pc_embed_dim=32,
            output_dim=hidden_dim,
        )

        # Stack of HMPGT layers.
        self.layers = nn.ModuleList([
            HMPGTLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_edge_types=num_edge_types,
                dropout=dropout,
                use_cross_modal_bias=use_cross_modal_bias,
            )
            for _ in range(num_layers)
        ])

        # Global pooling: mean + max → 2*D → D.
        self.pool_project = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        batch: Optional[Tensor] = None,
        token_context: Optional[Tensor] = None,
        token_batch: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor [total_nodes, 3]
            Raw integer node features (opcode_id, pc, basic_block_id).
        edge_index : Tensor [2, num_edges]
        edge_type : Tensor [num_edges]
        batch : Tensor [total_nodes], optional
        token_context : Tensor [total_tokens, hidden_dim], optional
        token_batch : Tensor [total_tokens], optional

        Returns
        -------
        Tensor [batch_size, hidden_dim]
            Graph-level embeddings.
        """
        h = self.node_embed(x)  # [N, hidden_dim]

        for layer in self.layers:
            if self.training and self.use_checkpointing and h.requires_grad:
                h = checkpoint.checkpoint(
                    layer,
                    h, edge_index, edge_type, batch, token_context, token_batch,
                    use_reentrant=False
                )
            else:
                h = layer(
                    h, edge_index, edge_type,
                    batch=batch,
                    token_context=token_context,
                    token_batch=token_batch,
                )

        # Global pooling.
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_pooled = torch.cat([h_mean, h_max], dim=-1)

        return self.pool_project(h_pooled)
