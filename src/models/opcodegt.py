from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from torch import nn

from src.models.classification_head import (
    ClassificationHeadConfig,
    LabelAwareHead,
    SimpleLinearHead,
    build_classification_head,
)
try:
    from src.models.cross_attention import CrossAttentionFusion
except ImportError:
    CrossAttentionFusion = None  # archived
from src.models.fusion import GatedFusion
from src.models.gnn_branch import GNNBranch
from src.models.hmpgt_layer import HMPGTBranch

VALID_MODES = {"opcode_only", "graph_only", "fused"}


@dataclass(frozen=True)
class OpcodeGTConfig:
    mode: str
    opcode_input_dim: int
    graph_input_dim: int
    hidden_dim: int
    num_labels: int
    dropout: float = 0.1
    use_gnn: bool = True
    num_gnn_layers: int = 3
    # --- v2 architecture flags (ignored by SimpleGINFusion) ---
    architecture: str = "opcodegt_v2"  # "simple_gin_fusion" or "opcodegt_v2"
    use_cross_attention: bool = False  # True → CrossAttentionFusion (hurts on small data)
    use_hmpgt: bool = True  # False → GINBlock (homogeneous)
    use_label_attention: bool = True  # False → nn.Linear head
    cfg_only: bool = False  # True → filter to CFG edges only (types 0-4)
    num_heads: int = 8  # attention heads for HMPGT and cross-attention
    num_edge_types: int = 9  # number of heterogeneous edge types
    gradient_checkpointing: bool = True  # True → wrap HMPGT layers in checkpoint


class _OpcodeEncoder(nn.Module):
    """Text branch: deeper MLP with residual connections and layer norm.

    Architecturally distinct from the baseline MLP (which has no residuals/norm).
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("Branch `input_dim` must be positive.")
        if hidden_dim <= 0:
            raise ValueError("Branch `hidden_dim` must be positive.")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("`dropout` must be in [0.0, 1.0).")

        self.project = nn.Linear(input_dim, hidden_dim)
        self.layer1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("Branch inputs must be rank-2 tensors shaped [batch, input_dim].")
        h = self.project(features)
        h = self.norm1(h + self.layer1(h))  # residual + norm
        h = self.norm2(h + self.layer2(h))  # residual + norm
        return h


class _FlatBranchEncoder(nn.Module):
    """Legacy flat-feature encoder (kept for backward compat when use_gnn=False)."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("Branch inputs must be rank-2 tensors shaped [batch, input_dim].")
        return self.encoder(features)


# ---------------------------------------------------------------------------
# SimpleGINFusion — the original (Level 0–2) architecture, kept as baseline
# ---------------------------------------------------------------------------


class SimpleGINFusion(nn.Module):
    """Original OpcodeGT architecture: GIN-based GNN + GatedFusion + linear head.

    Preserved as a baseline for ablation against the new OpcodeGT v2.
    This is the exact model that was used in Levels 0–2.
    """

    def __init__(self, config: OpcodeGTConfig):
        super().__init__()
        mode = str(config.mode).strip().lower()
        if mode not in VALID_MODES:
            raise ValueError(f"`mode` must be one of {sorted(VALID_MODES)}.")
        if config.num_labels <= 0:
            raise ValueError("`num_labels` must be positive.")

        self.config = OpcodeGTConfig(
            mode=mode,
            opcode_input_dim=int(config.opcode_input_dim),
            graph_input_dim=int(config.graph_input_dim),
            hidden_dim=int(config.hidden_dim),
            num_labels=int(config.num_labels),
            dropout=float(config.dropout),
            use_gnn=bool(config.use_gnn),
            num_gnn_layers=int(config.num_gnn_layers),
            architecture="simple_gin_fusion",
            gradient_checkpointing=False,
        )

        self.opcode_encoder: Optional[_OpcodeEncoder] = None
        self.graph_encoder: Optional[nn.Module] = None
        self.fusion: Optional[GatedFusion] = None

        if self.config.mode in {"opcode_only", "fused"}:
            self.opcode_encoder = _OpcodeEncoder(
                input_dim=self.config.opcode_input_dim,
                hidden_dim=self.config.hidden_dim,
                dropout=self.config.dropout,
            )

        if self.config.mode in {"graph_only", "fused"}:
            if self.config.use_gnn:
                self.graph_encoder = GNNBranch(
                    hidden_dim=self.config.hidden_dim,
                    num_gnn_layers=self.config.num_gnn_layers,
                    dropout=self.config.dropout,
                )
            else:
                self.graph_encoder = _FlatBranchEncoder(
                    input_dim=self.config.graph_input_dim,
                    hidden_dim=self.config.hidden_dim,
                    dropout=self.config.dropout,
                )

        if self.config.mode == "fused":
            self.fusion = GatedFusion(hidden_dim=self.config.hidden_dim)

        self.classifier = nn.Linear(self.config.hidden_dim, self.config.num_labels)

    def forward(
        self,
        opcode_features: Optional[torch.Tensor] = None,
        graph_features: Optional[torch.Tensor] = None,
        graph_x: Optional[torch.Tensor] = None,
        graph_edge_index: Optional[torch.Tensor] = None,
        graph_batch: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.config.mode == "opcode_only":
            if opcode_features is None:
                raise ValueError("`opcode_features` is required when mode is `opcode_only`.")
            if self.opcode_encoder is None:
                raise RuntimeError("Opcode encoder is not initialized.")
            hidden = self.opcode_encoder(opcode_features)

        elif self.config.mode == "graph_only":
            if self.graph_encoder is None:
                raise RuntimeError("Graph encoder is not initialized.")
            hidden = self._encode_graph(
                graph_features, graph_x, graph_edge_index, graph_batch
            )

        else:  # fused
            if opcode_features is None:
                raise ValueError("`opcode_features` is required when mode is `fused`.")
            if self.opcode_encoder is None or self.graph_encoder is None or self.fusion is None:
                raise RuntimeError("Fused model components are not initialized.")

            opcode_hidden = self.opcode_encoder(opcode_features)
            graph_hidden = self._encode_graph(
                graph_features, graph_x, graph_edge_index, graph_batch
            )
            hidden, _ = self.fusion(opcode_hidden, graph_hidden)

        logits = self.classifier(hidden)
        if logits.ndim != 2:
            raise RuntimeError(f"Classifier produced invalid output shape: {tuple(logits.shape)}")
        return logits

    def _encode_graph(
        self,
        graph_features: Optional[torch.Tensor],
        graph_x: Optional[torch.Tensor],
        graph_edge_index: Optional[torch.Tensor],
        graph_batch: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.graph_encoder is None:
            raise RuntimeError("Graph encoder is not initialized.")
        if self.config.use_gnn:
            if graph_x is None or graph_edge_index is None:
                raise ValueError(
                    "`graph_x` and `graph_edge_index` are required when use_gnn=True."
                )
            return self.graph_encoder(graph_x, graph_edge_index, graph_batch)
        else:
            if graph_features is None:
                raise ValueError(
                    "`graph_features` is required when use_gnn=False."
                )
            return self.graph_encoder(graph_features)

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters()))

    def branch_dims(self) -> Dict[str, int]:
        return {
            "opcode_input_dim": int(self.config.opcode_input_dim),
            "graph_input_dim": int(self.config.graph_input_dim),
            "hidden_dim": int(self.config.hidden_dim),
            "num_labels": int(self.config.num_labels),
            "use_gnn": self.config.use_gnn,
            "num_gnn_layers": self.config.num_gnn_layers,
        }


# ---------------------------------------------------------------------------
# OpcodeGT v2 — the novel architecture (Level 3)
# ---------------------------------------------------------------------------

# CFG edge types are indices 0–4 in the edge type vocabulary
_CFG_EDGE_TYPE_MAX = 4


class OpcodeGT(nn.Module):
    """OpcodeGT v2: HMPGT + Cross-Attention Fusion + Label-Aware Head.

    Novel architecture combining:
    - Heterogeneous Message-Passing Graph Transformer (HMPGT) for
      type-aware graph encoding with cross-modal bias from token context
    - Bidirectional cross-attention fusion between sequential (CodeBERT)
      and structural (graph) representations
    - Label-aware attention classification head with per-SWC query vectors

    Supports config-driven ablation modes to isolate contributions:
    - use_hmpgt=False → falls back to homogeneous GIN (SimpleGINFusion graph branch)
    - use_cross_attention=False → falls back to GatedFusion
    - use_label_attention=False → falls back to nn.Linear head
    - cfg_only=True → filters graph to CFG edges only

    When ``config.architecture == "simple_gin_fusion"``, use ``build_opcodegt()``
    factory which returns a ``SimpleGINFusion`` instead.
    """

    def __init__(self, config: OpcodeGTConfig):
        super().__init__()
        mode = str(config.mode).strip().lower()
        if mode not in VALID_MODES:
            raise ValueError(f"`mode` must be one of {sorted(VALID_MODES)}.")
        if config.num_labels <= 0:
            raise ValueError("`num_labels` must be positive.")

        self.config = OpcodeGTConfig(
            mode=mode,
            opcode_input_dim=int(config.opcode_input_dim),
            graph_input_dim=int(config.graph_input_dim),
            hidden_dim=int(config.hidden_dim),
            num_labels=int(config.num_labels),
            dropout=float(config.dropout),
            use_gnn=bool(config.use_gnn),
            num_gnn_layers=int(config.num_gnn_layers),
            architecture="opcodegt_v2",
            use_cross_attention=bool(config.use_cross_attention),
            use_hmpgt=bool(config.use_hmpgt),
            use_label_attention=bool(config.use_label_attention),
            cfg_only=bool(config.cfg_only),
            num_heads=int(config.num_heads),
            num_edge_types=int(config.num_edge_types),
            gradient_checkpointing=bool(getattr(config, "gradient_checkpointing", True)),
        )

        self.opcode_encoder: Optional[_OpcodeEncoder] = None
        self.graph_encoder: Optional[nn.Module] = None
        self.fusion: Optional[nn.Module] = None

        # --- Opcode branch (same as SimpleGINFusion) ---
        if self.config.mode in {"opcode_only", "fused"}:
            self.opcode_encoder = _OpcodeEncoder(
                input_dim=self.config.opcode_input_dim,
                hidden_dim=self.config.hidden_dim,
                dropout=self.config.dropout,
            )

        # --- Graph branch (HMPGT or GIN fallback) ---
        if self.config.mode in {"graph_only", "fused"}:
            if self.config.use_gnn:
                if self.config.use_hmpgt:
                    self.graph_encoder = HMPGTBranch(
                        hidden_dim=self.config.hidden_dim,
                        num_layers=self.config.num_gnn_layers,
                        num_heads=self.config.num_heads,
                        num_edge_types=self.config.num_edge_types,
                        dropout=self.config.dropout,
                        use_cross_modal_bias=(self.config.mode == "fused"),
                        use_checkpointing=self.config.gradient_checkpointing,
                    )
                else:
                    self.graph_encoder = GNNBranch(
                        hidden_dim=self.config.hidden_dim,
                        num_gnn_layers=self.config.num_gnn_layers,
                        dropout=self.config.dropout,
                    )
            else:
                self.graph_encoder = _FlatBranchEncoder(
                    input_dim=self.config.graph_input_dim,
                    hidden_dim=self.config.hidden_dim,
                    dropout=self.config.dropout,
                )

        # --- Fusion (cross-attention or gated) ---
        if self.config.mode == "fused":
            if self.config.use_cross_attention:
                self.fusion = CrossAttentionFusion(
                    hidden_dim=self.config.hidden_dim,
                    num_heads=self.config.num_heads,
                    dropout=self.config.dropout,
                )
            else:
                self.fusion = GatedFusion(hidden_dim=self.config.hidden_dim)

        # --- Classification head (label-aware or linear) ---
        if self.config.use_label_attention:
            self.classifier = LabelAwareHead(
                hidden_dim=self.config.hidden_dim,
                num_labels=self.config.num_labels,
                dropout=self.config.dropout,
            )
        else:
            self.classifier = nn.Linear(self.config.hidden_dim, self.config.num_labels)

    def forward(
        self,
        opcode_features: Optional[torch.Tensor] = None,
        graph_features: Optional[torch.Tensor] = None,
        graph_x: Optional[torch.Tensor] = None,
        graph_edge_index: Optional[torch.Tensor] = None,
        graph_batch: Optional[torch.Tensor] = None,
        graph_edge_type: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Optionally filter to CFG-only edges
        if self.config.cfg_only and graph_edge_type is not None and graph_edge_index is not None:
            cfg_mask = graph_edge_type <= _CFG_EDGE_TYPE_MAX
            graph_edge_index = graph_edge_index[:, cfg_mask]
            graph_edge_type = graph_edge_type[cfg_mask]

        if self.config.mode == "opcode_only":
            if opcode_features is None:
                raise ValueError("`opcode_features` is required when mode is `opcode_only`.")
            if self.opcode_encoder is None:
                raise RuntimeError("Opcode encoder is not initialized.")
            hidden = self.opcode_encoder(opcode_features)

        elif self.config.mode == "graph_only":
            if self.graph_encoder is None:
                raise RuntimeError("Graph encoder is not initialized.")
            hidden = self._encode_graph(
                graph_features, graph_x, graph_edge_index, graph_batch, graph_edge_type
            )

        else:  # fused
            if opcode_features is None:
                raise ValueError("`opcode_features` is required when mode is `fused`.")
            if self.opcode_encoder is None or self.graph_encoder is None or self.fusion is None:
                raise RuntimeError("Fused model components are not initialized.")

            opcode_hidden = self.opcode_encoder(opcode_features)

            # For HMPGT in fused mode, pass opcode_hidden as token_context
            if self.config.use_hmpgt and self.config.use_gnn:
                graph_hidden = self._encode_graph(
                    graph_features, graph_x, graph_edge_index, graph_batch,
                    graph_edge_type, token_context=opcode_hidden,
                )
            else:
                graph_hidden = self._encode_graph(
                    graph_features, graph_x, graph_edge_index, graph_batch,
                    graph_edge_type,
                )

            # Fusion
            if self.config.use_cross_attention:
                # Both branches produce graph-level [B, hidden_dim] embeddings.
                # CrossAttentionFusion expects a graph_batch vector assigning
                # each "node" to a graph.  Since each graph has exactly one
                # pooled embedding, graph_batch = arange(B).
                B = graph_hidden.size(0)
                fusion_batch = torch.arange(B, device=graph_hidden.device)
                hidden = self.fusion(
                    opcode_hidden, graph_hidden, graph_batch=fusion_batch
                )
            else:
                hidden, _ = self.fusion(opcode_hidden, graph_hidden)

        logits = self.classifier(hidden)
        if logits.ndim != 2:
            raise RuntimeError(f"Classifier produced invalid output shape: {tuple(logits.shape)}")
        return logits

    def _encode_graph(
        self,
        graph_features: Optional[torch.Tensor],
        graph_x: Optional[torch.Tensor],
        graph_edge_index: Optional[torch.Tensor],
        graph_batch: Optional[torch.Tensor],
        graph_edge_type: Optional[torch.Tensor] = None,
        token_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Route to HMPGT, GNN, or flat encoder depending on config."""
        if self.graph_encoder is None:
            raise RuntimeError("Graph encoder is not initialized.")

        if self.config.use_gnn:
            if graph_x is None or graph_edge_index is None:
                raise ValueError(
                    "`graph_x` and `graph_edge_index` are required when use_gnn=True."
                )
            if self.config.use_hmpgt:
                if graph_edge_type is None:
                    # Fallback: treat all edges as type 0
                    graph_edge_type = torch.zeros(
                        graph_edge_index.size(1), dtype=torch.long,
                        device=graph_edge_index.device,
                    )
                # token_context is [B, D] — one embedding per graph.
                # token_batch assigns each to its own graph index.
                token_batch_vec = None
                if token_context is not None:
                    token_batch_vec = torch.arange(
                        token_context.size(0), dtype=torch.long,
                        device=token_context.device,
                    )
                return self.graph_encoder(
                    graph_x, graph_edge_index, graph_edge_type,
                    batch=graph_batch,
                    token_context=token_context,
                    token_batch=token_batch_vec,
                )
            else:
                return self.graph_encoder(graph_x, graph_edge_index, graph_batch)
        else:
            if graph_features is None:
                raise ValueError(
                    "`graph_features` is required when use_gnn=False."
                )
            return self.graph_encoder(graph_features)

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters()))

    def branch_dims(self) -> Dict[str, int]:
        return {
            "opcode_input_dim": int(self.config.opcode_input_dim),
            "graph_input_dim": int(self.config.graph_input_dim),
            "hidden_dim": int(self.config.hidden_dim),
            "num_labels": int(self.config.num_labels),
            "use_gnn": self.config.use_gnn,
            "num_gnn_layers": self.config.num_gnn_layers,
            "architecture": self.config.architecture,
            "use_hmpgt": self.config.use_hmpgt,
            "use_cross_attention": self.config.use_cross_attention,
            "use_label_attention": self.config.use_label_attention,
            "cfg_only": self.config.cfg_only,
            "gradient_checkpointing": self.config.gradient_checkpointing,
        }


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def build_opcodegt(config: OpcodeGTConfig) -> nn.Module:
    """Build the appropriate OpcodeGT variant based on config.architecture.

    Returns ``SimpleGINFusion`` for backward compatibility with Level 0–2
    configs, or the new ``OpcodeGT`` v2 for Level 3+ configs.
    """
    arch = str(config.architecture).strip().lower()
    if arch == "simple_gin_fusion":
        return SimpleGINFusion(config)
    elif arch == "opcodegt_v2":
        return OpcodeGT(config)
    else:
        raise ValueError(
            f"Unknown architecture '{arch}'. "
            "Must be 'simple_gin_fusion' or 'opcodegt_v2'."
        )

