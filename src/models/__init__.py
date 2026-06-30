"""Phase 4 model modules for OpcodeGT."""

from src.models.opcodegt import (
    OpcodeGT,
    OpcodeGTConfig,
    SimpleGINFusion,
    build_opcodegt,
)
from src.models.fusion import GatedFusion
from src.models.gnn_branch import GNNBranch, OpcodeEmbedding
from src.models.hmpgt_layer import HMPGTBranch, HMPGTLayer
from src.models.cross_attention import CrossAttentionFusion
from src.models.classification_head import (
    LabelAwareHead,
    SimpleLinearHead,
    build_classification_head,
)

__all__ = [
    "OpcodeGT",
    "OpcodeGTConfig",
    "SimpleGINFusion",
    "build_opcodegt",
    "GatedFusion",
    "GNNBranch",
    "OpcodeEmbedding",
    "HMPGTBranch",
    "HMPGTLayer",
    "CrossAttentionFusion",
    "LabelAwareHead",
    "SimpleLinearHead",
    "build_classification_head",
]

