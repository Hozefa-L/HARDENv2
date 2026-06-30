"""BiLSTM baseline for smart contract vulnerability detection.

Bidirectional LSTM operating on opcode token sequences with attention pooling.
Provides a standard sequential model baseline to demonstrate the advantage of
transformer-based and graph-based approaches.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import nn


# Standard EVM opcode vocabulary: 256 possible opcodes + PAD (0) + UNK
OPCODE_VOCAB_SIZE = 258
PAD_IDX = 0
UNK_IDX = 257


@dataclass(frozen=True)
class BiLSTMBaselineConfig:
    num_labels: int
    vocab_size: int = OPCODE_VOCAB_SIZE
    embed_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 2
    max_seq_len: int = 1024
    dropout: float = 0.3


class AttentionPooling(nn.Module):
    """Additive attention pooling over sequence dimension."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            lstm_out: (batch, seq_len, hidden_dim)
            mask: (batch, seq_len) — True for valid positions, False for padding
        Returns:
            pooled: (batch, hidden_dim)
        """
        scores = self.attention(lstm_out).squeeze(-1)  # (batch, seq_len)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)  # (batch, seq_len)
        # Handle all-padding case
        weights = weights.nan_to_num(0.0)
        pooled = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)  # (batch, hidden_dim)
        return pooled


class BiLSTMBaseline(nn.Module):
    """Bidirectional LSTM baseline over opcode token sequences.

    Architecture:
        opcode token IDs → Embedding → BiLSTM (2 layers) → Attention pooling
        → Dropout → Linear classifier

    Input: opcode_token_ids (batch, seq_len) with PAD_IDX=0 for padding.
    Can also accept pre-extracted opcode_features (batch, 768) as a fallback,
    in which case it uses a projection + the classifier head directly.
    """

    def __init__(self, config: BiLSTMBaselineConfig):
        super().__init__()
        if config.num_labels <= 0:
            raise ValueError("`num_labels` must be positive.")
        if config.hidden_dim <= 0:
            raise ValueError("`hidden_dim` must be positive.")
        if config.embed_dim <= 0:
            raise ValueError("`embed_dim` must be positive.")
        if config.dropout < 0.0 or config.dropout >= 1.0:
            raise ValueError("`dropout` must be in [0.0, 1.0).")

        self.config = BiLSTMBaselineConfig(
            num_labels=int(config.num_labels),
            vocab_size=int(config.vocab_size),
            embed_dim=int(config.embed_dim),
            hidden_dim=int(config.hidden_dim),
            num_layers=int(config.num_layers),
            max_seq_len=int(config.max_seq_len),
            dropout=float(config.dropout),
        )

        bidir_dim = self.config.hidden_dim * 2

        self.embedding = nn.Embedding(
            self.config.vocab_size, self.config.embed_dim, padding_idx=PAD_IDX
        )
        self.lstm = nn.LSTM(
            input_size=self.config.embed_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0.0,
        )
        self.attention_pool = AttentionPooling(bidir_dim)
        self.dropout = nn.Dropout(self.config.dropout)
        self.classifier = nn.Linear(bidir_dim, self.config.num_labels)

    def forward(
        self,
        opcode_features: Optional[torch.Tensor] = None,
        graph_features: Optional[torch.Tensor] = None,
        opcode_token_ids: Optional[torch.Tensor] = None,
        graph_x: Optional[torch.Tensor] = None,
        graph_edge_index: Optional[torch.Tensor] = None,
        graph_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Expects opcode_token_ids (batch, seq_len) as integer token IDs.
        """
        if opcode_token_ids is None:
            raise ValueError(
                "BiLSTM baseline requires `opcode_token_ids`. "
                "Ensure the dataset provides opcode token sequences."
            )

        token_ids = opcode_token_ids.long()
        # Truncate to max_seq_len
        if token_ids.size(1) > self.config.max_seq_len:
            token_ids = token_ids[:, : self.config.max_seq_len]

        mask = token_ids != PAD_IDX  # (batch, seq_len)
        embedded = self.embedding(token_ids)  # (batch, seq_len, embed_dim)
        lstm_out, _ = self.lstm(embedded)  # (batch, seq_len, hidden_dim*2)
        pooled = self.attention_pool(lstm_out, mask)  # (batch, hidden_dim*2)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # (batch, num_labels)
        return logits

    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def branch_dims(self) -> Dict[str, int]:
        return {
            "hidden_dim": int(self.config.hidden_dim),
            "num_labels": int(self.config.num_labels),
            "num_layers": int(self.config.num_layers),
            "embed_dim": int(self.config.embed_dim),
            "vocab_size": int(self.config.vocab_size),
            "max_seq_len": int(self.config.max_seq_len),
        }


def tokenize_opcode_text(text: str, max_len: int = 1024) -> List[int]:
    """Convert space-delimited opcode text to integer token IDs.

    Maps EVM opcode mnemonics to indices 1–256 (0=PAD, 257=UNK).
    Unknown tokens map to UNK_IDX.
    """
    tokens = text.strip().split()[:max_len]
    ids = []
    for tok in tokens:
        # Simple hash-based mapping: known EVM opcodes map to stable IDs
        idx = _OPCODE_TO_ID.get(tok.upper(), UNK_IDX)
        ids.append(idx)
    return ids


# Standard EVM opcode mnemonic to ID mapping (1-indexed, 0=PAD)
_EVM_OPCODES = [
    "STOP", "ADD", "MUL", "SUB", "DIV", "SDIV", "MOD", "SMOD",
    "ADDMOD", "MULMOD", "EXP", "SIGNEXTEND",
    "LT", "GT", "SLT", "SGT", "EQ", "ISZERO", "AND", "OR", "XOR", "NOT",
    "BYTE", "SHL", "SHR", "SAR",
    "SHA3", "KECCAK256",
    "ADDRESS", "BALANCE", "ORIGIN", "CALLER", "CALLVALUE", "CALLDATALOAD",
    "CALLDATASIZE", "CALLDATACOPY", "CODESIZE", "CODECOPY", "GASPRICE",
    "EXTCODESIZE", "EXTCODECOPY", "RETURNDATASIZE", "RETURNDATACOPY",
    "EXTCODEHASH",
    "BLOCKHASH", "COINBASE", "TIMESTAMP", "NUMBER", "DIFFICULTY", "GASLIMIT",
    "CHAINID", "SELFBALANCE", "BASEFEE",
    "POP", "MLOAD", "MSTORE", "MSTORE8", "SLOAD", "SSTORE",
    "JUMP", "JUMPI", "PC", "MSIZE", "GAS", "JUMPDEST",
    "PUSH0",
    "PUSH1", "PUSH2", "PUSH3", "PUSH4", "PUSH5", "PUSH6", "PUSH7", "PUSH8",
    "PUSH9", "PUSH10", "PUSH11", "PUSH12", "PUSH13", "PUSH14", "PUSH15", "PUSH16",
    "PUSH17", "PUSH18", "PUSH19", "PUSH20", "PUSH21", "PUSH22", "PUSH23", "PUSH24",
    "PUSH25", "PUSH26", "PUSH27", "PUSH28", "PUSH29", "PUSH30", "PUSH31", "PUSH32",
    "DUP1", "DUP2", "DUP3", "DUP4", "DUP5", "DUP6", "DUP7", "DUP8",
    "DUP9", "DUP10", "DUP11", "DUP12", "DUP13", "DUP14", "DUP15", "DUP16",
    "SWAP1", "SWAP2", "SWAP3", "SWAP4", "SWAP5", "SWAP6", "SWAP7", "SWAP8",
    "SWAP9", "SWAP10", "SWAP11", "SWAP12", "SWAP13", "SWAP14", "SWAP15", "SWAP16",
    "LOG0", "LOG1", "LOG2", "LOG3", "LOG4",
    "CREATE", "CALL", "CALLCODE", "RETURN", "DELEGATECALL",
    "CREATE2", "STATICCALL",
    "REVERT", "INVALID", "SELFDESTRUCT",
]
_OPCODE_TO_ID = {name: idx + 1 for idx, name in enumerate(_EVM_OPCODES)}
