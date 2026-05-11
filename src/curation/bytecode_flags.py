import argparse
import json
import re
from collections import Counter
from typing import Any, Dict, Tuple

from .bytecode_normalize import normalize_hex

EIP1167_RUNTIME_PATTERN = re.compile(
    r"^363d3d373d3d3d363d73[0-9a-f]{40}5af43d82803e903d91602b57fd5bf3$"
)


def opcode_histogram(runtime_hex: Any) -> Tuple[Counter, int]:
    normalized = normalize_hex(runtime_hex)
    data = bytes.fromhex(normalized) if normalized else b""
    counts: Counter = Counter()
    opcode_count = 0
    pc = 0
    while pc < len(data):
        opcode = data[pc]
        counts[opcode] += 1
        opcode_count += 1
        if 0x60 <= opcode <= 0x7F:
            pc += 1 + (opcode - 0x5F)
        else:
            pc += 1
    return counts, opcode_count


def is_eip1167_minimal_proxy(runtime_hex: Any) -> bool:
    normalized = normalize_hex(runtime_hex)
    return bool(EIP1167_RUNTIME_PATTERN.fullmatch(normalized))


def compute_delegatecall_ratio(runtime_hex: Any) -> Tuple[float, int, int]:
    counts, opcode_count = opcode_histogram(runtime_hex)
    delegatecall_count = int(counts.get(0xF4, 0))
    ratio = (delegatecall_count / opcode_count) if opcode_count else 0.0
    return float(ratio), delegatecall_count, opcode_count


def compute_bytecode_flags(
    runtime_hex: Any,
    stub_threshold_bytes: int = 100,
    delegatecall_proxy_threshold: float = 0.02,
) -> Dict[str, Any]:
    normalized = normalize_hex(runtime_hex)
    runtime_size_bytes = int(len(normalized) // 2)
    delegatecall_ratio, delegatecall_count, opcode_count = compute_delegatecall_ratio(normalized)
    eip1167_proxy = is_eip1167_minimal_proxy(normalized)
    is_proxy_like = bool(
        eip1167_proxy or (delegatecall_count > 0 and delegatecall_ratio >= delegatecall_proxy_threshold)
    )
    is_stub_like = bool(runtime_size_bytes < stub_threshold_bytes)
    return {
        "runtime_size_bytes": runtime_size_bytes,
        "is_proxy_like": is_proxy_like,
        "is_eip1167_proxy": eip1167_proxy,
        "is_stub_like": is_stub_like,
        "delegatecall_ratio": round(delegatecall_ratio, 6),
        "delegatecall_count": int(delegatecall_count),
        "opcode_count": int(opcode_count),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute graph-quality flags from runtime bytecode.")
    parser.add_argument("--hex", required=True, help="Runtime bytecode hex string.")
    parser.add_argument("--stub-threshold-bytes", type=int, default=100)
    parser.add_argument("--delegatecall-proxy-threshold", type=float, default=0.02)
    args = parser.parse_args()
    result = compute_bytecode_flags(
        runtime_hex=args.hex,
        stub_threshold_bytes=args.stub_threshold_bytes,
        delegatecall_proxy_threshold=args.delegatecall_proxy_threshold,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
