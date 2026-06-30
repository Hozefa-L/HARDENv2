import argparse
import json
import re
from typing import Any, Dict, List, Optional, Tuple

HEX_RE = re.compile(r"^[0-9a-f]*$")
METADATA_MARKERS = (b"ipfs", b"bzzr0", b"bzzr1", b"bzzr", b"solc", b"experimental")


def normalize_hex(hex_value: Any) -> str:
    text = "" if hex_value is None else str(hex_value)
    normalized = "".join(text.strip().split()).lower()
    if normalized.startswith("0x"):
        normalized = normalized[2:]
    if not normalized:
        return ""
    if len(normalized) % 2 != 0:
        raise ValueError("Hex string has odd length.")
    if not HEX_RE.fullmatch(normalized):
        raise ValueError("Hex string contains non-hex characters.")
    return normalized


def _decode_cbor_header(blob: bytes, offset: int = 0) -> Optional[Tuple[int, int, int]]:
    if offset >= len(blob):
        return None
    head = blob[offset]
    major = head >> 5
    addl = head & 0x1F
    if addl < 24:
        return major, addl, 1
    if addl == 24 and offset + 1 < len(blob):
        return major, int(blob[offset + 1]), 2
    if addl == 25 and offset + 2 < len(blob):
        return major, int.from_bytes(blob[offset + 1 : offset + 3], "big"), 3
    if addl == 26 and offset + 4 < len(blob):
        return major, int.from_bytes(blob[offset + 1 : offset + 5], "big"), 5
    if addl == 27 and offset + 8 < len(blob):
        return major, int.from_bytes(blob[offset + 1 : offset + 9], "big"), 9
    return None


def _looks_like_solc_metadata(metadata_blob: bytes) -> Tuple[bool, List[str]]:
    markers_found = [marker.decode("ascii") for marker in METADATA_MARKERS if marker in metadata_blob]
    if not metadata_blob:
        return False, markers_found
    decoded = _decode_cbor_header(metadata_blob, 0)
    if decoded is None:
        return False, markers_found
    major_type, _, _ = decoded
    is_map = major_type == 5
    return bool(is_map and markers_found), markers_found


def strip_solc_metadata(runtime_hex: Any, mode: str = "strip") -> Tuple[str, Dict[str, Any]]:
    if mode not in {"strip", "zero"}:
        raise ValueError("mode must be 'strip' or 'zero'.")
    normalized = normalize_hex(runtime_hex)
    if not normalized:
        return "", {
            "metadata_detected": False,
            "metadata_removed": False,
            "metadata_len_bytes": 0,
            "markers_found": [],
            "metadata_mode": mode,
        }

    runtime_bytes = bytes.fromhex(normalized)
    if len(runtime_bytes) < 2:
        return normalized, {
            "metadata_detected": False,
            "metadata_removed": False,
            "metadata_len_bytes": 0,
            "markers_found": [],
            "metadata_mode": mode,
        }

    metadata_len = int.from_bytes(runtime_bytes[-2:], "big")
    if metadata_len <= 0 or metadata_len + 2 > len(runtime_bytes):
        return normalized, {
            "metadata_detected": False,
            "metadata_removed": False,
            "metadata_len_bytes": 0,
            "markers_found": [],
            "metadata_mode": mode,
        }

    metadata_start = len(runtime_bytes) - metadata_len - 2
    metadata_blob = runtime_bytes[metadata_start:-2]
    looks_like_metadata, markers_found = _looks_like_solc_metadata(metadata_blob)
    if not looks_like_metadata:
        return normalized, {
            "metadata_detected": False,
            "metadata_removed": False,
            "metadata_len_bytes": 0,
            "markers_found": markers_found,
            "metadata_mode": mode,
        }

    if mode == "strip":
        out_bytes = runtime_bytes[:metadata_start]
    else:
        out_bytes = runtime_bytes[:metadata_start] + b"\x00" * (metadata_len + 2)

    return out_bytes.hex(), {
        "metadata_detected": True,
        "metadata_removed": True,
        "metadata_len_bytes": int(metadata_len + 2),
        "metadata_start_offset": int(metadata_start),
        "markers_found": markers_found,
        "metadata_mode": mode,
    }


def canonicalize_runtime_hex(runtime_hex: Any, mode: str = "strip") -> Tuple[str, Dict[str, Any]]:
    raw_normalized = normalize_hex(runtime_hex)
    canonical, metadata_info = strip_solc_metadata(raw_normalized, mode=mode)
    info = {
        "raw_length_bytes": int(len(raw_normalized) // 2),
        "canonical_length_bytes": int(len(canonical) // 2),
    }
    info.update(metadata_info)
    return canonical, info


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize and strip Solidity runtime metadata.")
    parser.add_argument("--hex", required=True, help="Runtime bytecode hex string.")
    parser.add_argument("--mode", choices=["strip", "zero"], default="strip")
    args = parser.parse_args()
    normalized, info = canonicalize_runtime_hex(args.hex, mode=args.mode)
    payload = {"normalized_runtime_hex": normalized, "info": info}
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
