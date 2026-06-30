import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .dappscan_label_semantics import (
    NEGATIVE_SIGNAL_KEYS,
    NEGATIVE_SIGNAL_VALUES,
    generate_dappscan_label_semantics_report,
    negatives_allowed_from_verdict,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DAPPSCAN_ROOT = PROJECT_ROOT / "data/raw/dappscan"
DEFAULT_SEMANTICS_REPORT = PROJECT_ROOT / "reports/phase1/dappscan_label_semantics.md"
DEFAULT_LABELS_OUT = PROJECT_ROOT / "data/intermediate/dappscan_labels.parquet"


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        return str(path)


def _write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(out_path, index=False, engine="pyarrow")
        return
    except Exception as pyarrow_exc:
        try:
            df.to_parquet(out_path, index=False, engine="fastparquet")
            return
        except Exception:
            raise RuntimeError(
                "Unable to write parquet: neither pyarrow nor fastparquet is available."
            ) from pyarrow_exc


def _extract_swc_id(category: str) -> Optional[int]:
    match = re.search(r"SWC-(\d+)", str(category))
    if not match:
        return None
    return int(match.group(1))


def _has_negative_marker(finding: Dict[str, Any]) -> bool:
    for key, value in finding.items():
        key_s = str(key).strip().lower()
        value_s = str(value).strip().lower()
        if key_s in NEGATIVE_SIGNAL_KEYS:
            return True
        if value_s in NEGATIVE_SIGNAL_VALUES:
            return True
    return False


def _read_semantics_verdict(report_path: Path) -> Optional[str]:
    if not report_path.exists():
        return None
    text = report_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"Verdict:\s*`([^`]+)`", text)
    if not match:
        return None
    return str(match.group(1)).strip()


def _resolve_semantics_verdict(
    dappscan_root: Path,
    semantics_report_path: Path,
    sample_dapps: int = 20,
    contracts_per_dapp: int = 5,
    seed: int = 42,
) -> Tuple[str, bool]:
    verdict = _read_semantics_verdict(semantics_report_path)
    if verdict in {"POS_ONLY", "POS+NEG_EXPLICIT"}:
        return verdict, False

    result = generate_dappscan_label_semantics_report(
        dappscan_root=dappscan_root,
        out_path=semantics_report_path,
        sample_dapps=sample_dapps,
        contracts_per_dapp=contracts_per_dapp,
        seed=seed,
    )
    return str(result["verdict"]), True


def run_dappscan_label_extraction(
    dappscan_root: Path,
    labels_out: Path,
    semantics_report_path: Path = DEFAULT_SEMANTICS_REPORT,
    sample_dapps: int = 20,
    contracts_per_dapp: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    swc_root = dappscan_root / "DAppSCAN-bytecode" / "SWCbytecode"
    verdict, semantics_report_regenerated = _resolve_semantics_verdict(
        dappscan_root=dappscan_root,
        semantics_report_path=semantics_report_path,
        sample_dapps=sample_dapps,
        contracts_per_dapp=contracts_per_dapp,
        seed=seed,
    )
    negatives_allowed = negatives_allowed_from_verdict(verdict)

    contract_rows: List[Dict[str, Any]] = []
    pair_counts: Dict[Tuple[str, int], Dict[str, int]] = {}
    report_files_scanned = 0
    parse_errors = 0
    findings_total = 0
    findings_without_swc = 0
    explicit_negative_findings_detected = 0
    explicit_negative_findings_included = 0

    for report_path in sorted(swc_root.rglob("*.json")):
        report_files_scanned += 1
        try:
            rel = report_path.relative_to(swc_root)
            dapp = rel.parts[0] if rel.parts else report_path.parent.name
        except Exception:
            dapp = report_path.parent.name

        try:
            obj = json.loads(report_path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            parse_errors += 1
            continue

        node_name = str(obj.get("node name", report_path.name)).strip()
        contract_name = node_name[:-5] if node_name.lower().endswith(".json") else node_name
        if not contract_name:
            contract_name = report_path.stem
        contract_id = f"{dapp}/{contract_name}"

        contract_rows.append(
            {
                "contract_id": contract_id,
                "dapp": dapp,
                "contract_name": contract_name,
                "report_file": report_path.name,
                "report_path": _rel(report_path),
            }
        )

        swcs = obj.get("SWCs", [])
        if not isinstance(swcs, list):
            continue
        for finding in swcs:
            if not isinstance(finding, dict):
                continue
            findings_total += 1
            swc_id = _extract_swc_id(str(finding.get("category", "")))
            if swc_id is None:
                findings_without_swc += 1
                continue

            pair_key = (contract_id, int(swc_id))
            if pair_key not in pair_counts:
                pair_counts[pair_key] = {"positive_findings": 0, "negative_findings": 0}

            if _has_negative_marker(finding):
                explicit_negative_findings_detected += 1
                if negatives_allowed:
                    pair_counts[pair_key]["negative_findings"] += 1
                    explicit_negative_findings_included += 1
                continue

            pair_counts[pair_key]["positive_findings"] += 1

    contracts_df = pd.DataFrame(contract_rows).drop_duplicates(subset=["contract_id"])
    if contracts_df.empty:
        raise ValueError("No valid SWCbytecode contract reports found.")

    swc_ids = sorted({pair[1] for pair in pair_counts.keys()})
    if not swc_ids:
        raise ValueError("No parseable SWC IDs found in SWCbytecode reports.")

    swc_df = pd.DataFrame({"swc_id": swc_ids})
    contracts_df["__k"] = 1
    swc_df["__k"] = 1
    labels = contracts_df.merge(swc_df, on="__k", how="inner").drop(columns="__k")

    pair_rows = [
        {
            "contract_id": contract_id,
            "swc_id": int(swc_id),
            "positive_findings": int(counts["positive_findings"]),
            "negative_findings": int(counts["negative_findings"]),
        }
        for (contract_id, swc_id), counts in pair_counts.items()
    ]
    pair_df = pd.DataFrame(pair_rows)
    if pair_df.empty:
        pair_df = pd.DataFrame(
            columns=["contract_id", "swc_id", "positive_findings", "negative_findings"]
        )

    labels = labels.merge(pair_df, on=["contract_id", "swc_id"], how="left")
    labels["positive_findings"] = labels["positive_findings"].fillna(0).astype(int)
    labels["negative_findings"] = labels["negative_findings"].fillna(0).astype(int)
    labels["is_positive_evidence"] = labels["positive_findings"] > 0
    labels["is_negative_evidence"] = labels["negative_findings"] > 0
    labels["is_assessed"] = labels["is_positive_evidence"] | labels["is_negative_evidence"]
    labels["has_conflict"] = labels["is_positive_evidence"] & labels["is_negative_evidence"]
    labels["label"] = pd.Series(pd.NA, index=labels.index, dtype="Int64")
    labels.loc[
        labels["is_positive_evidence"] & ~labels["is_negative_evidence"],
        "label",
    ] = 1
    labels.loc[
        labels["is_negative_evidence"] & ~labels["is_positive_evidence"],
        "label",
    ] = 0
    labels["semantics_verdict"] = verdict
    labels["negatives_allowed"] = bool(negatives_allowed)
    labels = labels.sort_values(["dapp", "contract_name", "swc_id"]).reset_index(drop=True)

    _write_parquet(labels, labels_out)

    swc_distribution = []
    for swc_id, swc_frame in labels.groupby("swc_id", sort=True):
        swc_distribution.append(
            {
                "swc_id": int(swc_id),
                "positive": int((swc_frame["label"] == 1).sum()),
                "negative": int((swc_frame["label"] == 0).sum()),
                "unlabeled": int(swc_frame["label"].isna().sum()),
            }
        )

    summary = {
        "dataset": "DAppSCAN-bytecode",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "paths": {
            "dappscan_root": _rel(dappscan_root),
            "swc_root": _rel(swc_root),
            "semantics_report": _rel(semantics_report_path),
            "labels_parquet": _rel(labels_out),
        },
        "semantics": {
            "verdict": verdict,
            "negatives_allowed": bool(negatives_allowed),
            "report_regenerated": bool(semantics_report_regenerated),
        },
        "counts": {
            "report_files_scanned": int(report_files_scanned),
            "parse_errors": int(parse_errors),
            "contracts": int(labels["contract_id"].nunique()),
            "swc_ids": int(labels["swc_id"].nunique()),
            "rows": int(len(labels)),
            "positive_rows": int((labels["label"] == 1).sum()),
            "negative_rows": int((labels["label"] == 0).sum()),
            "unlabeled_rows": int(labels["label"].isna().sum()),
            "findings_total": int(findings_total),
            "findings_without_swc_id": int(findings_without_swc),
            "explicit_negative_findings_detected": int(explicit_negative_findings_detected),
            "explicit_negative_findings_included": int(explicit_negative_findings_included),
        },
        "class_distribution_by_swc": swc_distribution,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract DAppSCAN SWC bytecode labels.")
    parser.add_argument("--dappscan-root", type=Path, default=DEFAULT_DAPPSCAN_ROOT)
    parser.add_argument("--labels-out", type=Path, default=DEFAULT_LABELS_OUT)
    parser.add_argument("--semantics-report", type=Path, default=DEFAULT_SEMANTICS_REPORT)
    parser.add_argument("--sample-dapps", type=int, default=20)
    parser.add_argument("--contracts-per-dapp", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = run_dappscan_label_extraction(
        dappscan_root=args.dappscan_root,
        labels_out=args.labels_out,
        semantics_report_path=args.semantics_report,
        sample_dapps=args.sample_dapps,
        contracts_per_dapp=args.contracts_per_dapp,
        seed=args.seed,
    )
    print(f"Semantics verdict: {summary['semantics']['verdict']}")
    print(f"Negatives allowed: {summary['semantics']['negatives_allowed']}")
    print(
        f"Label rows: {summary['counts']['rows']} "
        f"(+{summary['counts']['positive_rows']}, "
        f"-{summary['counts']['negative_rows']}, "
        f"unlabeled={summary['counts']['unlabeled_rows']})"
    )
    print(f"Labels parquet: {args.labels_out}")


if __name__ == "__main__":
    main()
