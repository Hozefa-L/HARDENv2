import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UNIFIED_LABELS = PROJECT_ROOT / "data/curated/unified_labels.parquet"
DEFAULT_DECISION_MATRIX_OUT = PROJECT_ROOT / "reports/phase1/swc_decision_matrix.parquet"
DEFAULT_DECISION_REPORT_OUT = PROJECT_ROOT / "reports/phase1/swc_selection_report.json"


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


def _write_json(payload: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def _parse_candidates(
    labels_df: pd.DataFrame,
    swc_candidates: Optional[str],
    swc_min: int,
    swc_max: int,
) -> List[int]:
    if swc_candidates:
        parsed = sorted(
            {
                int(token.strip())
                for token in str(swc_candidates).split(",")
                if token is not None and str(token).strip()
            }
        )
        if parsed:
            return parsed
    observed = pd.to_numeric(labels_df["swc_id"], errors="coerce").dropna().astype(int)
    if observed.empty:
        return []
    in_range = sorted({swc for swc in observed.tolist() if swc_min <= swc <= swc_max})
    return in_range


def _decide_action(
    cgt_pos: int,
    cgt_neg: int,
    dapp_pos: int,
    dapp_neg: int,
    min_total_known: int,
) -> str:
    cgt_known = int(cgt_pos + cgt_neg)
    dapp_known = int(dapp_pos + dapp_neg)
    total_known = int(cgt_known + dapp_known)

    if total_known < min_total_known:
        return "drop_low_count"
    if cgt_known > 0 and dapp_pos == 0 and dapp_neg == 0:
        return "keep_cgt_only"
    if cgt_known == 0 and dapp_pos > 0:
        return "keep_pu_only"
    return "keep"


def run_swc_selection(
    unified_labels_path: Path,
    decision_matrix_out: Path,
    report_out: Path = DEFAULT_DECISION_REPORT_OUT,
    swc_candidates: Optional[str] = None,
    swc_min: int = 100,
    swc_max: int = 136,
    min_total_known: int = 20,
) -> Dict[str, Any]:
    labels = pd.read_parquet(unified_labels_path).copy()
    required_columns = {"swc_id", "source", "label"}
    missing = [column for column in sorted(required_columns) if column not in labels.columns]
    if missing:
        raise ValueError(f"Unified labels missing required columns: {missing}")

    labels["swc_id"] = pd.to_numeric(labels["swc_id"], errors="coerce").astype("Int64")
    labels = labels[labels["swc_id"].notna()].copy()
    labels["swc_id"] = labels["swc_id"].astype(int)
    labels["source"] = labels["source"].fillna("").astype(str).str.strip().str.lower()
    labels["label"] = pd.to_numeric(labels["label"], errors="coerce").astype("Int64")

    candidates = _parse_candidates(labels, swc_candidates=swc_candidates, swc_min=swc_min, swc_max=swc_max)
    if not candidates:
        raise ValueError("No SWC candidates available for selection.")

    stats_rows = []
    for swc_id in candidates:
        swc_frame = labels[labels["swc_id"] == swc_id]
        cgt_frame = swc_frame[swc_frame["source"] == "cgt"]
        dapp_frame = swc_frame[swc_frame["source"] == "dappscan"]

        cgt_pos = int((cgt_frame["label"] == 1).sum())
        cgt_neg = int((cgt_frame["label"] == 0).sum())
        dapp_pos = int((dapp_frame["label"] == 1).sum())
        dapp_neg = int((dapp_frame["label"] == 0).sum())
        cgt_known = cgt_pos + cgt_neg
        dapp_known = dapp_pos + dapp_neg
        action = _decide_action(
            cgt_pos=cgt_pos,
            cgt_neg=cgt_neg,
            dapp_pos=dapp_pos,
            dapp_neg=dapp_neg,
            min_total_known=min_total_known,
        )
        stats_rows.append(
            {
                "swc_id": int(swc_id),
                "cgt_positive": int(cgt_pos),
                "cgt_negative": int(cgt_neg),
                "cgt_known_total": int(cgt_known),
                "dappscan_positive": int(dapp_pos),
                "dappscan_negative": int(dapp_neg),
                "dappscan_known_total": int(dapp_known),
                "known_total": int(cgt_known + dapp_known),
                "action": action,
            }
        )

    matrix = pd.DataFrame(stats_rows).sort_values("swc_id").reset_index(drop=True)
    _write_parquet(matrix, decision_matrix_out)

    action_counts = matrix["action"].value_counts().to_dict()
    report = {
        "inputs": {
            "unified_labels": _rel(unified_labels_path),
        },
        "selection_policy": {
            "swc_candidates": candidates,
            "swc_min": int(swc_min),
            "swc_max": int(swc_max),
            "min_total_known": int(min_total_known),
            "actions": ["keep", "drop_low_count", "keep_cgt_only", "keep_pu_only"],
        },
        "counts": {
            "candidate_swcs": int(len(matrix)),
            "action_distribution": {str(k): int(v) for k, v in action_counts.items()},
        },
        "outputs": {
            "swc_decision_matrix_parquet": _rel(decision_matrix_out),
            "swc_selection_report_json": _rel(report_out),
        },
    }
    _write_json(report, report_out)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute SWC decision matrix from harmonized labels.")
    parser.add_argument("--unified-labels", type=Path, default=DEFAULT_UNIFIED_LABELS)
    parser.add_argument("--decision-matrix-out", type=Path, default=DEFAULT_DECISION_MATRIX_OUT)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_DECISION_REPORT_OUT)
    parser.add_argument("--swc-candidates", type=str, default=None)
    parser.add_argument("--swc-min", type=int, default=100)
    parser.add_argument("--swc-max", type=int, default=136)
    parser.add_argument("--min-total-known", type=int, default=20)
    args = parser.parse_args()

    report = run_swc_selection(
        unified_labels_path=args.unified_labels,
        decision_matrix_out=args.decision_matrix_out,
        report_out=args.report_out,
        swc_candidates=args.swc_candidates,
        swc_min=args.swc_min,
        swc_max=args.swc_max,
        min_total_known=args.min_total_known,
    )
    print(f"SWC candidates: {report['counts']['candidate_swcs']}")
    print(f"Decision matrix: {args.decision_matrix_out}")


if __name__ == "__main__":
    main()
