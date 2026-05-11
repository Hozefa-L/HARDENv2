import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UNIFIED_LABELS = PROJECT_ROOT / "data/curated/unified_labels.parquet"
DEFAULT_UNIFIED_CONTRACTS = PROJECT_ROOT / "data/curated/unified_contracts.parquet"
DEFAULT_SWC_DECISION_MATRIX = PROJECT_ROOT / "reports/phase1/swc_decision_matrix.parquet"
DEFAULT_CGT_CONTRACTS_FP = PROJECT_ROOT / "data/intermediate/cgt_contracts_fp.parquet"
DEFAULT_CGT_CSV = PROJECT_ROOT / "data/raw/cgt-main/consolidated.csv"
DEFAULT_SPLITS_ROOT = PROJECT_ROOT / "data/splits"
DEFAULT_SPLIT_STATS_OUT = PROJECT_ROOT / "reports/phase1/split_stats.json"


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        return str(path)


def _write_json(payload: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def _write_ids(ids: Sequence[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(str(v) for v in ids)
    if text:
        text += "\n"
    out_path.write_text(text, encoding="utf-8")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip()).strip("_").lower()
    return slug or "dataset"


def _clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _parse_json_list(text: Any) -> List[str]:
    raw = "" if text is None else str(text).strip()
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except Exception:
        return []
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


def _selected_swcs(unified_labels: pd.DataFrame, swc_decision_matrix: Optional[pd.DataFrame]) -> List[int]:
    observed = sorted(
        {
            int(v)
            for v in pd.to_numeric(unified_labels["swc_id"], errors="coerce").dropna().astype(int).tolist()
        }
    )
    if swc_decision_matrix is None or swc_decision_matrix.empty:
        return observed
    matrix = swc_decision_matrix.copy()
    if "swc_id" not in matrix.columns:
        return observed
    matrix["swc_id"] = pd.to_numeric(matrix["swc_id"], errors="coerce").astype("Int64")
    matrix = matrix[matrix["swc_id"].notna()].copy()
    matrix["swc_id"] = matrix["swc_id"].astype(int)
    if "action" not in matrix.columns:
        return sorted(set(matrix["swc_id"].tolist()) & set(observed))
    keep_actions = {"keep", "keep_cgt_only", "keep_pu_only"}
    selected = matrix[matrix["action"].isin(keep_actions)]["swc_id"].tolist()
    selected = sorted(set(selected) & set(observed))
    return selected or observed


def _aggregate_pair_labels(labels: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    grouped = (
        labels.groupby(["fp_runtime_unified", "swc_id"])["label"]
        .agg(
            has_positive=lambda s: bool((s == 1).any()),
            has_negative=lambda s: bool((s == 0).any()),
        )
        .reset_index()
    )
    grouped["has_positive"] = grouped["has_positive"].fillna(0).astype(bool)
    grouped["has_negative"] = grouped["has_negative"].fillna(0).astype(bool)
    grouped["label"] = pd.Series(pd.NA, index=grouped.index, dtype="Int64")
    grouped.loc[grouped["has_positive"] & ~grouped["has_negative"], "label"] = 1
    grouped.loc[grouped["has_negative"] & ~grouped["has_positive"], "label"] = 0
    conflicts = int((grouped["has_positive"] & grouped["has_negative"]).sum())
    return grouped, conflicts


def _primary_split(
    ids: List[str],
    y: np.ndarray,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    n = len(ids)
    if n < 3:
        raise ValueError("Need at least 3 contracts with known labels for train/val/test splitting.")

    outer_splits = 10 if n >= 10 else max(2, n)
    outer = MultilabelStratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)
    train_val_idx, test_idx = next(outer.split(np.zeros((n, 1)), y))

    train_val_ids = [ids[i] for i in train_val_idx]
    y_train_val = y[train_val_idx]
    inner_n = len(train_val_ids)
    inner_splits = 9 if inner_n >= 9 else max(2, inner_n)
    inner = MultilabelStratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    train_rel_idx, val_rel_idx = next(inner.split(np.zeros((inner_n, 1)), y_train_val))

    train_ids = sorted(train_val_ids[i] for i in train_rel_idx)
    val_ids = sorted(train_val_ids[i] for i in val_rel_idx)
    test_ids = sorted(ids[i] for i in test_idx)
    return train_ids, val_ids, test_ids


def _label_stats_for_ids(
    label_matrix: pd.DataFrame,
    ids: Sequence[str],
) -> Dict[str, Dict[str, int]]:
    if not ids:
        return {
            str(swc): {"known": 0, "positive": 0, "negative": 0}
            for swc in label_matrix.columns.tolist()
        }
    sub = label_matrix.loc[list(ids)]
    stats: Dict[str, Dict[str, int]] = {}
    for swc in sub.columns:
        col = sub[swc]
        stats[str(swc)] = {
            "known": int(col.notna().sum()),
            "positive": int((col == 1).sum()),
            "negative": int((col == 0).sum()),
        }
    return stats


def _build_leave_one_dataset_splits(
    universe_ids: Sequence[str],
    unified_contracts_path: Path,
    cgt_contracts_fp_path: Path,
    cgt_csv_path: Path,
) -> Dict[str, Dict[str, List[str]]]:
    if not unified_contracts_path.exists() or not cgt_csv_path.exists():
        return {}

    unified_contracts = pd.read_parquet(unified_contracts_path)
    if "fp_runtime_unified" not in unified_contracts.columns or "cgt_fp_runtime_ids" not in unified_contracts.columns:
        return {}

    mapping_rows: List[Dict[str, str]] = []
    for _, row in unified_contracts.iterrows():
        fp_unified = str(row.get("fp_runtime_unified", "")).strip()
        if not fp_unified:
            continue
        cgt_ids = _parse_json_list(row.get("cgt_fp_runtime_ids"))
        for cgt_fp in cgt_ids:
            mapping_rows.append({"fp_runtime_unified": fp_unified, "fp_runtime": cgt_fp})
    if not mapping_rows:
        return {}

    fp_map = pd.DataFrame(mapping_rows).drop_duplicates()

    cgt_csv = pd.read_csv(
        cgt_csv_path,
        sep=";",
        dtype="string",
        keep_default_na=False,
        usecols=["fp_runtime", "dataset"],
    )
    cgt_csv["fp_runtime"] = _clean_text(cgt_csv["fp_runtime"])
    cgt_csv["dataset"] = _clean_text(cgt_csv["dataset"])
    cgt_csv = cgt_csv[(cgt_csv["fp_runtime"] != "") & (cgt_csv["dataset"] != "")].drop_duplicates()

    # Optional sanity: restrict fp_runtime universe to those seen in current CGT contracts FP.
    if cgt_contracts_fp_path.exists():
        cgt_contracts_fp = pd.read_parquet(cgt_contracts_fp_path)
        if "fp_runtime" in cgt_contracts_fp.columns:
            allowed = set(_clean_text(cgt_contracts_fp["fp_runtime"]).tolist())
            cgt_csv = cgt_csv[cgt_csv["fp_runtime"].isin(allowed)].copy()

    joined = fp_map.merge(cgt_csv, on="fp_runtime", how="left")
    joined = joined[joined["dataset"].notna() & (joined["dataset"] != "")].copy()
    if joined.empty:
        return {}

    universe = set(universe_ids)
    dataset_to_ids: Dict[str, set] = {}
    for dataset, frame in joined.groupby("dataset"):
        ids = set(frame["fp_runtime_unified"].astype(str).tolist()) & universe
        if ids:
            dataset_to_ids[str(dataset)] = ids

    result: Dict[str, Dict[str, List[str]]] = {}
    all_ids = set(universe_ids)
    for dataset, test_set in sorted(dataset_to_ids.items()):
        test_ids = sorted(test_set)
        train_ids = sorted(all_ids - test_set)
        result[dataset] = {"train_ids": train_ids, "val_ids": [], "test_ids": test_ids}
    return result


def run_splits(
    unified_labels_path: Path,
    splits_root: Path,
    split_stats_out: Path,
    swc_decision_matrix_path: Optional[Path] = DEFAULT_SWC_DECISION_MATRIX,
    unified_contracts_path: Optional[Path] = DEFAULT_UNIFIED_CONTRACTS,
    cgt_contracts_fp_path: Optional[Path] = DEFAULT_CGT_CONTRACTS_FP,
    cgt_csv_path: Optional[Path] = DEFAULT_CGT_CSV,
    seed: int = 42,
    cv_folds: int = 5,
    generate_cv: bool = True,
) -> Dict[str, Any]:
    labels = pd.read_parquet(unified_labels_path).copy()
    required = {"fp_runtime_unified", "swc_id", "label"}
    missing = [column for column in sorted(required) if column not in labels.columns]
    if missing:
        raise ValueError(f"Unified labels missing required columns: {missing}")

    decision_matrix = None
    if swc_decision_matrix_path and swc_decision_matrix_path.exists():
        decision_matrix = pd.read_parquet(swc_decision_matrix_path)

    labels["fp_runtime_unified"] = _clean_text(labels["fp_runtime_unified"])
    labels["swc_id"] = pd.to_numeric(labels["swc_id"], errors="coerce").astype("Int64")
    labels["label"] = pd.to_numeric(labels["label"], errors="coerce").astype("Int64")
    labels = labels[(labels["fp_runtime_unified"] != "") & labels["swc_id"].notna()].copy()
    labels["swc_id"] = labels["swc_id"].astype(int)

    selected_swcs = _selected_swcs(labels, decision_matrix)
    labels = labels[labels["swc_id"].isin(selected_swcs)].copy()
    known_labels = labels[labels["label"].notna()].copy()

    aggregated_pairs, conflict_pairs = _aggregate_pair_labels(known_labels)
    pivot = aggregated_pairs.pivot(index="fp_runtime_unified", columns="swc_id", values="label")
    pivot = pivot.reindex(columns=selected_swcs)
    eligible = pivot[pivot.notna().any(axis=1)].copy()
    if eligible.empty:
        raise ValueError("No contracts with known labels found for selected SWCs.")

    contract_ids = eligible.index.astype(str).tolist()
    y = eligible.fillna(0).astype(int).to_numpy()

    train_ids, val_ids, test_ids = _primary_split(contract_ids, y, seed=seed)

    primary_root = splits_root / "primary"
    _write_ids(train_ids, primary_root / "train_ids.txt")
    _write_ids(val_ids, primary_root / "val_ids.txt")
    _write_ids(test_ids, primary_root / "test_ids.txt")

    cv_summary: Dict[str, Any] = {"enabled": bool(generate_cv), "folds": []}
    if generate_cv:
        n = len(contract_ids)
        n_folds = max(2, min(int(cv_folds), n))
        cv = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold_idx, (tr_idx, te_idx) in enumerate(cv.split(np.zeros((n, 1)), y), start=1):
            fold_train = sorted(contract_ids[i] for i in tr_idx)
            fold_test = sorted(contract_ids[i] for i in te_idx)
            fold_root = splits_root / f"cv_fold_{fold_idx}"
            _write_ids(fold_train, fold_root / "train_ids.txt")
            _write_ids([], fold_root / "val_ids.txt")
            _write_ids(fold_test, fold_root / "test_ids.txt")
            cv_summary["folds"].append(
                {
                    "fold": int(fold_idx),
                    "train_size": int(len(fold_train)),
                    "test_size": int(len(fold_test)),
                }
            )

    loocgt_summary: Dict[str, Any] = {"splits": []}
    if unified_contracts_path and cgt_csv_path and cgt_contracts_fp_path:
        leave_one = _build_leave_one_dataset_splits(
            universe_ids=contract_ids,
            unified_contracts_path=unified_contracts_path,
            cgt_contracts_fp_path=cgt_contracts_fp_path,
            cgt_csv_path=cgt_csv_path,
        )
    else:
        leave_one = {}

    for dataset, split in leave_one.items():
        split_name = f"leave_one_cgt_{_slugify(dataset)}"
        split_root = splits_root / split_name
        _write_ids(split["train_ids"], split_root / "train_ids.txt")
        _write_ids(split["val_ids"], split_root / "val_ids.txt")
        _write_ids(split["test_ids"], split_root / "test_ids.txt")
        loocgt_summary["splits"].append(
            {
                "dataset": dataset,
                "split_name": split_name,
                "train_size": int(len(split["train_ids"])),
                "test_size": int(len(split["test_ids"])),
            }
        )

    split_stats = {
        "inputs": {
            "unified_labels": _rel(unified_labels_path),
            "swc_decision_matrix": _rel(swc_decision_matrix_path)
            if swc_decision_matrix_path and swc_decision_matrix_path.exists()
            else None,
            "unified_contracts": _rel(unified_contracts_path)
            if unified_contracts_path and unified_contracts_path.exists()
            else None,
            "cgt_contracts_fp": _rel(cgt_contracts_fp_path)
            if cgt_contracts_fp_path and cgt_contracts_fp_path.exists()
            else None,
            "cgt_csv": _rel(cgt_csv_path) if cgt_csv_path and cgt_csv_path.exists() else None,
        },
        "selected_swcs": selected_swcs,
        "seed": int(seed),
        "contracts_with_known_labels": int(len(contract_ids)),
        "pair_label_conflicts_across_sources": int(conflict_pairs),
        "primary_split": {
            "train_size": int(len(train_ids)),
            "val_size": int(len(val_ids)),
            "test_size": int(len(test_ids)),
            "train_ratio": round(len(train_ids) / len(contract_ids), 6),
            "val_ratio": round(len(val_ids) / len(contract_ids), 6),
            "test_ratio": round(len(test_ids) / len(contract_ids), 6),
            "label_stats": {
                "train": _label_stats_for_ids(eligible, train_ids),
                "val": _label_stats_for_ids(eligible, val_ids),
                "test": _label_stats_for_ids(eligible, test_ids),
            },
        },
        "cv": cv_summary,
        "leave_one_cgt_dataset_out": loocgt_summary,
        "outputs": {
            "splits_root": _rel(splits_root),
            "split_stats_json": _rel(split_stats_out),
        },
    }
    _write_json(split_stats, split_stats_out)
    return split_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multilabel stratified splits for unified runtime dataset.")
    parser.add_argument("--unified-labels", type=Path, default=DEFAULT_UNIFIED_LABELS)
    parser.add_argument("--swc-decision-matrix", type=Path, default=DEFAULT_SWC_DECISION_MATRIX)
    parser.add_argument("--unified-contracts", type=Path, default=DEFAULT_UNIFIED_CONTRACTS)
    parser.add_argument("--cgt-contracts-fp", type=Path, default=DEFAULT_CGT_CONTRACTS_FP)
    parser.add_argument("--cgt-csv", type=Path, default=DEFAULT_CGT_CSV)
    parser.add_argument("--splits-root", type=Path, default=DEFAULT_SPLITS_ROOT)
    parser.add_argument("--split-stats-out", type=Path, default=DEFAULT_SPLIT_STATS_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--no-cv", action="store_true")
    args = parser.parse_args()

    stats = run_splits(
        unified_labels_path=args.unified_labels,
        swc_decision_matrix_path=args.swc_decision_matrix,
        unified_contracts_path=args.unified_contracts,
        cgt_contracts_fp_path=args.cgt_contracts_fp,
        cgt_csv_path=args.cgt_csv,
        splits_root=args.splits_root,
        split_stats_out=args.split_stats_out,
        seed=args.seed,
        cv_folds=args.cv_folds,
        generate_cv=not args.no_cv,
    )
    print(f"Contracts with known labels: {stats['contracts_with_known_labels']}")
    print(f"Primary split train/val/test: {stats['primary_split']['train_size']}/"
          f"{stats['primary_split']['val_size']}/{stats['primary_split']['test_size']}")
    print(f"Splits root: {args.splits_root}")
    print(f"Split stats: {args.split_stats_out}")


if __name__ == "__main__":
    main()
