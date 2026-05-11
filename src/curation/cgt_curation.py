import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _clean(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _runtime_hashes(runtime_dir: Path) -> set:
    hashes = set()
    for file_path in runtime_dir.glob("*"):
        if not file_path.is_file():
            continue
        name = file_path.name
        if name.endswith(".rt.hex"):
            hashes.add(name[:-7])
        else:
            hashes.add(file_path.stem)
    return hashes


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


def _candidate_swcs(df: pd.DataFrame, swc_min: int, swc_max: int) -> List[int]:
    swc_num = pd.to_numeric(_clean(df["swc"]), errors="coerce").astype("Int64")
    mask = swc_num.between(swc_min, swc_max, inclusive="both")
    return sorted(swc_num[mask].dropna().astype(int).unique().tolist())


def curate_cgt(
    csv_path: Path,
    runtime_dir: Path,
    contracts_out: Path,
    labels_out: Path,
    report_out: Path,
    swc_min: int = 100,
    swc_max: int = 136,
) -> Dict[str, Any]:
    df = pd.read_csv(csv_path, sep=";", dtype="string", keep_default_na=False)

    df = df.copy()
    df["fp_runtime_clean"] = _clean(df["fp_runtime"])
    df["swc_clean"] = _clean(df["swc"])
    df["property_holds_clean"] = _clean(df["property_holds"]).str.lower()
    df["swc_id"] = pd.to_numeric(df["swc_clean"], errors="coerce").astype("Int64")

    candidates = _candidate_swcs(df, swc_min=swc_min, swc_max=swc_max)
    if not candidates:
        raise ValueError("No candidate SWCs found in the configured SWC range.")

    candidate_mask = df["swc_id"].isin(candidates)
    runtime_mask = df["fp_runtime_clean"] != ""
    valid_vote_mask = df["property_holds_clean"].isin(["t", "f"])

    filtered = df[candidate_mask & runtime_mask].copy()
    filtered_valid = filtered[filtered["property_holds_clean"].isin(["t", "f"])].copy()

    vote_pairs = (
        filtered_valid.groupby(["fp_runtime_clean", "swc_id"], as_index=False)
        .agg(
            vote_true=("property_holds_clean", lambda s: int((s == "t").sum())),
            vote_false=("property_holds_clean", lambda s: int((s == "f").sum())),
            n_votes=("property_holds_clean", "size"),
        )
        .rename(columns={"fp_runtime_clean": "fp_runtime"})
    )

    vote_pairs["swc_id"] = vote_pairs["swc_id"].astype(int)
    vote_pairs["has_conflict"] = (vote_pairs["vote_true"] > 0) & (vote_pairs["vote_false"] > 0)
    vote_pairs["is_tie"] = (vote_pairs["vote_true"] == vote_pairs["vote_false"]) & (
        vote_pairs["n_votes"] > 0
    )
    vote_pairs["label"] = pd.Series(pd.NA, index=vote_pairs.index, dtype="Int64")
    vote_pairs.loc[vote_pairs["vote_true"] > vote_pairs["vote_false"], "label"] = 1
    vote_pairs.loc[vote_pairs["vote_false"] > vote_pairs["vote_true"], "label"] = 0

    contracts = (
        filtered[["fp_runtime_clean"]]
        .drop_duplicates()
        .rename(columns={"fp_runtime_clean": "fp_runtime"})
        .sort_values("fp_runtime")
        .reset_index(drop=True)
    )

    runtime_hashes = _runtime_hashes(runtime_dir)
    contracts["has_runtime_artifact"] = contracts["fp_runtime"].isin(runtime_hashes)
    contracts["in_runtime_only"] = contracts["has_runtime_artifact"]
    contracts["excluded_from_runtime_only"] = ~contracts["in_runtime_only"]

    assessment_rows = (
        filtered_valid.groupby("fp_runtime_clean").size().rename("candidate_assessment_rows")
    )
    assessed_pairs = vote_pairs.groupby("fp_runtime").size().rename("assessed_swc_pairs")

    contracts = contracts.merge(
        assessment_rows, left_on="fp_runtime", right_index=True, how="left"
    ).merge(assessed_pairs, left_on="fp_runtime", right_index=True, how="left")
    contracts["candidate_assessment_rows"] = (
        contracts["candidate_assessment_rows"].fillna(0).astype(int)
    )
    contracts["assessed_swc_pairs"] = contracts["assessed_swc_pairs"].fillna(0).astype(int)

    swc_frame = pd.DataFrame({"swc_id": candidates})
    cross = contracts[["fp_runtime", "has_runtime_artifact", "in_runtime_only"]].copy()
    cross["__k"] = 1
    swc_frame["__k"] = 1
    labels = cross.merge(swc_frame, on="__k", how="inner").drop(columns="__k")

    labels = labels.merge(
        vote_pairs[
            [
                "fp_runtime",
                "swc_id",
                "label",
                "vote_true",
                "vote_false",
                "n_votes",
                "has_conflict",
                "is_tie",
            ]
        ],
        on=["fp_runtime", "swc_id"],
        how="left",
    )
    labels["vote_true"] = labels["vote_true"].fillna(0).astype(int)
    labels["vote_false"] = labels["vote_false"].fillna(0).astype(int)
    labels["n_votes"] = labels["n_votes"].fillna(0).astype(int)
    labels["label"] = labels["label"].astype("Int64")
    labels["has_conflict"] = labels["has_conflict"].fillna(False).astype(bool)
    labels["is_tie"] = labels["is_tie"].fillna(False).astype(bool)
    labels["is_assessed"] = labels["n_votes"] > 0
    labels = labels.sort_values(["fp_runtime", "swc_id"]).reset_index(drop=True)

    contracts = contracts.sort_values("fp_runtime").reset_index(drop=True)

    _write_parquet(contracts, contracts_out)
    _write_parquet(labels, labels_out)

    swc_report = []
    for swc_id, swc_df in labels.groupby("swc_id", sort=True):
        swc_report.append(
            {
                "swc_id": int(swc_id),
                "total_contract_rows": int(len(swc_df)),
                "assessed_rows": int(swc_df["is_assessed"].sum()),
                "positive_labels": int((swc_df["label"] == 1).sum()),
                "negative_labels": int((swc_df["label"] == 0).sum()),
                "null_labels": int(swc_df["label"].isna().sum()),
                "conflict_rows": int(swc_df["has_conflict"].sum()),
                "tie_rows": int(swc_df["is_tie"].sum()),
            }
        )

    report = {
        "dataset": "CGT",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(csv_path.relative_to(PROJECT_ROOT)),
        "runtime_dir": str(runtime_dir.relative_to(PROJECT_ROOT)),
        "candidate_swc_range": {"min": swc_min, "max": swc_max},
        "candidate_swcs": candidates,
        "row_counts": {
            "input_total_rows": int(len(df)),
            "rows_with_candidate_swc": int(candidate_mask.sum()),
            "rows_excluded_non_candidate_swc": int((~candidate_mask).sum()),
            "rows_with_candidate_swc_and_runtime": int((candidate_mask & runtime_mask).sum()),
            "rows_excluded_missing_fp_runtime": int((candidate_mask & ~runtime_mask).sum()),
            "rows_excluded_invalid_property_holds": int(
                (candidate_mask & runtime_mask & ~valid_vote_mask).sum()
            ),
        },
        "aggregation": {
            "assessed_pairs": int(len(vote_pairs)),
            "pairs_with_multiple_votes": int((vote_pairs["n_votes"] > 1).sum()),
            "pairs_with_conflicts": int(vote_pairs["has_conflict"].sum()),
            "pairs_with_ties": int(vote_pairs["is_tie"].sum()),
            "max_votes_per_pair": int(vote_pairs["n_votes"].max()) if len(vote_pairs) else 0,
        },
        "contracts": {
            "total_contracts": int(len(contracts)),
            "contracts_with_runtime_artifact": int(contracts["has_runtime_artifact"].sum()),
            "contracts_missing_runtime_artifact": int((~contracts["has_runtime_artifact"]).sum()),
            "runtime_only_contracts": int(contracts["in_runtime_only"].sum()),
        },
        "labels_long_format": {
            "total_rows": int(len(labels)),
            "assessed_rows": int(labels["is_assessed"].sum()),
            "not_assessed_rows": int((~labels["is_assessed"]).sum()),
            "positive_labels": int((labels["label"] == 1).sum()),
            "negative_labels": int((labels["label"] == 0).sum()),
            "null_labels": int(labels["label"].isna().sum()),
            "conflict_rows": int(labels["has_conflict"].sum()),
            "tie_rows": int(labels["is_tie"].sum()),
        },
        "runtime_only_variant": {
            "included_label_rows": int(labels["in_runtime_only"].sum()),
            "excluded_label_rows_missing_artifact": int((~labels["in_runtime_only"]).sum()),
        },
        "swc_label_balance": swc_report,
        "outputs": {
            "contracts_parquet": str(contracts_out.relative_to(PROJECT_ROOT)),
            "labels_parquet": str(labels_out.relative_to(PROJECT_ROOT)),
            "report_json": str(report_out.relative_to(PROJECT_ROOT)),
        },
    }

    report_out.parent.mkdir(parents=True, exist_ok=True)
    with report_out.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    return report

