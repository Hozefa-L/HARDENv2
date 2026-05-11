from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHASE3_CONFIG_PATH = PROJECT_ROOT / "configs/phase3.yaml"
CONTRACT_ID_COLUMN = "fp_runtime_unified"


@dataclass(frozen=True)
class Phase3DatasetConfig:
    swc_ids: List[int]
    feature_index_path: Path
    codebert_features_path: Path
    graph_features_path: Path
    opcode_text_path: Path


def _resolve_path(path_value: Any) -> Path:
    raw = Path(str(path_value))
    if raw.is_absolute():
        return raw
    return (PROJECT_ROOT / raw).resolve()


def _load_phase3_dataset_config(config_path: Path) -> Phase3DatasetConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"Phase 3 config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp)
    if not isinstance(raw, dict):
        raise ValueError("Phase 3 config must be a mapping.")

    benchmark = raw.get("main_benchmark", {})
    outputs = raw.get("outputs", {})
    if benchmark is None:
        benchmark = {}
    if outputs is None:
        outputs = {}
    if not isinstance(benchmark, dict) or not isinstance(outputs, dict):
        raise ValueError("`main_benchmark` and `outputs` must both be mappings.")

    swc_values = benchmark.get("swc_ids", [])
    if not isinstance(swc_values, list) or not swc_values:
        raise ValueError("`main_benchmark.swc_ids` must be a non-empty list.")
    swc_ids = [int(value) for value in swc_values]

    return Phase3DatasetConfig(
        swc_ids=swc_ids,
        feature_index_path=_resolve_path(
            outputs.get("feature_index_parquet") or "data/features/main_benchmark/phase3_feature_index.parquet"
        ),
        codebert_features_path=_resolve_path(
            outputs.get("codebert_features_parquet") or "data/features/main_benchmark/codebert_features.parquet"
        ),
        graph_features_path=_resolve_path(
            outputs.get("graph_level_features_parquet") or "data/features/main_benchmark/graph_level_features.parquet"
        ),
        opcode_text_path=_resolve_path(
            outputs.get("opcode_text_corpus_parquet") or "data/features/main_benchmark/opcode_text_corpus.parquet"
        ),
    )


def _assert_unique_contract_ids(frame: pd.DataFrame, name: str) -> None:
    if CONTRACT_ID_COLUMN not in frame.columns:
        raise ValueError(f"`{name}` missing required column `{CONTRACT_ID_COLUMN}`.")
    duplicated = frame[CONTRACT_ID_COLUMN].duplicated()
    if bool(duplicated.any()):
        dup_ids = sorted(frame.loc[duplicated, CONTRACT_ID_COLUMN].astype(str).tolist())
        raise ValueError(f"`{name}` has duplicate contract IDs: {dup_ids[:10]}")


class Phase3Dataset:
    def __init__(self, config_path: Path = DEFAULT_PHASE3_CONFIG_PATH, split: Optional[str] = None):
        self.config_path = config_path.resolve()
        self.config = _load_phase3_dataset_config(self.config_path)

        required_paths = [
            self.config.feature_index_path,
            self.config.codebert_features_path,
            self.config.graph_features_path,
            self.config.opcode_text_path,
        ]
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing Phase 3 dataset artifact(s): {missing}")

        feature_index = pd.read_parquet(self.config.feature_index_path).copy()
        codebert = pd.read_parquet(self.config.codebert_features_path).copy()
        graph_features = pd.read_parquet(self.config.graph_features_path).copy()
        opcode_text = pd.read_parquet(self.config.opcode_text_path).copy()

        for frame in [feature_index, codebert, graph_features, opcode_text]:
            frame[CONTRACT_ID_COLUMN] = frame[CONTRACT_ID_COLUMN].fillna("").astype(str).str.strip()
            frame.drop(frame[frame[CONTRACT_ID_COLUMN] == ""].index, inplace=True)

        _assert_unique_contract_ids(feature_index, "feature_index")
        _assert_unique_contract_ids(codebert, "codebert_features")
        _assert_unique_contract_ids(graph_features, "graph_level_features")
        _assert_unique_contract_ids(opcode_text, "opcode_text_corpus")

        merged = feature_index.merge(codebert, on=[CONTRACT_ID_COLUMN, "split"], how="left", validate="one_to_one")
        merged = merged.merge(graph_features, on=[CONTRACT_ID_COLUMN, "split"], how="left", validate="one_to_one")
        merged = merged.merge(
            opcode_text[[CONTRACT_ID_COLUMN, "split", "opcode_text", "opcode_token_count"]],
            on=[CONTRACT_ID_COLUMN, "split"],
            how="left",
            validate="one_to_one",
        )

        if split is not None:
            split_value = str(split).strip()
            if split_value not in {"train", "val", "test"}:
                raise ValueError("`split` must be one of: train, val, test.")
            merged = merged[merged["split"] == split_value].copy()

        merged = merged.sort_values(["split", CONTRACT_ID_COLUMN]).reset_index(drop=True)
        self.frame = merged
        self.target_columns = [f"swc_{swc}" for swc in self.config.swc_ids]
        self.target_mask_columns = [f"swc_{swc}_assessed" for swc in self.config.swc_ids]
        self.codebert_columns = sorted([column for column in merged.columns if column.startswith("cb_")])
        self.graph_feature_columns = sorted([column for column in merged.columns if column.startswith("gf_")])

        missing_targets = [column for column in self.target_columns + self.target_mask_columns if column not in merged.columns]
        if missing_targets:
            raise ValueError(f"Phase 3 feature index is missing target columns: {missing_targets}")

    def __len__(self) -> int:
        return int(len(self.frame))

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.frame.iloc[int(index)]
        targets = row[self.target_columns].fillna(-1).astype(int).to_numpy(dtype=np.int64)
        target_mask = row[self.target_mask_columns].fillna(False).astype(bool).to_numpy(dtype=bool)
        codebert_features = row[self.codebert_columns].fillna(0.0).astype(float).to_numpy(dtype=np.float32)
        graph_features = row[self.graph_feature_columns].fillna(0.0).astype(float).to_numpy(dtype=np.float32)

        return {
            "contract_id": str(row[CONTRACT_ID_COLUMN]),
            "split": str(row["split"]),
            "targets": targets,
            "target_mask": target_mask,
            "codebert_features": codebert_features,
            "graph_features": graph_features,
            "graph_unavailable": bool(row.get("graph_unavailable", False)),
            "unavailable_cause": str(row.get("unavailable_cause", "")),
        }

    def to_dataframe(self) -> pd.DataFrame:
        return self.frame.copy()

    def feature_shapes(self) -> Mapping[str, int]:
        return {
            "codebert_dim": int(len(self.codebert_columns)),
            "graph_feature_dim": int(len(self.graph_feature_columns)),
            "target_dim": int(len(self.target_columns)),
        }
