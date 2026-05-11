import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHASE4_CONFIG_PATH = PROJECT_ROOT / "configs/phase4.yaml"
DEFAULT_PHASE3_RUN_MANIFEST = PROJECT_ROOT / "reports/phase3/phase3_run_manifest.json"
CONTRACT_ID_COLUMN = "fp_runtime_unified"
REQUIRED_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Phase4DatasetConfig:
    swc_ids: List[int]
    phase3_run_manifest_path: Path
    phase2_graph_builder_run_manifest_path: Path
    feature_index_path: Path
    tfidf_features_path: Path
    pattern_features_path: Path
    graph_features_path: Path
    split_root: Path
    graph_artifacts_dir: Optional[Path] = None  # for GNN mode


def _resolve_path(path_value: Any) -> Path:
    raw = Path(str(path_value))
    if raw.is_absolute():
        return raw
    return (PROJECT_ROOT / raw).resolve()


def _safe_read_mapping(path: Path, context: str) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {context}: {path}")
    with path.open("r", encoding="utf-8") as fp:
        if path.suffix in {".yaml", ".yml"}:
            payload = yaml.safe_load(fp)
        else:
            payload = json.load(fp)
    if not isinstance(payload, dict):
        raise ValueError(f"{context} must be a mapping: {path}")
    return payload


def _normalize_swc_ids(raw_values: Sequence[Any]) -> List[int]:
    if not raw_values:
        raise ValueError("`main_benchmark.swc_ids` must be a non-empty list.")
    values: List[int] = []
    seen = set()
    for raw in raw_values:
        swc = int(raw)
        if swc not in seen:
            values.append(swc)
            seen.add(swc)
    return values


def _load_dataset_config(config_path: Path) -> Phase4DatasetConfig:
    raw = _safe_read_mapping(config_path, "Phase 4 config")
    benchmark = raw.get("main_benchmark", {})
    if benchmark is None:
        benchmark = {}
    if not isinstance(benchmark, dict):
        raise ValueError("`main_benchmark` must be a mapping when provided.")
    swc_ids = _normalize_swc_ids(benchmark.get("swc_ids", []))

    inputs = raw.get("inputs", {})
    if inputs is None:
        inputs = {}
    if not isinstance(inputs, dict):
        raise ValueError("`inputs` must be a mapping when provided.")

    phase3_manifest_path = _resolve_path(inputs.get("phase3_run_manifest_json") or str(DEFAULT_PHASE3_RUN_MANIFEST))
    phase3_manifest = _safe_read_mapping(phase3_manifest_path, "Phase 3 run manifest")

    phase3_inputs = phase3_manifest.get("inputs", {})
    phase3_outputs = phase3_manifest.get("outputs", {})
    if not isinstance(phase3_inputs, dict) or not isinstance(phase3_outputs, dict):
        raise ValueError("Phase 3 run manifest must contain mapping `inputs` and mapping `outputs`.")

    phase2_manifest_path = _resolve_path(
        inputs.get("phase2_graph_builder_run_manifest_json")
        or phase3_inputs.get("phase2_graph_builder_run_manifest_json")
        or "reports/phase2/graph_builder_run_manifest.json"
    )

    feature_index_path = _resolve_path(
        inputs.get("feature_index_parquet")
        or phase3_outputs.get("feature_index_parquet")
        or "data/features/main_benchmark/phase3_feature_index.parquet"
    )
    tfidf_features_path = _resolve_path(
        inputs.get("tfidf_features_parquet")
        or phase3_outputs.get("tfidf_features_parquet")
        or "data/features/main_benchmark/tfidf_features.parquet"
    )
    pattern_features_path = _resolve_path(
        inputs.get("pattern_features_parquet")
        or phase3_outputs.get("pattern_features_parquet")
        or "data/features/main_benchmark/pattern_features.parquet"
    )
    graph_features_path = _resolve_path(
        inputs.get("graph_level_features_parquet")
        or phase3_outputs.get("graph_level_features_parquet")
        or "data/features/main_benchmark/graph_level_features.parquet"
    )
    split_root = _resolve_path(
        inputs.get("split_root") or phase3_inputs.get("split_root") or "data/splits/main_benchmark"
    )

    graph_artifacts_dir_raw = inputs.get("graph_artifacts_dir")
    if graph_artifacts_dir_raw:
        graph_artifacts_dir: Optional[Path] = _resolve_path(graph_artifacts_dir_raw)
    else:
        # Auto-discover from phase2 manifest
        phase2_manifest = _safe_read_mapping(phase2_manifest_path, "Phase 2 graph builder manifest")
        phase2_outputs = phase2_manifest.get("outputs", {})
        if isinstance(phase2_outputs, dict) and "graph_dir" in phase2_outputs:
            graph_artifacts_dir = _resolve_path(phase2_outputs["graph_dir"])
        else:
            # Fallback: look for the standard location
            candidate = PROJECT_ROOT / "data" / "curated" / "graphs"
            if candidate.exists():
                subdirs = [d for d in candidate.iterdir() if d.is_dir()]
                if subdirs:
                    graph_artifacts_dir = sorted(subdirs)[-1]  # latest
                else:
                    graph_artifacts_dir = candidate
            else:
                graph_artifacts_dir = None

    return Phase4DatasetConfig(
        swc_ids=swc_ids,
        phase3_run_manifest_path=phase3_manifest_path,
        phase2_graph_builder_run_manifest_path=phase2_manifest_path,
        feature_index_path=feature_index_path,
        tfidf_features_path=tfidf_features_path,
        pattern_features_path=pattern_features_path,
        graph_features_path=graph_features_path,
        split_root=split_root,
        graph_artifacts_dir=graph_artifacts_dir,
    )


def _assert_unique_contract_ids(frame: pd.DataFrame, name: str) -> None:
    if CONTRACT_ID_COLUMN not in frame.columns:
        raise ValueError(f"`{name}` missing required column `{CONTRACT_ID_COLUMN}`.")
    duplicated = frame[CONTRACT_ID_COLUMN].duplicated()
    if bool(duplicated.any()):
        duplicates = sorted(frame.loc[duplicated, CONTRACT_ID_COLUMN].astype(str).tolist())
        raise ValueError(f"`{name}` has duplicate contract IDs: {duplicates[:10]}")


def _load_split_map(split_root: Path) -> Dict[str, str]:
    split_map: Dict[str, str] = {}
    for split in REQUIRED_SPLITS:
        split_path = split_root / f"{split}.parquet"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split parquet: {split_path}")
        frame = pd.read_parquet(split_path).copy()
        if CONTRACT_ID_COLUMN not in frame.columns:
            raise ValueError(f"Split parquet missing `{CONTRACT_ID_COLUMN}`: {split_path}")
        frame[CONTRACT_ID_COLUMN] = frame[CONTRACT_ID_COLUMN].fillna("").astype(str).str.strip()
        frame = frame[frame[CONTRACT_ID_COLUMN] != ""]
        if bool(frame[CONTRACT_ID_COLUMN].duplicated().any()):
            duplicates = sorted(frame.loc[frame[CONTRACT_ID_COLUMN].duplicated(), CONTRACT_ID_COLUMN].tolist())
            raise ValueError(f"Duplicate contract IDs in split `{split}`: {duplicates[:10]}")
        for contract_id in frame[CONTRACT_ID_COLUMN].tolist():
            if contract_id in split_map:
                raise ValueError(f"Split overlap detected for contract `{contract_id}`.")
            split_map[contract_id] = split
    return split_map


def _validate_split_alignment(feature_index: pd.DataFrame, split_map: Mapping[str, str]) -> None:
    feature_ids = set(feature_index[CONTRACT_ID_COLUMN].astype(str).tolist())
    split_ids = set(split_map.keys())
    if feature_ids != split_ids:
        only_features = sorted(feature_ids - split_ids)
        only_splits = sorted(split_ids - feature_ids)
        raise ValueError(
            "Split alignment mismatch between feature index and split files. "
            f"only_features={only_features[:10]}, only_splits={only_splits[:10]}"
        )

    mismatches = []
    for _, row in feature_index[[CONTRACT_ID_COLUMN, "split"]].iterrows():
        contract_id = str(row[CONTRACT_ID_COLUMN]).strip()
        observed = str(row["split"]).strip()
        expected = split_map.get(contract_id, "")
        if observed != expected:
            mismatches.append((contract_id, observed, expected))
            if len(mismatches) >= 10:
                break
    if mismatches:
        raise ValueError(f"Feature index split mismatch against split files: {mismatches}")


def _validate_label_order(feature_index: pd.DataFrame, swc_ids: Sequence[int]) -> None:
    expected_labels = [f"swc_{swc_id}" for swc_id in swc_ids]
    expected_masks = [f"swc_{swc_id}_assessed" for swc_id in swc_ids]

    missing = [column for column in expected_labels + expected_masks if column not in feature_index.columns]
    if missing:
        raise ValueError(f"Feature index missing expected label columns: {missing}")

    observed_labels = [column for column in feature_index.columns if column in set(expected_labels)]
    observed_masks = [column for column in feature_index.columns if column in set(expected_masks)]
    if observed_labels != expected_labels:
        raise ValueError(
            "SWC label order mismatch in feature index. "
            f"expected={expected_labels}, observed={observed_labels}"
        )
    if observed_masks != expected_masks:
        raise ValueError(
            "SWC assessed order mismatch in feature index. "
            f"expected={expected_masks}, observed={observed_masks}"
        )


class Phase4Dataset(Dataset):
    REQUIRED_METADATA_COLUMNS = [
        "source_group",
        "has_cgt",
        "has_dappscan",
        "is_proxy_like",
        "is_stub_like",
        "graph_unavailable",
        "unavailable_cause",
    ]

    def __init__(
        self,
        config_path: Path = DEFAULT_PHASE4_CONFIG_PATH,
        split: Optional[str] = None,
        feature_index_path_override: Optional[Path] = None,
    ):
        self.config_path = config_path.resolve()
        self.config = _load_dataset_config(self.config_path)

        # Allow callers to override the feature index parquet (e.g. a versioned
        # 10-SWC variant that excludes SWC-132 at the data level).
        if feature_index_path_override is not None:
            from dataclasses import replace as _dc_replace
            override_path = Path(feature_index_path_override).resolve()
            # Derive swc_ids from the override parquet's actual columns so the
            # label-order validator stays consistent with the new file.
            _tmp = pd.read_parquet(override_path, columns=None)
            _override_swc_ids = [
                int(c.split("_")[1])
                for c in _tmp.columns
                if c.startswith("swc_") and not c.endswith("_assessed")
                and c.split("_")[1].isdigit()
            ]
            self.config = _dc_replace(
                self.config,
                feature_index_path=override_path,
                swc_ids=_override_swc_ids,
            )

        required_paths = [
            self.config.phase3_run_manifest_path,
            self.config.phase2_graph_builder_run_manifest_path,
            self.config.feature_index_path,
            self.config.tfidf_features_path,
            self.config.pattern_features_path,
            self.config.graph_features_path,
            self.config.split_root / "train.parquet",
            self.config.split_root / "val.parquet",
            self.config.split_root / "test.parquet",
        ]
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing Phase 4 dataset artifact(s): {missing}")

        feature_index = pd.read_parquet(self.config.feature_index_path).copy()
        tfidf_features = pd.read_parquet(self.config.tfidf_features_path).copy()
        pattern_features = pd.read_parquet(self.config.pattern_features_path).copy()
        graph_features = pd.read_parquet(self.config.graph_features_path).copy()

        for frame in [feature_index, tfidf_features, pattern_features, graph_features]:
            frame[CONTRACT_ID_COLUMN] = frame[CONTRACT_ID_COLUMN].fillna("").astype(str).str.strip()
            frame.drop(frame[frame[CONTRACT_ID_COLUMN] == ""].index, inplace=True)

        _assert_unique_contract_ids(feature_index, "feature_index")
        _assert_unique_contract_ids(tfidf_features, "tfidf_features")
        _assert_unique_contract_ids(pattern_features, "pattern_features")
        _assert_unique_contract_ids(graph_features, "graph_features")

        _validate_label_order(feature_index, self.config.swc_ids)
        split_map = _load_split_map(self.config.split_root)
        _validate_split_alignment(feature_index, split_map)

        missing_metadata = [
            column for column in self.REQUIRED_METADATA_COLUMNS if column not in feature_index.columns
        ]
        if missing_metadata:
            raise ValueError(f"Feature index missing required metadata columns: {missing_metadata}")

        self.target_columns = [f"swc_{swc_id}" for swc_id in self.config.swc_ids]
        self.target_mask_columns = [f"swc_{swc_id}_assessed" for swc_id in self.config.swc_ids]
        tfidf_feature_columns = sorted(
            [column for column in tfidf_features.columns if column.startswith("tfidf_")]
        )
        pattern_feature_columns = sorted(
            [column for column in pattern_features.columns if column.startswith("pat_")]
        )
        graph_feature_columns = sorted(
            [column for column in graph_features.columns if column.startswith("gf_")]
        )
        if not tfidf_feature_columns:
            raise ValueError("No TF-IDF feature columns (prefix `tfidf_`) were found.")
        if not pattern_feature_columns:
            raise ValueError("No pattern feature columns (prefix `pat_`) were found.")
        if not graph_feature_columns:
            raise ValueError("No graph feature columns (prefix `gf_`) were found.")

        merged = feature_index.merge(
            tfidf_features[[CONTRACT_ID_COLUMN, "split"] + tfidf_feature_columns],
            on=[CONTRACT_ID_COLUMN, "split"],
            how="left",
            validate="one_to_one",
        )
        merged = merged.merge(
            pattern_features[[CONTRACT_ID_COLUMN, "split"] + pattern_feature_columns],
            on=[CONTRACT_ID_COLUMN, "split"],
            how="left",
            validate="one_to_one",
        )
        merged = merged.merge(
            graph_features[[CONTRACT_ID_COLUMN, "split"] + graph_feature_columns],
            on=[CONTRACT_ID_COLUMN, "split"],
            how="left",
            validate="one_to_one",
        )
        self.opcode_feature_columns = tfidf_feature_columns + pattern_feature_columns
        self.graph_feature_columns = graph_feature_columns

        if split is not None:
            split_value = str(split).strip()
            if split_value not in set(REQUIRED_SPLITS):
                raise ValueError(f"`split` must be one of {REQUIRED_SPLITS}.")
            merged = merged[merged["split"] == split_value].copy()

        self.frame = merged.sort_values(["split", CONTRACT_ID_COLUMN]).reset_index(drop=True)
        self._split_counts = {
            str(name): int(count) for name, count in feature_index["split"].value_counts().to_dict().items()
        }

        # Prepare graph artifact lookup for GNN mode
        self._graph_artifact_dir = self.config.graph_artifacts_dir
        self._graph_file_cache: Dict[str, Optional[Path]] = {}
        if self._graph_artifact_dir and self._graph_artifact_dir.exists():
            for pt_file in self._graph_artifact_dir.glob("*.pt"):
                contract_id = pt_file.stem
                self._graph_file_cache[contract_id] = pt_file

    @property
    def has_graph_artifacts(self) -> bool:
        """Whether actual graph .pt files are available for GNN loading."""
        return bool(self._graph_file_cache)

    def __len__(self) -> int:
        return int(len(self.frame))

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.frame.iloc[int(index)]

        opcode_features = row[self.opcode_feature_columns].fillna(0.0).astype(float).to_numpy(dtype=np.float32)
        graph_features = row[self.graph_feature_columns].fillna(0.0).astype(float).to_numpy(dtype=np.float32)
        targets = row[self.target_columns].fillna(0.0).astype(float).to_numpy(dtype=np.float32)
        target_mask = row[self.target_mask_columns].fillna(False).astype(bool).to_numpy(dtype=bool)

        metadata = {
            "source_group": str(row["source_group"]),
            "has_cgt": bool(row["has_cgt"]),
            "has_dappscan": bool(row["has_dappscan"]),
            "is_proxy_like": bool(row["is_proxy_like"]),
            "is_stub_like": bool(row["is_stub_like"]),
            "graph_unavailable": bool(row["graph_unavailable"]),
            "unavailable_cause": str(row["unavailable_cause"]),
        }

        result = {
            "contract_id": str(row[CONTRACT_ID_COLUMN]),
            "split": str(row["split"]),
            "opcode_features": torch.tensor(opcode_features, dtype=torch.float32),
            "graph_features": torch.tensor(graph_features, dtype=torch.float32),
            "targets": torch.tensor(targets, dtype=torch.float32),
            "target_mask": torch.tensor(target_mask, dtype=torch.bool),
            "metadata": metadata,
        }

        # Load actual graph structure for GNN if available
        contract_id = str(row[CONTRACT_ID_COLUMN])
        graph_path = self._graph_file_cache.get(contract_id)
        if graph_path is not None and graph_path.exists():
            try:
                graph_data = torch.load(graph_path, map_location="cpu", weights_only=False)
                pyg = graph_data.get("pyg", {})
                result["graph_x"] = pyg.get("x", torch.zeros(1, 3, dtype=torch.long))
                result["graph_edge_index"] = pyg.get("edge_index", torch.zeros(2, 0, dtype=torch.long))
                # Load edge type tensor for heterogeneous message passing
                result["graph_edge_type"] = pyg.get(
                    "edge_type", torch.zeros(result["graph_edge_index"].size(1), dtype=torch.long)
                )
                result["has_graph_structure"] = True
            except Exception:
                result["graph_x"] = torch.zeros(1, 3, dtype=torch.long)
                result["graph_edge_index"] = torch.zeros(2, 0, dtype=torch.long)
                result["graph_edge_type"] = torch.zeros(0, dtype=torch.long)
                result["has_graph_structure"] = False
        else:
            result["graph_x"] = torch.zeros(1, 3, dtype=torch.long)
            result["graph_edge_index"] = torch.zeros(2, 0, dtype=torch.long)
            result["graph_edge_type"] = torch.zeros(0, dtype=torch.long)
            result["has_graph_structure"] = False

        return result

    def split_counts(self) -> Mapping[str, int]:
        return dict(self._split_counts)

    def feature_shapes(self) -> Mapping[str, int]:
        return {
            "opcode_dim": int(len(self.opcode_feature_columns)),
            "graph_dim": int(len(self.graph_feature_columns)),
            "target_dim": int(len(self.target_columns)),
        }

    def preservation_checks(self) -> Mapping[str, bool]:
        metadata_columns = set(self.REQUIRED_METADATA_COLUMNS)
        present_columns = set(self.frame.columns)
        return {
            "split_alignment_preserved": True,
            "swc_order_preserved": True,
            "provenance_columns_preserved": bool({"source_group", "has_cgt", "has_dappscan"}.issubset(present_columns)),
            "proxy_stub_columns_preserved": bool({"is_proxy_like", "is_stub_like"}.issubset(present_columns)),
            "required_metadata_preserved": bool(metadata_columns.issubset(present_columns)),
        }


def phase4_collate_fn(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not batch:
        raise ValueError("Empty batch passed to `phase4_collate_fn`.")

    collated = {
        "contract_id": [str(item["contract_id"]) for item in batch],
        "split": [str(item["split"]) for item in batch],
        "opcode_features": torch.stack([item["opcode_features"] for item in batch], dim=0),
        "graph_features": torch.stack([item["graph_features"] for item in batch], dim=0),
        "targets": torch.stack([item["targets"] for item in batch], dim=0),
        "target_mask": torch.stack([item["target_mask"] for item in batch], dim=0),
        "metadata": [dict(item["metadata"]) for item in batch],
    }

    # Batch graph structures for GNN (manual PyG-style batching)
    if "graph_x" in batch[0]:
        all_x = []
        all_edge_index = []
        all_edge_type = []
        batch_vec = []
        node_offset = 0
        for i, item in enumerate(batch):
            x = item["graph_x"]
            ei = item["graph_edge_index"]
            num_nodes = x.size(0)
            all_x.append(x)
            all_edge_index.append(ei + node_offset)
            # Edge types don't need offset — they are categorical labels
            if "graph_edge_type" in item:
                all_edge_type.append(item["graph_edge_type"])
            batch_vec.extend([i] * num_nodes)
            node_offset += num_nodes

        collated["graph_x"] = torch.cat(all_x, dim=0)
        collated["graph_edge_index"] = torch.cat(all_edge_index, dim=1)
        collated["graph_batch"] = torch.tensor(batch_vec, dtype=torch.long)
        if all_edge_type:
            collated["graph_edge_type"] = torch.cat(all_edge_type, dim=0)
        collated["has_graph_structure"] = [
            item.get("has_graph_structure", False) for item in batch
        ]

    return collated
