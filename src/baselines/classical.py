from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


@dataclass(frozen=True)
class ClassicalBaselineConfig:
    max_iter: int = 200
    C: float = 1.0
    solver: str = "liblinear"
    random_state: int = 42


class MaskedOneVsRestLogisticRegression:
    """Per-label logistic regression with assessed-label masking support."""

    def __init__(self, config: Optional[ClassicalBaselineConfig] = None):
        self.config = config or ClassicalBaselineConfig()
        self._feature_dim: Optional[int] = None
        self._num_labels: Optional[int] = None
        self._label_models: List[Optional[LogisticRegression]] = []
        self._label_constants: List[float] = []
        self._label_fit_info: List[Dict[str, Any]] = []

    @property
    def is_fitted(self) -> bool:
        return self._feature_dim is not None and self._num_labels is not None

    @property
    def feature_dim(self) -> int:
        if self._feature_dim is None:
            raise RuntimeError("Classical baseline is not fitted yet.")
        return int(self._feature_dim)

    @property
    def num_labels(self) -> int:
        if self._num_labels is None:
            raise RuntimeError("Classical baseline is not fitted yet.")
        return int(self._num_labels)

    @property
    def fit_info(self) -> List[Dict[str, Any]]:
        return [dict(entry) for entry in self._label_fit_info]

    @staticmethod
    def _validate_inputs(features: np.ndarray, targets: np.ndarray, target_mask: np.ndarray) -> None:
        if features.ndim != 2:
            raise ValueError(f"`features` must be rank-2. Got shape {tuple(features.shape)}.")
        if targets.ndim != 2:
            raise ValueError(f"`targets` must be rank-2. Got shape {tuple(targets.shape)}.")
        if target_mask.ndim != 2:
            raise ValueError(f"`target_mask` must be rank-2. Got shape {tuple(target_mask.shape)}.")
        if features.shape[0] != targets.shape[0] or targets.shape != target_mask.shape:
            raise ValueError(
                "Shape mismatch. "
                f"features={tuple(features.shape)}, targets={tuple(targets.shape)}, mask={tuple(target_mask.shape)}"
            )
        if features.shape[0] == 0:
            raise ValueError("Cannot fit with zero rows.")
        assessed_values = np.unique(targets[target_mask])
        invalid_values = [value for value in assessed_values.tolist() if value not in {0, 1}]
        if invalid_values:
            raise ValueError(f"Assessed targets must be binary 0/1. Found: {invalid_values}")

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        target_mask: np.ndarray,
    ) -> "MaskedOneVsRestLogisticRegression":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.int64)
        mask = np.asarray(target_mask, dtype=bool)
        self._validate_inputs(x, y, mask)

        self._feature_dim = int(x.shape[1])
        self._num_labels = int(y.shape[1])
        self._label_models = []
        self._label_constants = []
        self._label_fit_info = []

        for label_idx in range(self._num_labels):
            label_mask = mask[:, label_idx]
            assessed_count = int(label_mask.sum())

            if assessed_count == 0:
                self._label_models.append(None)
                self._label_constants.append(0.0)
                self._label_fit_info.append(
                    {
                        "label_index": label_idx,
                        "strategy": "constant",
                        "constant_value": 0.0,
                        "assessed_count": assessed_count,
                        "reason": "no_assessed_labels",
                    }
                )
                continue

            x_label = x[label_mask]
            y_label = y[label_mask, label_idx]
            unique_values = np.unique(y_label)

            if unique_values.size == 1:
                constant_value = float(unique_values[0])
                self._label_models.append(None)
                self._label_constants.append(constant_value)
                self._label_fit_info.append(
                    {
                        "label_index": label_idx,
                        "strategy": "constant",
                        "constant_value": constant_value,
                        "assessed_count": assessed_count,
                        "reason": "single_class_in_assessed_rows",
                    }
                )
                continue

            model = LogisticRegression(
                max_iter=int(self.config.max_iter),
                C=float(self.config.C),
                solver=str(self.config.solver),
                random_state=int(self.config.random_state),
                class_weight="balanced",
            )
            model.fit(x_label, y_label)
            self._label_models.append(model)
            self._label_constants.append(0.0)
            self._label_fit_info.append(
                {
                    "label_index": label_idx,
                    "strategy": "logistic_regression",
                    "constant_value": None,
                    "assessed_count": assessed_count,
                    "reason": "",
                }
            )

        return self

    def _ensure_ready(self, features: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Classical baseline must be fitted before prediction.")
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"`features` must be rank-2. Got shape {tuple(x.shape)}.")
        if x.shape[1] != self.feature_dim:
            raise ValueError(f"Feature dimension mismatch: expected {self.feature_dim}, got {x.shape[1]}.")
        return x

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        x = self._ensure_ready(features)
        proba = np.zeros((x.shape[0], self.num_labels), dtype=np.float64)

        for label_idx in range(self.num_labels):
            model = self._label_models[label_idx]
            if model is None:
                proba[:, label_idx] = float(self._label_constants[label_idx])
            else:
                proba[:, label_idx] = model.predict_proba(x)[:, 1]

        return np.clip(proba, 0.0, 1.0)

    def predict_logits(self, features: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        if eps <= 0.0:
            raise ValueError("`eps` must be positive.")
        proba = self.predict_proba(features)
        clipped = np.clip(proba, eps, 1.0 - eps)
        return np.log(clipped / (1.0 - clipped))

    def predict_binary(self, features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        if threshold <= 0.0 or threshold >= 1.0:
            raise ValueError("`threshold` must be in (0.0, 1.0).")
        return (self.predict_proba(features) >= threshold).astype(np.int64)

    def parameter_count(self) -> int:
        if not self.is_fitted:
            raise RuntimeError("Classical baseline must be fitted before counting parameters.")
        total = 0
        for model in self._label_models:
            if model is None:
                continue
            total += int(model.coef_.size)
            total += int(model.intercept_.size)
        return int(total)

    def save(self, path: Path) -> None:
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted classical baseline.")
        resolved = path.resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "max_iter": int(self.config.max_iter),
                "C": float(self.config.C),
                "solver": str(self.config.solver),
                "random_state": int(self.config.random_state),
            },
            "feature_dim": int(self.feature_dim),
            "num_labels": int(self.num_labels),
            "label_models": self._label_models,
            "label_constants": list(self._label_constants),
            "label_fit_info": [dict(entry) for entry in self._label_fit_info],
        }
        with resolved.open("wb") as fp:
            pickle.dump(payload, fp)

    @classmethod
    def load(cls, path: Path) -> "MaskedOneVsRestLogisticRegression":
        resolved = path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Missing classical baseline checkpoint: {resolved}")
        with resolved.open("rb") as fp:
            payload = pickle.load(fp)

        if not isinstance(payload, dict):
            raise ValueError(f"Invalid classical baseline checkpoint payload: {resolved}")

        raw_config = payload.get("config", {})
        if not isinstance(raw_config, dict):
            raise ValueError("Classical baseline checkpoint missing `config` mapping.")

        model = cls(
            ClassicalBaselineConfig(
                max_iter=int(raw_config.get("max_iter", 200)),
                C=float(raw_config.get("C", 1.0)),
                solver=str(raw_config.get("solver", "liblinear")),
                random_state=int(raw_config.get("random_state", 42)),
            )
        )
        model._feature_dim = int(payload.get("feature_dim"))
        model._num_labels = int(payload.get("num_labels"))
        model._label_models = list(payload.get("label_models", []))
        model._label_constants = [float(value) for value in payload.get("label_constants", [])]
        model._label_fit_info = [dict(entry) for entry in payload.get("label_fit_info", [])]

        if len(model._label_models) != model._num_labels:
            raise ValueError("Checkpoint label model count does not match `num_labels`.")
        if len(model._label_constants) != model._num_labels:
            raise ValueError("Checkpoint label constant count does not match `num_labels`.")
        if len(model._label_fit_info) != model._num_labels:
            raise ValueError("Checkpoint fit info count does not match `num_labels`.")

        return model


class _MaskedOneVsRestBase:
    """Base class for OneVsRest sklearn-based classifiers with assessed-label masking."""

    _MODEL_TYPE = "base"

    def __init__(self, random_state: int = 42, **kwargs: Any):
        self.random_state = random_state
        self._extra_params = kwargs
        self._feature_dim: Optional[int] = None
        self._num_labels: Optional[int] = None
        self._label_models: List[Optional[Any]] = []
        self._label_constants: List[float] = []
        self._label_fit_info: List[Dict[str, Any]] = []

    def _create_estimator(self, random_state: int) -> Any:
        raise NotImplementedError

    @property
    def is_fitted(self) -> bool:
        return self._feature_dim is not None and self._num_labels is not None

    @property
    def feature_dim(self) -> int:
        if self._feature_dim is None:
            raise RuntimeError("Model is not fitted yet.")
        return int(self._feature_dim)

    @property
    def num_labels(self) -> int:
        if self._num_labels is None:
            raise RuntimeError("Model is not fitted yet.")
        return int(self._num_labels)

    @property
    def fit_info(self) -> List[Dict[str, Any]]:
        return [dict(entry) for entry in self._label_fit_info]

    @staticmethod
    def _validate_inputs(features: np.ndarray, targets: np.ndarray, target_mask: np.ndarray) -> None:
        if features.ndim != 2:
            raise ValueError(f"`features` must be rank-2. Got shape {tuple(features.shape)}.")
        if targets.ndim != 2:
            raise ValueError(f"`targets` must be rank-2. Got shape {tuple(targets.shape)}.")
        if target_mask.ndim != 2:
            raise ValueError(f"`target_mask` must be rank-2. Got shape {tuple(target_mask.shape)}.")
        if features.shape[0] != targets.shape[0] or targets.shape != target_mask.shape:
            raise ValueError(
                f"Shape mismatch. features={tuple(features.shape)}, "
                f"targets={tuple(targets.shape)}, mask={tuple(target_mask.shape)}"
            )
        if features.shape[0] == 0:
            raise ValueError("Cannot fit with zero rows.")

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        target_mask: np.ndarray,
    ) -> "_MaskedOneVsRestBase":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.int64)
        mask = np.asarray(target_mask, dtype=bool)
        self._validate_inputs(x, y, mask)

        self._feature_dim = int(x.shape[1])
        self._num_labels = int(y.shape[1])
        self._label_models = []
        self._label_constants = []
        self._label_fit_info = []

        for label_idx in range(self._num_labels):
            label_mask = mask[:, label_idx]
            assessed_count = int(label_mask.sum())

            if assessed_count == 0:
                self._label_models.append(None)
                self._label_constants.append(0.0)
                self._label_fit_info.append({
                    "label_index": label_idx, "strategy": "constant",
                    "constant_value": 0.0, "assessed_count": 0,
                    "reason": "no_assessed_labels",
                })
                continue

            x_label = x[label_mask]
            y_label = y[label_mask, label_idx]
            unique_values = np.unique(y_label)

            if unique_values.size == 1:
                constant_value = float(unique_values[0])
                self._label_models.append(None)
                self._label_constants.append(constant_value)
                self._label_fit_info.append({
                    "label_index": label_idx, "strategy": "constant",
                    "constant_value": constant_value, "assessed_count": assessed_count,
                    "reason": "single_class_in_assessed_rows",
                })
                continue

            model = self._create_estimator(random_state=self.random_state)
            model.fit(x_label, y_label)
            self._label_models.append(model)
            self._label_constants.append(0.0)
            self._label_fit_info.append({
                "label_index": label_idx, "strategy": self._MODEL_TYPE,
                "constant_value": None, "assessed_count": assessed_count,
                "reason": "",
            })

        return self

    def _ensure_ready(self, features: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"`features` must be rank-2. Got shape {tuple(x.shape)}.")
        if x.shape[1] != self.feature_dim:
            raise ValueError(f"Feature dimension mismatch: expected {self.feature_dim}, got {x.shape[1]}.")
        return x

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        x = self._ensure_ready(features)
        proba = np.zeros((x.shape[0], self.num_labels), dtype=np.float64)
        for label_idx in range(self.num_labels):
            model = self._label_models[label_idx]
            if model is None:
                proba[:, label_idx] = float(self._label_constants[label_idx])
            else:
                proba[:, label_idx] = model.predict_proba(x)[:, 1]
        return np.clip(proba, 0.0, 1.0)

    def predict_logits(self, features: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        proba = self.predict_proba(features)
        clipped = np.clip(proba, eps, 1.0 - eps)
        return np.log(clipped / (1.0 - clipped))

    def predict_binary(self, features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(features) >= threshold).astype(np.int64)

    def parameter_count(self) -> int:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before counting parameters.")
        total = 0
        for model in self._label_models:
            if model is None:
                continue
            if hasattr(model, "coef_"):
                total += int(model.coef_.size) + int(model.intercept_.size)
            elif hasattr(model, "get_booster"):
                total += len(model.get_booster().get_dump())
            elif hasattr(model, "estimators_"):
                total += sum(tree.tree_.node_count for tree in model.estimators_)
        return int(total)

    def save(self, path: Path) -> None:
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")
        resolved = path.resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": self._MODEL_TYPE,
            "random_state": self.random_state,
            "extra_params": self._extra_params,
            "feature_dim": self.feature_dim,
            "num_labels": self.num_labels,
            "label_models": self._label_models,
            "label_constants": list(self._label_constants),
            "label_fit_info": [dict(e) for e in self._label_fit_info],
        }
        with resolved.open("wb") as fp:
            pickle.dump(payload, fp)

    @classmethod
    def load(cls, path: Path) -> "_MaskedOneVsRestBase":
        resolved = path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Missing checkpoint: {resolved}")
        with resolved.open("rb") as fp:
            payload = pickle.load(fp)
        model = cls(
            random_state=int(payload.get("random_state", 42)),
            **payload.get("extra_params", {}),
        )
        model._feature_dim = int(payload["feature_dim"])
        model._num_labels = int(payload["num_labels"])
        model._label_models = list(payload["label_models"])
        model._label_constants = [float(v) for v in payload["label_constants"]]
        model._label_fit_info = [dict(e) for e in payload["label_fit_info"]]
        return model


class MaskedOneVsRestXGBoost(_MaskedOneVsRestBase):
    """Per-label XGBoost with assessed-label masking."""

    _MODEL_TYPE = "xgboost"

    def __init__(
        self,
        random_state: int = 42,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        scale_pos_weight_auto: bool = True,
    ):
        super().__init__(
            random_state=random_state,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight_auto=scale_pos_weight_auto,
        )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.scale_pos_weight_auto = scale_pos_weight_auto

    def _create_estimator(self, random_state: int) -> Any:
        return XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        target_mask: np.ndarray,
    ) -> "MaskedOneVsRestXGBoost":
        """Fit with per-label scale_pos_weight for class imbalance."""
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.int64)
        mask = np.asarray(target_mask, dtype=bool)
        self._validate_inputs(x, y, mask)

        self._feature_dim = int(x.shape[1])
        self._num_labels = int(y.shape[1])
        self._label_models = []
        self._label_constants = []
        self._label_fit_info = []

        for label_idx in range(self._num_labels):
            label_mask = mask[:, label_idx]
            assessed_count = int(label_mask.sum())

            if assessed_count == 0:
                self._label_models.append(None)
                self._label_constants.append(0.0)
                self._label_fit_info.append({
                    "label_index": label_idx, "strategy": "constant",
                    "constant_value": 0.0, "assessed_count": 0,
                    "reason": "no_assessed_labels",
                })
                continue

            x_label = x[label_mask]
            y_label = y[label_mask, label_idx]
            unique_values = np.unique(y_label)

            if unique_values.size == 1:
                constant_value = float(unique_values[0])
                self._label_models.append(None)
                self._label_constants.append(constant_value)
                self._label_fit_info.append({
                    "label_index": label_idx, "strategy": "constant",
                    "constant_value": constant_value, "assessed_count": assessed_count,
                    "reason": "single_class_in_assessed_rows",
                })
                continue

            spw = 1.0
            if self.scale_pos_weight_auto:
                n_pos = int(y_label.sum())
                n_neg = int(assessed_count - n_pos)
                if n_pos > 0:
                    spw = float(n_neg) / float(n_pos)

            model = XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                scale_pos_weight=spw,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric="logloss",
                verbosity=0,
            )
            model.fit(x_label, y_label)
            self._label_models.append(model)
            self._label_constants.append(0.0)
            self._label_fit_info.append({
                "label_index": label_idx, "strategy": self._MODEL_TYPE,
                "constant_value": None, "assessed_count": assessed_count,
                "reason": "",
                "scale_pos_weight": spw,
            })

        return self


class MaskedOneVsRestRandomForest(_MaskedOneVsRestBase):
    """Per-label Random Forest with assessed-label masking."""

    _MODEL_TYPE = "random_forest"

    def __init__(
        self,
        random_state: int = 42,
        n_estimators: int = 300,
        max_depth: Optional[int] = None,
    ):
        super().__init__(
            random_state=random_state,
            n_estimators=n_estimators,
            max_depth=max_depth,
        )
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def _create_estimator(self, random_state: int) -> Any:
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )


class MaskedOneVsRestLightGBM(_MaskedOneVsRestBase):
    """Per-label LightGBM with assessed-label masking."""

    _MODEL_TYPE = "lightgbm"

    def __init__(
        self,
        random_state: int = 42,
        n_estimators: int = 300,
        num_leaves: int = 31,
        learning_rate: float = 0.1,
        is_unbalance: bool = True,
    ):
        super().__init__(
            random_state=random_state,
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            is_unbalance=is_unbalance,
        )
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.is_unbalance = is_unbalance

    def _create_estimator(self, random_state: int) -> Any:
        return LGBMClassifier(
            n_estimators=self.n_estimators,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            is_unbalance=self.is_unbalance,
            random_state=random_state,
            verbosity=-1,
            n_jobs=1,
        )
