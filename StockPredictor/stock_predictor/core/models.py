"""Model factory utilities providing configurable estimators."""

from __future__ import annotations

import logging
import inspect
from typing import Any, Dict, Sequence

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)

try:  # Optional dependencies
    from lightgbm import LGBMClassifier, LGBMRegressor  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    LGBMClassifier = LGBMRegressor = None  # type: ignore

try:  # Optional dependencies
    from xgboost import XGBClassifier, XGBRegressor  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = XGBRegressor = None  # type: ignore

try:  # scikit fallback for gradient boosting
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
except ImportError:  # pragma: no cover - available in supported sklearn versions
    HistGradientBoostingClassifier = HistGradientBoostingRegressor = None  # type: ignore


class ModelFactory:
    """Create estimators with sensible defaults for different targets."""

    DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
        "random_forest": {"n_estimators": 300, "random_state": 42, "n_jobs": -1},
        "lightgbm": {"n_estimators": 400, "learning_rate": 0.05},
        "xgboost": {"n_estimators": 400, "learning_rate": 0.05},
        "mlp": {"hidden_layer_sizes": (128, 64), "activation": "relu", "max_iter": 300},
        "hist_gb": {"max_depth": 8, "learning_rate": 0.05, "max_iter": 400},
        "logistic": {"C": 1.0, "max_iter": 500},
        "ensemble": {"interval_alpha": 0.2},
        "lstm": {
            "sequence_length": 20,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.1,
            "batch_size": 64,
            "epochs": 10,
            "lr": 1e-3,
        },
        "gru": {
            "sequence_length": 20,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.1,
            "batch_size": 64,
            "epochs": 10,
            "lr": 1e-3,
        },
        "transformer": {
            "sequence_length": 30,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.1,
            "batch_size": 64,
            "epochs": 10,
            "lr": 1e-3,
            "nhead": 4,
            "dim_feedforward": 256,
        },
    }

    def __init__(self, model_type: str, overrides: Dict[str, Any] | None = None) -> None:
        self.model_type = model_type.lower()
        self.overrides = overrides or {}

    def create(
        self,
        task: str,
        *,
        calibrate: bool = False,
        calibration_params: Dict[str, Any] | None = None,
    ) -> Pipeline:
        """Return a pipeline for the requested task (regression or classification)."""

        params = self.DEFAULT_PARAMS.get(self.model_type, {}).copy()
        params.update(self.overrides)

        calibrate = bool(params.pop("calibrate", False) or calibrate)
        calibration_params = calibration_params or params.pop("calibration_params", None) or {}
        if task == "regression":
            params.pop("class_weight", None)
        elif self.model_type == "mlp" and "class_weight" in params:
            LOGGER.warning(
                "class_weight not supported for MLPClassifier; removing for model type '%s'.",
                self.model_type,
            )
            params.pop("class_weight", None)

        def _instantiate(factory: Any, arguments: Dict[str, Any]) -> Any:
            try:
                return factory(**arguments)
            except TypeError as exc:
                if "class_weight" in arguments:
                    LOGGER.warning(
                        "Removing unsupported class_weight for model type '%s': %s",
                        self.model_type,
                        exc,
                    )
                    trimmed = dict(arguments)
                    trimmed.pop("class_weight", None)
                    return factory(**trimmed)
                raise

        estimator: Any
        needs_scaler = False

        if self.model_type == "random_forest":
            estimator = (
                _instantiate(RandomForestRegressor, params)
                if task == "regression"
                else _instantiate(RandomForestClassifier, params)
            )
        elif self.model_type == "lightgbm" and LGBMRegressor is not None:
            estimator = (
                _instantiate(LGBMRegressor, params)
                if task == "regression"
                else _instantiate(LGBMClassifier, params)
            )
        elif self.model_type == "xgboost" and XGBRegressor is not None:
            estimator = (
                _instantiate(XGBRegressor, params)
                if task == "regression"
                else _instantiate(XGBClassifier, params)
            )
        elif self.model_type == "hist_gb" and HistGradientBoostingRegressor is not None:
            estimator = (
                _instantiate(HistGradientBoostingRegressor, params)
                if task == "regression"
                else _instantiate(HistGradientBoostingClassifier, params)
            )
        elif self.model_type == "mlp":
            estimator = (
                _instantiate(MLPRegressor, params)
                if task == "regression"
                else _instantiate(MLPClassifier, params)
            )
            needs_scaler = True
        elif self.model_type == "logistic" and task == "classification":
            estimator = _instantiate(LogisticRegression, params)
            needs_scaler = True
        elif self.model_type == "ensemble":
            if task != "regression":
                raise ValueError("Ensemble model only supports regression targets.")
            from .modeling import create_default_regression_ensemble

            estimator = create_default_regression_ensemble(**params)
        elif self.model_type in {"lstm", "gru", "transformer"}:
            from .deep_models import (
                GRUClassifier,
                GRURegressor,
                LSTMClassifier,
                LSTMRegressor,
                TransformerRegressor,
            )

            if task == "classification":
                model_cls = {
                    "lstm": LSTMClassifier,
                    "gru": GRUClassifier,
                }.get(self.model_type)
                if model_cls is None:
                    raise ValueError("Transformer classifier is not implemented.")
            else:
                model_cls = {
                    "lstm": LSTMRegressor,
                    "gru": GRURegressor,
                    "transformer": TransformerRegressor,
                }[self.model_type]
            estimator = _instantiate(model_cls, params)
            needs_scaler = True
        else:
            if self.model_type in {"xgboost", "lightgbm", "hist_gb"}:
                LOGGER.warning(
                    "Model type '%s' unavailable; falling back to RandomForest.",
                    self.model_type,
                )
            estimator = (
                _instantiate(RandomForestRegressor, params)
                if task == "regression"
                else _instantiate(RandomForestClassifier, params)
            )

        if calibrate and task == "classification":
            calibration_options = {"cv": 5, "method": "sigmoid"}
            calibration_options.update(calibration_params)
            cal_params = inspect.signature(CalibratedClassifierCV).parameters
            if "estimator" in cal_params:
                estimator = CalibratedClassifierCV(estimator=estimator, **calibration_options)
            else:
                estimator = CalibratedClassifierCV(base_estimator=estimator, **calibration_options)

        steps: list[tuple[str, Any]] = []
        if needs_scaler:
            steps.append(("scaler", StandardScaler()))
        steps.append(("estimator", estimator))
        return Pipeline(steps=steps)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100)
    return {"rmse": rmse, "mae": mae, "mape": mape}


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    classes: Sequence[Any] | None = None,
) -> Dict[str, float]:
    accuracy = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    precision = float(
        precision_score(y_true, y_pred, average="weighted", zero_division=0)
    )
    recall = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))

    metrics: Dict[str, float] = {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

    return metrics


def extract_feature_importance(model: Pipeline, feature_names: list[str]) -> Dict[str, float]:
    estimator = model.named_steps.get("estimator")
    if estimator is None:
        return {}

    importance: Dict[str, float] = {}
    if hasattr(estimator, "feature_importances_"):
        values = getattr(estimator, "feature_importances_")
        for name, score in zip(feature_names, values):
            importance[name] = float(score)
    elif hasattr(estimator, "coef_"):
        coef = getattr(estimator, "coef_")
        if coef.ndim > 1:
            coef = np.abs(coef).mean(axis=0)
        for name, score in zip(feature_names, coef):
            importance[name] = float(abs(score))
    return dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))


def model_supports_proba(model: Pipeline) -> bool:
    estimator = model.named_steps.get("estimator")
    return hasattr(estimator, "predict_proba")
