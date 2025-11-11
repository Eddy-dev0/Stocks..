"""Model factory utilities providing configurable estimators."""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
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
    }

    def __init__(self, model_type: str, overrides: Dict[str, Any] | None = None) -> None:
        self.model_type = model_type.lower()
        self.overrides = overrides or {}

    def create(self, task: str) -> Pipeline:
        """Return a pipeline for the requested task (regression or classification)."""

        params = self.DEFAULT_PARAMS.get(self.model_type, {}).copy()
        params.update(self.overrides)

        estimator: Any
        needs_scaler = False

        if self.model_type == "random_forest":
            estimator = RandomForestRegressor(**params) if task == "regression" else RandomForestClassifier(**params)
        elif self.model_type == "lightgbm" and LGBMRegressor is not None:
            estimator = LGBMRegressor(**params) if task == "regression" else LGBMClassifier(**params)
        elif self.model_type == "xgboost" and XGBRegressor is not None:
            estimator = XGBRegressor(**params) if task == "regression" else XGBClassifier(**params)
        elif self.model_type == "hist_gb" and HistGradientBoostingRegressor is not None:
            estimator = (
                HistGradientBoostingRegressor(**params)
                if task == "regression"
                else HistGradientBoostingClassifier(**params)
            )
        elif self.model_type == "mlp":
            estimator = MLPRegressor(**params) if task == "regression" else MLPClassifier(**params)
            needs_scaler = True
        elif self.model_type == "logistic" and task == "classification":
            estimator = LogisticRegression(**params)
            needs_scaler = True
        else:
            if self.model_type in {"xgboost", "lightgbm", "hist_gb"}:
                LOGGER.warning(
                    "Model type '%s' unavailable; falling back to RandomForest.",
                    self.model_type,
                )
            estimator = RandomForestRegressor(**params) if task == "regression" else RandomForestClassifier(**params)

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


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    accuracy = float(accuracy_score(y_true, y_pred))
    return {"accuracy": accuracy}


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
