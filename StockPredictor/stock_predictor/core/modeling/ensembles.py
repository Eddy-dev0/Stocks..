"""Ensemble forecasters combining complementary regressors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV

try:  # Optional dependencies
    from lightgbm import LGBMRegressor  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    LGBMRegressor = None  # type: ignore

try:  # Optional dependencies
    from xgboost import XGBRegressor  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    XGBRegressor = None  # type: ignore


@dataclass(frozen=True)
class EnsembleMember:
    """Configuration for an ensemble member estimator."""

    name: str
    estimator: Any
    weight: float = 1.0


class EnsembleRegressor(BaseEstimator, RegressorMixin):
    """Stacked/ blended ensemble producing point, quantile and interval forecasts."""

    def __init__(
        self,
        members: Optional[Sequence[EnsembleMember]] = None,
        *,
        blender: Optional[Any] = None,
        quantiles: Sequence[float] | None = None,
        interval_alpha: float = 0.2,
    ) -> None:
        self.members = list(members) if members is not None else []
        self.blender = blender
        self.quantiles = tuple(sorted(set(float(q) for q in quantiles))) if quantiles else (0.1, 0.5, 0.9)
        self.interval_alpha = float(interval_alpha)

    def fit(self, X: Any, y: Any) -> "EnsembleRegressor":
        if not self.members:
            raise ValueError("EnsembleRegressor requires at least one member estimator.")

        self._fitted_members_: List[Tuple[str, Any, float]] = []
        X_array = self._to_numpy(X)
        y_array = np.asarray(y, dtype=float)
        member_predictions: List[np.ndarray] = []
        feature_importances: List[np.ndarray] = []
        feature_count = X_array.shape[1] if X_array.ndim == 2 else len(X_array)

        for member in self.members:
            estimator = clone(member.estimator)
            estimator.fit(X, y)
            self._fitted_members_.append((member.name, estimator, float(member.weight)))
            preds = np.asarray(estimator.predict(X), dtype=float).reshape(-1, 1)
            member_predictions.append(preds)
            importance = self._extract_feature_importance(estimator)
            if importance is not None and importance.size == feature_count:
                feature_importances.append(importance)

        if member_predictions:
            self._in_sample_predictions_ = np.hstack(member_predictions)
        else:
            self._in_sample_predictions_ = np.empty((len(y_array), 0))

        if feature_importances:
            stacked = np.vstack(feature_importances)
            self.feature_importances_ = stacked.mean(axis=0)
        else:
            self.feature_importances_ = None

        if self.blender is not None and self._in_sample_predictions_.size:
            blender = clone(self.blender)
            blender.fit(self._in_sample_predictions_, y_array)
            self._blender_ = blender
        else:
            self._blender_ = None

        weights = np.array([weight for _, _, weight in self._fitted_members_], dtype=float)
        weights = np.where(np.isfinite(weights), weights, 0.0)
        if weights.sum() <= 0:
            weights = np.ones_like(weights)
        self._weights_ = weights / weights.sum()

        self._update_uncertainty_cache(self._in_sample_predictions_)
        return self

    def predict(self, X: Any) -> np.ndarray:
        member_preds = self._collect_member_predictions(X)
        if member_preds.size == 0:
            return np.zeros(self._to_numpy(X).shape[0])

        weighted = member_preds @ self._weights_
        if self._blender_ is not None:
            blended = self._blender_.predict(member_preds)
            point = 0.5 * (weighted + blended)
        else:
            point = weighted
        self._update_uncertainty_cache(member_preds)
        return point

    def predict_quantiles(
        self, X: Any, quantiles: Optional[Iterable[float]] = None
    ) -> Dict[float, np.ndarray]:
        quantiles = tuple(sorted(set(float(q) for q in (quantiles or self.quantiles))))
        member_preds = self._collect_member_predictions(X)
        if member_preds.size == 0:
            return {q: np.zeros(self._to_numpy(X).shape[0]) for q in quantiles}
        return self._quantiles_from_member_preds(member_preds, quantiles)

    def prediction_interval(
        self, X: Any, alpha: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        alpha = float(alpha if alpha is not None else self.interval_alpha)
        lower_q = max(0.0, min(1.0, alpha / 2))
        upper_q = max(0.0, min(1.0, 1 - alpha / 2))
        member_preds = self._collect_member_predictions(X)
        quantiles = self._quantiles_from_member_preds(member_preds, (lower_q, 0.5, upper_q))
        return {
            "lower": quantiles.get(lower_q),
            "median": quantiles.get(0.5),
            "upper": quantiles.get(upper_q),
            "alpha": alpha,
        }

    def get_uncertainty_summary(self, X: Any | None = None) -> Dict[str, Any]:
        if X is not None:
            member_preds = self._collect_member_predictions(X)
        else:
            member_preds = getattr(self, "_last_member_predictions_", None)
        if member_preds is None or member_preds.size == 0:
            return {}
        dispersion = np.nanstd(member_preds, axis=1, ddof=1)
        median_std = float(np.nanmedian(dispersion))
        mean_std = float(np.nanmean(dispersion))
        spread = np.nanpercentile(member_preds, 75, axis=1) - np.nanpercentile(
            member_preds, 25, axis=1
        )
        quantile_summary = self._quantiles_from_member_preds(
            member_preds, self.quantiles
        )
        interval_summary = {"alpha": self.interval_alpha}
        lower_q = max(0.0, min(1.0, self.interval_alpha / 2))
        upper_q = max(0.0, min(1.0, 1 - self.interval_alpha / 2))
        interval_quantiles = self._quantiles_from_member_preds(
            member_preds, (lower_q, 0.5, upper_q)
        )
        for key, quantile_key in (("lower", lower_q), ("median", 0.5), ("upper", upper_q)):
            values = interval_quantiles.get(quantile_key)
            if values is not None:
                interval_summary[key] = float(np.nanmean(values))
        return {
            "std": mean_std,
            "median_std": median_std,
            "iqr": float(np.nanmean(spread)),
            "quantiles": {float(k): float(np.nanmean(v)) for k, v in quantile_summary.items()},
            "interval": interval_summary,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _collect_member_predictions(self, X: Any) -> np.ndarray:
        X_array = self._to_numpy(X)
        predictions: List[np.ndarray] = []
        for _, estimator, _ in getattr(self, "_fitted_members_", []):
            preds = np.asarray(estimator.predict(X), dtype=float).reshape(-1, 1)
            predictions.append(preds)
        if not predictions:
            member_preds = np.empty((len(X_array), 0))
        else:
            member_preds = np.hstack(predictions)
        self._update_uncertainty_cache(member_preds)
        return member_preds

    def _quantiles_from_member_preds(
        self, member_preds: np.ndarray, quantiles: Iterable[float]
    ) -> Dict[float, np.ndarray]:
        quantiles = tuple(sorted(set(float(q) for q in quantiles)))
        if member_preds.size == 0:
            return {q: np.array([]) for q in quantiles}
        results: Dict[float, np.ndarray] = {}
        for q in quantiles:
            results[q] = np.quantile(member_preds, q, axis=1)
        return results

    def _update_uncertainty_cache(self, member_preds: np.ndarray) -> None:
        self._last_member_predictions_ = member_preds
        if member_preds.size:
            self._last_member_std_ = np.nanstd(member_preds, axis=1)
        else:
            self._last_member_std_ = None

    @staticmethod
    def _to_numpy(X: Any) -> np.ndarray:
        if isinstance(X, np.ndarray):
            return X
        if hasattr(X, "to_numpy"):
            return np.asarray(X.to_numpy(), dtype=float)
        return np.asarray(X, dtype=float)

    @staticmethod
    def _extract_feature_importance(estimator: Any) -> Optional[np.ndarray]:
        if hasattr(estimator, "feature_importances_"):
            return np.asarray(getattr(estimator, "feature_importances_"), dtype=float)
        if hasattr(estimator, "coef_"):
            coef = getattr(estimator, "coef_")
            coef_arr = np.asarray(coef, dtype=float)
            if coef_arr.ndim > 1:
                coef_arr = np.mean(np.abs(coef_arr), axis=0)
            return np.abs(coef_arr)
        return None


def create_default_regression_ensemble(
    *,
    random_state: int | None = 42,
    quantiles: Sequence[float] | None = None,
    interval_alpha: float = 0.2,
    members: Sequence[EnsembleMember] | None = None,
) -> EnsembleRegressor:
    """Return a diversified ensemble suitable for price/return forecasts."""

    if members is None:
        members = _build_default_members(random_state)

    blender = RidgeCV(alphas=(0.1, 1.0, 10.0))
    return EnsembleRegressor(
        members=members,
        blender=blender,
        quantiles=quantiles,
        interval_alpha=interval_alpha,
    )


def _build_default_members(random_state: int | None) -> List[EnsembleMember]:
    members: List[EnsembleMember] = []
    rf = RandomForestRegressor(
        n_estimators=400,
        random_state=random_state,
        n_jobs=-1,
        max_depth=None,
        min_samples_leaf=2,
    )
    members.append(EnsembleMember("random_forest", rf, weight=1.0))

    gb = GradientBoostingRegressor(random_state=random_state, learning_rate=0.05)
    members.append(EnsembleMember("gradient_boosting", gb, weight=0.9))

    if LGBMRegressor is not None:
        lgbm = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=random_state,
        )
        members.append(EnsembleMember("lightgbm", lgbm, weight=1.1))

    if XGBRegressor is not None:
        xgb = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.7,
            random_state=random_state,
            tree_method="hist",
            n_jobs=-1,
        )
        members.append(EnsembleMember("xgboost", xgb, weight=1.1))

    linear = LinearRegression()
    members.append(EnsembleMember("linear", linear, weight=0.6))

    return members
