"""Utilities for building reusable preprocessing pipelines for model training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _ensure_dataframe(
    data: pd.DataFrame | np.ndarray,
    *,
    columns: Sequence[str] | None = None,
    index: Sequence | None = None,
) -> pd.DataFrame:
    """Return *data* as a pandas ``DataFrame`` with optional column alignment."""

    if isinstance(data, pd.DataFrame):
        frame = data.copy()
        if columns is not None:
            frame = frame.reindex(columns=list(columns))
        return frame

    array = np.asarray(data)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if columns is None:
        columns = [f"feature_{idx}" for idx in range(array.shape[1])]
    if index is None:
        index = range(array.shape[0])
    return pd.DataFrame(array, columns=list(columns), index=index)


class _BaseDataFrameTransformer(BaseEstimator, TransformerMixin):
    """Transformer base-class that enforces DataFrame inputs/outputs."""

    feature_names_: list[str]

    def _set_feature_names(self, names: Iterable[str]) -> None:
        self.feature_names_ = [str(name) for name in names]

    def get_feature_names_out(self) -> list[str]:
        return list(getattr(self, "feature_names_", []))


class NonFiniteHandler(_BaseDataFrameTransformer):
    """Replace non-finite values with ``NaN`` for downstream imputers."""

    def fit(self, X: pd.DataFrame, y=None):  # type: ignore[override]
        frame = _ensure_dataframe(X)
        self._set_feature_names(frame.columns)
        return self

    def transform(self, X: pd.DataFrame):  # type: ignore[override]
        frame = _ensure_dataframe(X, columns=self.feature_names_)
        return frame.replace([np.inf, -np.inf], np.nan)


class DataFrameSimpleImputer(_BaseDataFrameTransformer):
    """Wrapper around :class:`SimpleImputer` preserving DataFrame structure."""

    def __init__(self, strategy: str = "median") -> None:
        self.strategy = strategy
        self.imputer = SimpleImputer(strategy=strategy)

    def fit(self, X: pd.DataFrame, y=None):  # type: ignore[override]
        frame = _ensure_dataframe(X)
        self.imputer.set_params(strategy=self.strategy)
        self.imputer.fit(frame, y)
        # Align stored feature names with the imputer's learned schema to avoid
        # mismatches during ``transform`` if the underlying estimator trims or
        # reorders inputs.
        expected_features = getattr(
            self.imputer, "feature_names_in_", frame.columns
        )
        if hasattr(self.imputer, "get_feature_names_out"):
            expected_features = self.imputer.get_feature_names_out()
        self._set_feature_names(expected_features)
        return self

    def transform(self, X: pd.DataFrame):  # type: ignore[override]
        expected_features = getattr(self.imputer, "feature_names_in_", self.feature_names_)
        frame = _ensure_dataframe(X, columns=expected_features)
        transformed = self.imputer.transform(frame)
        return pd.DataFrame(
            transformed, columns=self.feature_names_, index=frame.index
        )


class OutlierClipper(_BaseDataFrameTransformer):
    """Clip extreme values using quantile-based winsorisation."""

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float | None = None) -> None:
        if not 0.0 <= lower_quantile < 0.5:
            raise ValueError("lower_quantile must be between 0 and 0.5")
        if upper_quantile is None:
            upper_quantile = 1.0 - lower_quantile
        if not 0.5 < upper_quantile <= 1.0:
            raise ValueError("upper_quantile must be between 0.5 and 1.0")
        self.lower_quantile = float(lower_quantile)
        self.upper_quantile = float(upper_quantile)

    def fit(self, X: pd.DataFrame, y=None):  # type: ignore[override]
        frame = _ensure_dataframe(X)
        self._set_feature_names(frame.columns)
        numeric = frame.to_numpy(dtype=float)
        self.lower_bounds_ = np.nanquantile(numeric, self.lower_quantile, axis=0)
        self.upper_bounds_ = np.nanquantile(numeric, self.upper_quantile, axis=0)
        return self

    def transform(self, X: pd.DataFrame):  # type: ignore[override]
        frame = _ensure_dataframe(X, columns=self.feature_names_)
        numeric = frame.to_numpy(dtype=float)
        clipped = np.clip(numeric, self.lower_bounds_, self.upper_bounds_)
        return pd.DataFrame(clipped, columns=self.feature_names_, index=frame.index)


class DataFrameStandardScaler(_BaseDataFrameTransformer):
    """Standardise features while preserving DataFrame semantics."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()

    def fit(self, X: pd.DataFrame, y=None):  # type: ignore[override]
        frame = _ensure_dataframe(X)
        self.scaler.fit(frame, y)
        self._set_feature_names(frame.columns)
        return self

    def transform(self, X: pd.DataFrame):  # type: ignore[override]
        frame = _ensure_dataframe(X, columns=self.feature_names_)
        transformed = self.scaler.transform(frame)
        return pd.DataFrame(transformed, columns=self.feature_names_, index=frame.index)


class VarianceFilter(_BaseDataFrameTransformer):
    """Drop columns whose variance falls below a configured threshold."""

    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = float(threshold)
        self.selector = VarianceThreshold(threshold=self.threshold)

    def fit(self, X: pd.DataFrame, y=None):  # type: ignore[override]
        frame = _ensure_dataframe(X)
        self.selector.set_params(threshold=self.threshold)
        self.selector.fit(frame, y)
        mask = self.selector.get_support()
        columns = frame.columns.to_numpy()
        if mask.size == 0 or mask.sum() == 0:
            self._set_feature_names(columns)
        else:
            self._set_feature_names(columns[mask])
        return self

    def transform(self, X: pd.DataFrame):  # type: ignore[override]
        frame = _ensure_dataframe(X)
        transformed = self.selector.transform(frame)
        if transformed.ndim == 1:
            transformed = transformed.reshape(-1, 1)
        return pd.DataFrame(transformed, columns=self.feature_names_, index=frame.index)


class CorrelationFilter(_BaseDataFrameTransformer):
    """Remove highly correlated features using a greedy selection approach."""

    def __init__(self, threshold: float = 0.98) -> None:
        self.threshold = float(threshold)

    def fit(self, X: pd.DataFrame, y=None):  # type: ignore[override]
        frame = _ensure_dataframe(X)
        self._set_feature_names(self._select_columns(frame))
        return self

    def transform(self, X: pd.DataFrame):  # type: ignore[override]
        frame = _ensure_dataframe(X)
        available = [name for name in self.feature_names_ if name in frame.columns]
        return frame.loc[:, available]

    def _select_columns(self, frame: pd.DataFrame) -> list[str]:
        if frame.shape[1] <= 1:
            return list(frame.columns)
        corr = frame.corr().abs().fillna(0.0)
        selected: list[str] = []
        dropped: set[str] = set()
        for column in corr.columns:
            if column in dropped:
                continue
            selected.append(column)
            high_corr = corr.index[(corr[column] > self.threshold) & (corr.index != column)]
            dropped.update(set(high_corr))
        if not selected:
            return list(frame.columns)
        return selected


class PCAReducer(_BaseDataFrameTransformer):
    """Apply Principal Component Analysis for dimensionality reduction."""

    def __init__(
        self,
        n_components: float | int = 0.95,
        *,
        random_state: int | None = 42,
        whiten: bool = False,
    ) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self.whiten = bool(whiten)
        self.pca = PCA(n_components=n_components, random_state=random_state, whiten=self.whiten)

    def fit(self, X: pd.DataFrame, y=None):  # type: ignore[override]
        frame = _ensure_dataframe(X)
        self.pca.set_params(n_components=self.n_components, whiten=self.whiten)
        self.pca.fit(frame, y)
        count = int(getattr(self.pca, "n_components_", frame.shape[1]))
        self._set_feature_names([f"PC{i}" for i in range(1, count + 1)])
        return self

    def transform(self, X: pd.DataFrame):  # type: ignore[override]
        frame = _ensure_dataframe(X)
        transformed = self.pca.transform(frame)
        return pd.DataFrame(transformed, columns=self.feature_names_, index=frame.index)


def _infer_task_type(y: pd.Series | np.ndarray | None) -> str:
    if y is None:
        return "regression"
    series = pd.Series(y)
    if series.empty:
        return "regression"
    if pd.api.types.is_bool_dtype(series) or series.nunique() <= 10:
        return "classification"
    return "regression"


class EmbeddedImportanceSelector(_BaseDataFrameTransformer):
    """Use embedded model importance scores to select a subset of features."""

    def __init__(
        self,
        *,
        model_type: str = "random_forest",
        threshold: str | float = "median",
        max_features: int | None = None,
        random_state: int | None = 42,
    ) -> None:
        self.model_type = model_type
        self.threshold = threshold
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y=None):  # type: ignore[override]
        if y is None:
            raise ValueError("Embedded importance selection requires target values.")
        frame = _ensure_dataframe(X)
        task = _infer_task_type(y)
        estimator = self._build_estimator(task)
        selector = SelectFromModel(
            estimator,
            threshold=self.threshold,
            max_features=self.max_features,
        )
        selector.fit(frame, y)
        support = selector.get_support()
        columns = frame.columns.to_numpy()
        if support.any():
            selected = columns[support]
        else:
            selected = columns
        self.selector_ = selector
        self._set_feature_names(selected)
        return self

    def transform(self, X: pd.DataFrame):  # type: ignore[override]
        frame = _ensure_dataframe(X)
        transformed = self.selector_.transform(frame)
        if transformed.ndim == 1:
            transformed = transformed.reshape(-1, 1)
        columns = self.feature_names_
        if transformed.shape[1] != len(columns):
            columns = [f"Feature_{idx}" for idx in range(1, transformed.shape[1] + 1)]
        return pd.DataFrame(transformed, columns=columns, index=frame.index)

    def _build_estimator(self, task: str):
        params = {"random_state": self.random_state}
        if task == "classification":
            return RandomForestClassifier(**params)
        return RandomForestRegressor(**params)


class DataFrameIdentity(_BaseDataFrameTransformer):
    """Pass-through transformer that records column names explicitly."""

    def fit(self, X: pd.DataFrame, y=None):  # type: ignore[override]
        frame = _ensure_dataframe(X)
        self._set_feature_names(frame.columns)
        return self

    def transform(self, X: pd.DataFrame):  # type: ignore[override]
        frame = _ensure_dataframe(X, columns=self.feature_names_)
        return frame


@dataclass
class PreprocessingBuilder:
    """Factory for constructing preprocessing pipelines with consistent defaults."""

    imputation_strategy: str = "median"
    clip_outliers: bool = True
    clip_quantile: float = 0.01
    variance_threshold: float = 0.0
    correlation_threshold: float = 0.98
    use_pca: bool = False
    pca_components: float | int = 0.95
    whiten: bool = False
    embedded_importance: bool = False
    embedded_model_type: str = "random_forest"
    embedded_threshold: str | float = "median"
    embedded_max_features: int | None = None
    random_state: int | None = 42

    def create_pipeline(self) -> Pipeline:
        steps: list[tuple[str, TransformerMixin]] = [
            ("finite", NonFiniteHandler()),
            ("imputer", DataFrameSimpleImputer(strategy=self.imputation_strategy)),
        ]
        if self.clip_outliers:
            steps.append(
                (
                    "clipper",
                    OutlierClipper(
                        lower_quantile=self.clip_quantile,
                        upper_quantile=1.0 - self.clip_quantile,
                    ),
                )
            )
        steps.append(("scaler", DataFrameStandardScaler()))
        if self.variance_threshold > 0:
            steps.append(("variance", VarianceFilter(threshold=self.variance_threshold)))
        if self.correlation_threshold < 1.0:
            steps.append(("correlation", CorrelationFilter(threshold=self.correlation_threshold)))
        if self.use_pca:
            steps.append(
                (
                    "pca",
                    PCAReducer(
                        n_components=self.pca_components,
                        random_state=self.random_state,
                        whiten=self.whiten,
                    ),
                )
            )
        if self.embedded_importance:
            steps.append(
                (
                    "embedded",
                    EmbeddedImportanceSelector(
                        model_type=self.embedded_model_type,
                        threshold=self.embedded_threshold,
                        max_features=self.embedded_max_features,
                        random_state=self.random_state,
                    ),
                )
            )
        steps.append(("to_frame", DataFrameIdentity()))
        return Pipeline(steps=steps)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Pipeline:
        pipeline = self.create_pipeline()
        pipeline.fit(X, y)
        return pipeline


def build_preprocessing_pipeline(
    X: pd.DataFrame,
    *,
    y: pd.Series | None = None,
    options: Mapping[str, object] | None = None,
) -> Pipeline:
    """Convenience helper returning a fitted preprocessing pipeline."""

    opts = dict(options or {})
    builder = PreprocessingBuilder(**opts)
    return builder.fit(X, y)


def get_feature_names_from_pipeline(pipeline: Pipeline) -> list[str]:
    """Extract the output feature names from the final pipeline step."""

    for _, step in reversed(pipeline.steps):
        if hasattr(step, "get_feature_names_out"):
            names = step.get_feature_names_out()
            if isinstance(names, (list, tuple)):
                return list(names)
            if isinstance(names, np.ndarray):
                return names.tolist()
        if hasattr(step, "feature_names_"):
            names = getattr(step, "feature_names_")
            if isinstance(names, (list, tuple, np.ndarray)):
                return list(names)
    return []


__all__ = [
    "PreprocessingBuilder",
    "build_preprocessing_pipeline",
    "get_feature_names_from_pipeline",
    "clone",
]
