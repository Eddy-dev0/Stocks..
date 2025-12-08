"""Thin wrapper around the full featured provider fetcher."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Mapping, Optional, Type

import pandas as pd

from .config import PredictorConfig

try:  # pragma: no cover - exercised indirectly via integration flows
    from ..providers.data import DataFetcher as _ProviderFetcher
except Exception as exc:  # pragma: no cover - defensive import guard
    _PROVIDER_IMPORT_ERROR = exc

    class _ProviderFetcher:  # type: ignore[redeclaration]  # pragma: no cover - stub
        """Fallback stub used when the rich provider stack is unavailable."""

        def __init__(self, config: PredictorConfig) -> None:
            self.config = config
            self._sources: Dict[str, str] = {}

        def _raise(self, method: str) -> None:
            message = (
                "Data fetching requires the optional provider dependencies. "
                f"Attempted to call '{method}' but the provider stack failed to "
                "import."
            )
            raise RuntimeError(message) from _PROVIDER_IMPORT_ERROR

        def fetch_price_data(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
            self._raise("fetch_price_data")

        def fetch_news_data(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
            self._raise("fetch_news_data")

        def fetch_indicator_data(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
            self._raise("fetch_indicator_data")

        def fetch_fundamentals(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
            self._raise("fetch_fundamentals")

        def fetch_corporate_events(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
            self._raise("fetch_corporate_events")

        def fetch_options_surface(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
            self._raise("fetch_options_surface")

        def fetch_sentiment_signals(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
            self._raise("fetch_sentiment_signals")

        def fetch_esg_metrics(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
            self._raise("fetch_esg_metrics")

        def fetch_ownership_flows(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
            self._raise("fetch_ownership_flows")

        def refresh_all(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
            self._raise("refresh_all")

        def get_data_sources(self, *args: Any, **kwargs: Any) -> list[str]:
            return []

        def get_last_source(self, *args: Any, **kwargs: Any) -> Optional[str]:
            return None

        def get_last_updated(self, *args: Any, **kwargs: Any) -> Optional[datetime]:
            return None

        def __getattr__(self, item: str) -> Any:
            self._raise(item)
else:  # pragma: no cover - exercised indirectly via integration flows
    _PROVIDER_IMPORT_ERROR = None


@dataclass(slots=True)
class DataFetcher:  # pragma: no cover - thin delegation layer
    """Delegate data access to the provider implementation.

    The core pipeline only relies on a subset of data access methods such as
    :meth:`fetch_price_data` and :meth:`fetch_news_data`.  Historically this
    module exposed a lightweight stub for unit tests which caused ``AttributeError``
    when the real application attempted to call the richer provider API.  The
    new implementation wraps the provider ``DataFetcher`` so the CLI and other
    integration entry points have access to the full dataset functionality,
    while unit tests can still swap the fetcher with a simple fixture.
    """

    config: PredictorConfig
    delegate_cls: Optional[Type[Any]] = None
    _delegate: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        provider_cls = self.delegate_cls or _ProviderFetcher
        if provider_cls is None:
            message = (
                "The full data fetching stack is unavailable. Ensure optional "
                "dependencies are installed and the provider package imports "
                "correctly."
            )
            if _PROVIDER_IMPORT_ERROR is not None:
                raise RuntimeError(message) from _PROVIDER_IMPORT_ERROR
            raise RuntimeError(message)
        self._delegate = provider_cls(self.config)

    # ------------------------------------------------------------------
    # Delegated public API
    # ------------------------------------------------------------------
    def fetch_price_data(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self._delegate.fetch_price_data(*args, **kwargs)

    def fetch_news_data(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self._delegate.fetch_news_data(*args, **kwargs)

    def fetch_indicator_data(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self._delegate.fetch_indicator_data(*args, **kwargs)

    def fetch_fundamentals(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self._delegate.fetch_fundamentals(*args, **kwargs)

    def fetch_corporate_events(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self._delegate.fetch_corporate_events(*args, **kwargs)

    def fetch_options_surface(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self._delegate.fetch_options_surface(*args, **kwargs)

    def fetch_sentiment_signals(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self._delegate.fetch_sentiment_signals(*args, **kwargs)

    def fetch_esg_metrics(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self._delegate.fetch_esg_metrics(*args, **kwargs)

    def fetch_ownership_flows(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self._delegate.fetch_ownership_flows(*args, **kwargs)

    def refresh_all(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
        return self._delegate.refresh_all(*args, **kwargs)

    def get_data_sources(self, *args: Any, **kwargs: Any) -> list[str]:
        return self._delegate.get_data_sources(*args, **kwargs)

    def get_last_source(self, *args: Any, **kwargs: Any) -> Optional[str]:
        return self._delegate.get_last_source(*args, **kwargs)

    def get_last_updated(self, *args: Any, **kwargs: Any) -> Optional[datetime]:
        return self._delegate.get_last_updated(*args, **kwargs)

    def __getattr__(self, item: str) -> Any:
        """Fallback to the provider for any additional helpers."""

        return getattr(self._delegate, item)
