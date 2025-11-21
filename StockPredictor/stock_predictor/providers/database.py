"""Database utilities for persisting market data."""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator

import pandas as pd
from sqlalchemy import (
    Date,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    select,
)
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

LOGGER = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base class for all ORM models."""


class MetaEntry(Base):
    """Key/value store for keeping metadata such as refresh timestamps."""

    __tablename__ = "meta"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "updated_at": self.updated_at,
        }


class Price(Base):
    """Daily OHLCV prices."""

    __tablename__ = "prices"

    ticker: Mapped[str] = mapped_column(String(32), primary_key=True)
    interval: Mapped[str] = mapped_column(String(16), primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    open: Mapped[float | None] = mapped_column(Float)
    high: Mapped[float | None] = mapped_column(Float)
    low: Mapped[float | None] = mapped_column(Float)
    close: Mapped[float | None] = mapped_column(Float)
    adj_close: Mapped[float | None] = mapped_column(Float)
    volume: Mapped[float | None] = mapped_column(Float)

    def to_frame_dict(self) -> dict[str, Any]:
        return {
            "Ticker": self.ticker,
            "Interval": self.interval,
            "Date": pd.to_datetime(self.date),
            "Open": self.open,
            "High": self.high,
            "Low": self.low,
            "Close": self.close,
            "Adj Close": self.adj_close,
            "Volume": self.volume,
        }


class Indicator(Base):
    """Technical and macro indicators aligned with price data."""

    __tablename__ = "indicators"

    ticker: Mapped[str] = mapped_column(String(32), primary_key=True)
    interval: Mapped[str] = mapped_column(String(16), primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    category: Mapped[str] = mapped_column(String(32), default="technical")
    value: Mapped[float | None] = mapped_column(Float)
    extra: Mapped[str | None] = mapped_column(Text)

    def to_frame_dict(self) -> dict[str, Any]:
        payload = json.loads(self.extra) if self.extra else None
        return {
            "Ticker": self.ticker,
            "Interval": self.interval,
            "Date": pd.to_datetime(self.date),
            "Indicator": self.name,
            "Category": self.category,
            "Value": self.value,
            "Extra": payload,
        }


class Fundamental(Base):
    """Fundamental statement data normalised into a long format."""

    __tablename__ = "fundamentals"

    ticker: Mapped[str] = mapped_column(String(32), primary_key=True)
    statement: Mapped[str] = mapped_column(String(32), primary_key=True)
    period: Mapped[str] = mapped_column(String(32), primary_key=True)
    as_of: Mapped[date] = mapped_column(Date, primary_key=True)
    metric: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[float | None] = mapped_column(Float)
    raw: Mapped[str | None] = mapped_column(Text)

    def to_frame_dict(self) -> dict[str, Any]:
        raw_payload = json.loads(self.raw) if self.raw else None
        return {
            "Ticker": self.ticker,
            "Statement": self.statement,
            "Period": self.period,
            "AsOf": pd.to_datetime(self.as_of),
            "Metric": self.metric,
            "Value": self.value,
            "Raw": raw_payload,
        }


class NewsArticle(Base):
    """News articles tagged to a ticker symbol."""

    __tablename__ = "news"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(32), nullable=False)
    published_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    title: Mapped[str | None] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text)
    url: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str | None] = mapped_column(String(64))

    __table_args__ = (
        UniqueConstraint("ticker", "published_at", "title", name="uq_news_article"),
        {"sqlite_autoincrement": True},
    )

    def to_frame_dict(self) -> dict[str, Any]:
        return {
            "Ticker": self.ticker,
            "PublishedAt": pd.to_datetime(self.published_at),
            "Title": self.title,
            "Summary": self.summary,
            "Url": self.url,
            "Source": self.source,
        }


class CorporateEvent(Base):
    """Corporate actions such as dividends, splits or earnings calls."""

    __tablename__ = "corporate_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(32), nullable=False)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    event_date: Mapped[date] = mapped_column(Date, nullable=False)
    reference: Mapped[str] = mapped_column(String(64), default="general", nullable=False)
    value: Mapped[float | None] = mapped_column(Float)
    currency: Mapped[str | None] = mapped_column(String(16))
    details: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "ticker", "event_type", "event_date", "reference", name="uq_corporate_event"
        ),
        {"sqlite_autoincrement": True},
    )

    def to_frame_dict(self) -> dict[str, Any]:
        payload = json.loads(self.details) if self.details else None
        return {
            "Ticker": self.ticker,
            "EventType": self.event_type,
            "EventDate": pd.to_datetime(self.event_date),
            "Reference": self.reference,
            "Value": self.value,
            "Currency": self.currency,
            "Details": payload,
            "Source": self.source,
        }


class ResearchArtifact(Base):
    """Persisted research notes produced by the research services."""

    __tablename__ = "research_artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    captured_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="crawler")
    raw_content: Mapped[str] = mapped_column(Text, nullable=False)
    extractive_summary: Mapped[str | None] = mapped_column(Text)
    abstractive_summary: Mapped[str | None] = mapped_column(Text)
    sentiment_label: Mapped[str | None] = mapped_column(String(16))
    sentiment_score: Mapped[float | None] = mapped_column(Float)
    metadata_json: Mapped[str | None] = mapped_column("metadata", Text)

    __table_args__ = (
        UniqueConstraint("url", "content_hash", name="uq_research_artifact"),
        {"sqlite_autoincrement": True},
    )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "url": self.url,
            "content_hash": self.content_hash,
            "captured_at": self.captured_at,
            "source": self.source,
            "raw_content": self.raw_content,
            "extractive_summary": self.extractive_summary,
            "abstractive_summary": self.abstractive_summary,
            "sentiment_label": self.sentiment_label,
            "sentiment_score": self.sentiment_score,
        }
        if self.metadata_json:
            try:
                payload["metadata"] = json.loads(self.metadata_json)
            except (TypeError, json.JSONDecodeError):
                payload["metadata"] = self.metadata_json
        else:
            payload["metadata"] = None
        return payload


class OptionSurfacePoint(Base):
    """Point-in-time option greeks and implied volatility metrics."""

    __tablename__ = "option_surface"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(32), nullable=False)
    as_of: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    expiration: Mapped[date] = mapped_column(Date, nullable=False)
    strike: Mapped[float] = mapped_column(Float, nullable=False)
    option_type: Mapped[str] = mapped_column(String(8), nullable=False)
    metric: Mapped[str] = mapped_column(String(64), nullable=False)
    value: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str | None] = mapped_column(String(64))
    extra: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        UniqueConstraint(
            "ticker",
            "as_of",
            "expiration",
            "strike",
            "option_type",
            "metric",
            name="uq_option_surface_point",
        ),
        {"sqlite_autoincrement": True},
    )

    def to_frame_dict(self) -> dict[str, Any]:
        extra_payload = json.loads(self.extra) if self.extra else None
        return {
            "Ticker": self.ticker,
            "AsOf": pd.to_datetime(self.as_of),
            "Expiration": pd.to_datetime(self.expiration),
            "Strike": self.strike,
            "OptionType": self.option_type,
            "Metric": self.metric,
            "Value": self.value,
            "Source": self.source,
            "Extra": extra_payload,
        }


class SentimentSignal(Base):
    """Sentiment and alternative signals aligned to a ticker."""

    __tablename__ = "sentiment_signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(32), nullable=False)
    as_of: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    signal_type: Mapped[str] = mapped_column(String(64), nullable=False)
    score: Mapped[float | None] = mapped_column(Float)
    magnitude: Mapped[float | None] = mapped_column(Float)
    payload: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        UniqueConstraint(
            "ticker", "as_of", "provider", "signal_type", name="uq_sentiment_signal"
        ),
        {"sqlite_autoincrement": True},
    )

    def to_frame_dict(self) -> dict[str, Any]:
        payload = json.loads(self.payload) if self.payload else None
        return {
            "Ticker": self.ticker,
            "AsOf": pd.to_datetime(self.as_of),
            "Provider": self.provider,
            "SignalType": self.signal_type,
            "Score": self.score,
            "Magnitude": self.magnitude,
            "Payload": payload,
        }


class ESGMetricEntry(Base):
    """Environmental, Social and Governance metrics."""

    __tablename__ = "esg_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(32), nullable=False)
    as_of: Mapped[date] = mapped_column(Date, nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    metric: Mapped[str] = mapped_column(String(64), nullable=False)
    value: Mapped[float | None] = mapped_column(Float)
    raw: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        UniqueConstraint(
            "ticker", "as_of", "provider", "metric", name="uq_esg_metric"
        ),
        {"sqlite_autoincrement": True},
    )

    def to_frame_dict(self) -> dict[str, Any]:
        raw_payload = json.loads(self.raw) if self.raw else None
        return {
            "Ticker": self.ticker,
            "AsOf": pd.to_datetime(self.as_of),
            "Provider": self.provider,
            "Metric": self.metric,
            "Value": self.value,
            "Raw": raw_payload,
        }


class OwnershipFlow(Base):
    """Ownership and fund flow metrics for a ticker."""

    __tablename__ = "ownership_flows"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(32), nullable=False)
    as_of: Mapped[date] = mapped_column(Date, nullable=False)
    holder: Mapped[str] = mapped_column(String(128), nullable=False)
    holder_type: Mapped[str] = mapped_column(String(32), nullable=False)
    metric: Mapped[str] = mapped_column(String(64), nullable=False)
    value: Mapped[float | None] = mapped_column(Float)
    raw: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str | None] = mapped_column(String(64))

    __table_args__ = (
        UniqueConstraint(
            "ticker",
            "as_of",
            "holder",
            "metric",
            name="uq_ownership_flow",
        ),
        {"sqlite_autoincrement": True},
    )

    def to_frame_dict(self) -> dict[str, Any]:
        raw_payload = json.loads(self.raw) if self.raw else None
        return {
            "Ticker": self.ticker,
            "AsOf": pd.to_datetime(self.as_of),
            "Holder": self.holder,
            "HolderType": self.holder_type,
            "Metric": self.metric,
            "Value": self.value,
            "Raw": raw_payload,
            "Source": self.source,
        }


class ExperimentLog(Base):
    """Track metadata about training, inference, and backtesting runs."""

    __tablename__ = "experiment_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(32), nullable=False)
    target: Mapped[str] = mapped_column(String(64), nullable=False)
    run_type: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    parameters: Mapped[str | None] = mapped_column(Text)
    metrics: Mapped[str | None] = mapped_column(Text)
    context: Mapped[str | None] = mapped_column(Text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "ticker": self.ticker,
            "target": self.target,
            "run_type": self.run_type,
            "created_at": self.created_at,
            "parameters": json.loads(self.parameters) if self.parameters else None,
            "metrics": json.loads(self.metrics) if self.metrics else None,
            "context": json.loads(self.context) if self.context else None,
        }


if TYPE_CHECKING:  # pragma: no cover
    from .config import PredictorConfig


@dataclass(slots=True)
class Database:
    """High level database helper built on top of SQLAlchemy."""

    url: str
    engine: Engine = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.url.startswith("sqlite:///"):
            db_path = Path(self.url.replace("sqlite:///", ""))
            db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(self.url, future=True)
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = Session(self.engine)
        try:
            yield session
            session.commit()
        except Exception:  # pylint: disable=broad-except
            session.rollback()
            raise
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Meta helpers
    # ------------------------------------------------------------------
    def set_meta(self, key: str, value: Any) -> None:
        payload = json.dumps(value, default=str) if not isinstance(value, str) else value
        stmt = insert(MetaEntry).values(key=key, value=payload)
        stmt = stmt.on_conflict_do_update(index_elements=[MetaEntry.key], set_={"value": payload})
        with self.session() as session:
            session.execute(stmt)

    def get_meta(self, key: str) -> Any | None:
        value: str | None = None
        with self.session() as session:
            result = (
                session.execute(select(MetaEntry).where(MetaEntry.key == key))
                .scalar_one_or_none()
            )
            if result is not None:
                value = result.value
        if value is None:
            return None
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return value

    @staticmethod
    def _indicator_columns_key(ticker: str, interval: str) -> str:
        return f"{ticker}:{interval}:indicator_columns"

    def set_indicator_columns(
        self, ticker: str, interval: str, columns: Iterable[str]
    ) -> None:
        cleaned: list[str] = []
        for column in columns:
            label = str(column).strip()
            if not label:
                continue
            if label in cleaned:
                continue
            cleaned.append(label)
        key = self._indicator_columns_key(ticker, interval)
        self.set_meta(key, cleaned)

    def get_indicator_columns(self, ticker: str, interval: str) -> list[str]:
        key = self._indicator_columns_key(ticker, interval)
        raw = self.get_meta(key)
        if isinstance(raw, list):
            values: list[str] = []
            for column in raw:
                label = str(column).strip()
                if label:
                    values.append(label)
            return values
        if isinstance(raw, str):
            return [part.strip() for part in raw.split(",") if part.strip()]
        return []

    # ------------------------------------------------------------------
    # Upsert helpers
    # ------------------------------------------------------------------
    def upsert_prices(self, ticker: str, interval: str, frame: pd.DataFrame) -> int:
        if frame.empty:
            return 0
        records = []
        for row in frame.itertuples(index=False):
            date_val = self._coerce_date(getattr(row, "Date", getattr(row, "date", None)))
            records.append(
                {
                    "ticker": ticker,
                    "interval": interval,
                    "date": date_val,
                    "open": self._coerce_float(getattr(row, "Open", None)),
                    "high": self._coerce_float(getattr(row, "High", None)),
                    "low": self._coerce_float(getattr(row, "Low", None)),
                    "close": self._coerce_float(getattr(row, "Close", None)),
                    "adj_close": self._coerce_float(getattr(row, "Adj_Close", getattr(row, "AdjClose", getattr(row, "Adj Close", None)))),
                    "volume": self._coerce_float(getattr(row, "Volume", None)),
                }
            )
        if not records:
            return 0

        processed = 0
        columns_per_row = len(records[0]) if records else 0

        with self.session() as session:
            bind = session.get_bind()
            dialect = getattr(bind, "dialect", None)
            max_parameters = getattr(dialect, "max_parameters", None)

            if columns_per_row:
                effective_max_parameters = max_parameters if max_parameters is not None else 900
                batch_size = max(1, effective_max_parameters // columns_per_row)
            else:
                batch_size = len(records) or 1

            for start in range(0, len(records), batch_size):
                batch = records[start : start + batch_size]
                if not batch:
                    continue
                stmt = insert(Price).values(batch)
                stmt = stmt.on_conflict_do_update(
                    index_elements=[Price.ticker, Price.interval, Price.date],
                    set_={
                        "open": stmt.excluded.open,
                        "high": stmt.excluded.high,
                        "low": stmt.excluded.low,
                        "close": stmt.excluded.close,
                        "adj_close": stmt.excluded.adj_close,
                        "volume": stmt.excluded.volume,
                    },
                )
                session.execute(stmt)
                processed += len(batch)

        return processed

    def upsert_indicators(
        self,
        ticker: str,
        interval: str,
        records: Iterable[dict[str, Any]],
    ) -> int:
        batch_size = 200
        total_processed = 0
        batch: list[dict[str, Any]] = []

        with self.session() as session:

            def execute_batch(chunk: list[dict[str, Any]]) -> int:
                if not chunk:
                    return 0
                stmt = insert(Indicator).values(chunk)
                stmt = stmt.on_conflict_do_update(
                    index_elements=[
                        Indicator.ticker,
                        Indicator.interval,
                        Indicator.date,
                        Indicator.name,
                    ],
                    set_={
                        "category": stmt.excluded.category,
                        "value": stmt.excluded.value,
                        "extra": stmt.excluded.extra,
                    },
                )
                try:
                    with session.begin():
                        session.execute(stmt)
                except SQLAlchemyError as exc:  # pragma: no cover - defensive logging
                    LOGGER.error(
                        "Failed to upsert %d indicator rows for %s %s: %s",
                        len(chunk),
                        ticker,
                        interval,
                        exc,
                    )
                    return 0
                return len(chunk)

            for record in records:
                batch.append(
                    {
                        "ticker": ticker,
                        "interval": interval,
                        "date": self._coerce_date(record.get("Date")),
                        "name": record.get("Indicator") or record.get("name"),
                        "category": record.get("Category", "technical"),
                        "value": self._coerce_float(record.get("Value")),
                        "extra": json.dumps(record.get("Extra"))
                        if record.get("Extra") is not None
                        else None,
                    }
                )
                if len(batch) >= batch_size:
                    total_processed += execute_batch(batch)
                    batch = []

            if batch:
                total_processed += execute_batch(batch)

        return total_processed

    def upsert_fundamentals(self, records: Iterable[dict[str, Any]]) -> int:
        payload = []
        for record in records:
            payload.append(
                {
                    "ticker": record["Ticker"],
                    "statement": record["Statement"],
                    "period": record["Period"],
                    "as_of": self._coerce_date(record["AsOf"]),
                    "metric": record["Metric"],
                    "value": self._coerce_float(record.get("Value")),
                    "raw": json.dumps(record.get("Raw"), default=str)
                    if record.get("Raw") is not None
                    else None,
                }
            )
        if not payload:
            return 0
        stmt = insert(Fundamental).values(payload)
        stmt = stmt.on_conflict_do_update(
            index_elements=[
                Fundamental.ticker,
                Fundamental.statement,
                Fundamental.period,
                Fundamental.as_of,
                Fundamental.metric,
            ],
            set_={
                "value": stmt.excluded.value,
                "raw": stmt.excluded.raw,
            },
        )
        with self.session() as session:
            session.execute(stmt)
        return len(payload)

    def upsert_news(self, records: Iterable[dict[str, Any]]) -> int:
        payload = []
        for record in records:
            payload.append(
                {
                    "ticker": record["Ticker"],
                    "published_at": self._coerce_datetime(record["PublishedAt"]),
                    "title": record.get("Title"),
                    "summary": record.get("Summary"),
                    "url": record.get("Url"),
                    "source": record.get("Source"),
                }
            )
        if not payload:
            return 0
        stmt = insert(NewsArticle).values(payload)
        stmt = stmt.on_conflict_do_update(
            index_elements=[NewsArticle.ticker, NewsArticle.published_at, NewsArticle.title],
            set_={
                "summary": stmt.excluded.summary,
                "url": stmt.excluded.url,
                "source": stmt.excluded.source,
            },
        )
        with self.session() as session:
            session.execute(stmt)
        return len(payload)

    def upsert_corporate_events(self, records: Iterable[dict[str, Any]]) -> int:
        payload = []
        for record in records:
            payload.append(
                {
                    "ticker": record["Ticker"],
                    "event_type": record["EventType"],
                    "event_date": self._coerce_date(record["EventDate"]),
                    "reference": record.get("Reference", "general"),
                    "value": self._coerce_float(record.get("Value")),
                    "currency": record.get("Currency"),
                    "details": self._dump_json(record.get("Details")),
                    "source": record.get("Source"),
                }
            )
        if not payload:
            return 0
        stmt = insert(CorporateEvent).values(payload)
        stmt = stmt.on_conflict_do_update(
            index_elements=[
                CorporateEvent.ticker,
                CorporateEvent.event_type,
                CorporateEvent.event_date,
                CorporateEvent.reference,
            ],
            set_={
                "value": stmt.excluded.value,
                "currency": stmt.excluded.currency,
                "details": stmt.excluded.details,
                "source": stmt.excluded.source,
            },
        )
        with self.session() as session:
            session.execute(stmt)
        return len(payload)

    def upsert_option_surface(self, records: Iterable[dict[str, Any]]) -> int:
        payload = []
        for record in records:
            strike_value = self._coerce_float(record.get("Strike"))
            if strike_value is None:
                LOGGER.debug("Skipping option surface record with missing strike: %s", record)
                continue
            payload.append(
                {
                    "ticker": record["Ticker"],
                    "as_of": self._coerce_datetime(record["AsOf"]),
                    "expiration": self._coerce_date(record["Expiration"]),
                    "strike": strike_value,
                    "option_type": record["OptionType"],
                    "metric": record["Metric"],
                    "value": self._coerce_float(record.get("Value")),
                    "source": record.get("Source"),
                    "extra": self._dump_json(record.get("Extra")),
                }
            )
        if not payload:
            return 0
        stmt = insert(OptionSurfacePoint).values(payload)
        stmt = stmt.on_conflict_do_update(
            index_elements=[
                OptionSurfacePoint.ticker,
                OptionSurfacePoint.as_of,
                OptionSurfacePoint.expiration,
                OptionSurfacePoint.strike,
                OptionSurfacePoint.option_type,
                OptionSurfacePoint.metric,
            ],
            set_={
                "value": stmt.excluded.value,
                "source": stmt.excluded.source,
                "extra": stmt.excluded.extra,
            },
        )
        with self.session() as session:
            session.execute(stmt)
        return len(payload)

    def upsert_sentiment_signals(self, records: Iterable[dict[str, Any]]) -> int:
        payload = []
        for record in records:
            payload.append(
                {
                    "ticker": record["Ticker"],
                    "as_of": self._coerce_datetime(record["AsOf"]),
                    "provider": record["Provider"],
                    "signal_type": record["SignalType"],
                    "score": self._coerce_float(record.get("Score")),
                    "magnitude": self._coerce_float(record.get("Magnitude")),
                    "payload": self._dump_json(record.get("Payload")),
                }
            )
        if not payload:
            return 0
        stmt = insert(SentimentSignal).values(payload)
        stmt = stmt.on_conflict_do_update(
            index_elements=[
                SentimentSignal.ticker,
                SentimentSignal.as_of,
                SentimentSignal.provider,
                SentimentSignal.signal_type,
            ],
            set_={
                "score": stmt.excluded.score,
                "magnitude": stmt.excluded.magnitude,
                "payload": stmt.excluded.payload,
            },
        )
        with self.session() as session:
            session.execute(stmt)
        return len(payload)

    def upsert_esg_metrics(self, records: Iterable[dict[str, Any]]) -> int:
        payload = []
        for record in records:
            payload.append(
                {
                    "ticker": record["Ticker"],
                    "as_of": self._coerce_date(record["AsOf"]),
                    "provider": record["Provider"],
                    "metric": record["Metric"],
                    "value": self._coerce_float(record.get("Value")),
                    "raw": self._dump_json(record.get("Raw")),
                }
            )
        if not payload:
            return 0
        stmt = insert(ESGMetricEntry).values(payload)
        stmt = stmt.on_conflict_do_update(
            index_elements=[
                ESGMetricEntry.ticker,
                ESGMetricEntry.as_of,
                ESGMetricEntry.provider,
                ESGMetricEntry.metric,
            ],
            set_={
                "value": stmt.excluded.value,
                "raw": stmt.excluded.raw,
            },
        )
        with self.session() as session:
            session.execute(stmt)
        return len(payload)

    def upsert_ownership_flows(self, records: Iterable[dict[str, Any]]) -> int:
        payload = []
        for record in records:
            payload.append(
                {
                    "ticker": record["Ticker"],
                    "as_of": self._coerce_date(record["AsOf"]),
                    "holder": record["Holder"],
                    "holder_type": record.get("HolderType", "unknown"),
                    "metric": record["Metric"],
                    "value": self._coerce_float(record.get("Value")),
                    "raw": self._dump_json(record.get("Raw")),
                    "source": record.get("Source"),
                }
            )
        if not payload:
            return 0
        stmt = insert(OwnershipFlow).values(payload)
        stmt = stmt.on_conflict_do_update(
            index_elements=[
                OwnershipFlow.ticker,
                OwnershipFlow.as_of,
                OwnershipFlow.holder,
                OwnershipFlow.metric,
            ],
            set_={
                "holder_type": stmt.excluded.holder_type,
                "value": stmt.excluded.value,
                "raw": stmt.excluded.raw,
                "source": stmt.excluded.source,
            },
        )
        with self.session() as session:
            session.execute(stmt)
        return len(payload)

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------
    def get_prices(
        self,
        ticker: str,
        interval: str,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        stmt = select(Price).where(Price.ticker == ticker, Price.interval == interval)
        if start:
            stmt = stmt.where(Price.date >= start)
        if end:
            stmt = stmt.where(Price.date <= end)
        stmt = stmt.order_by(Price.date.asc())
        with self.session() as session:
            rows = session.execute(stmt).scalars().all()
            data = [row.to_frame_dict() for row in rows]
        if not data:
            LOGGER.debug(
                "No price records found for ticker=%s interval=%s in requested range.",
                ticker,
                interval,
            )
            return pd.DataFrame()
        frame = pd.DataFrame(data)
        return frame

    def get_latest_price_date(self, ticker: str, interval: str) -> date | None:
        stmt = (
            select(Price.date)
            .where(Price.ticker == ticker, Price.interval == interval)
            .order_by(Price.date.desc())
            .limit(1)
        )
        with self.session() as session:
            latest = session.execute(stmt).scalar_one_or_none()
        return latest

    def get_indicators(
        self,
        ticker: str,
        interval: str,
        category: str | None = None,
    ) -> pd.DataFrame:
        stmt = select(Indicator).where(Indicator.ticker == ticker, Indicator.interval == interval)
        if category:
            stmt = stmt.where(Indicator.category == category)
        stmt = stmt.order_by(Indicator.date.asc())
        with self.session() as session:
            rows = session.execute(stmt).scalars().all()
            data = [row.to_frame_dict() for row in rows]
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)

    def get_fundamentals(self, ticker: str) -> pd.DataFrame:
        stmt = select(Fundamental).where(Fundamental.ticker == ticker)
        with self.session() as session:
            rows = session.execute(stmt).scalars().all()
            data = [row.to_frame_dict() for row in rows]
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)

    def get_news(self, ticker: str) -> pd.DataFrame:
        stmt = select(NewsArticle).where(NewsArticle.ticker == ticker).order_by(
            NewsArticle.published_at.desc()
        )
        with self.session() as session:
            rows = session.execute(stmt).scalars().all()
            data = [row.to_frame_dict() for row in rows]
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)

    def get_corporate_events(
        self,
        ticker: str,
        event_type: str | None = None,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        stmt = select(CorporateEvent).where(CorporateEvent.ticker == ticker)
        if event_type:
            stmt = stmt.where(CorporateEvent.event_type == event_type)
        if start:
            stmt = stmt.where(CorporateEvent.event_date >= start)
        if end:
            stmt = stmt.where(CorporateEvent.event_date <= end)
        stmt = stmt.order_by(CorporateEvent.event_date.desc())
        with self.session() as session:
            rows = session.execute(stmt).scalars().all()
            data = [row.to_frame_dict() for row in rows]
        return pd.DataFrame(data)

    def get_option_surface(
        self,
        ticker: str,
        as_of: datetime | None = None,
        expiration: date | None = None,
    ) -> pd.DataFrame:
        stmt = select(OptionSurfacePoint).where(OptionSurfacePoint.ticker == ticker)
        if as_of:
            stmt = stmt.where(OptionSurfacePoint.as_of >= as_of)
        if expiration:
            stmt = stmt.where(OptionSurfacePoint.expiration == expiration)
        stmt = stmt.order_by(
            OptionSurfacePoint.as_of.desc(),
            OptionSurfacePoint.expiration.asc(),
            OptionSurfacePoint.strike.asc(),
        )
        with self.session() as session:
            rows = session.execute(stmt).scalars().all()
            data = [row.to_frame_dict() for row in rows]
        return pd.DataFrame(data)

    def get_sentiment_signals(
        self,
        ticker: str,
        provider: str | None = None,
        signal_type: str | None = None,
    ) -> pd.DataFrame:
        stmt = select(SentimentSignal).where(SentimentSignal.ticker == ticker)
        if provider:
            stmt = stmt.where(SentimentSignal.provider == provider)
        if signal_type:
            stmt = stmt.where(SentimentSignal.signal_type == signal_type)
        stmt = stmt.order_by(SentimentSignal.as_of.desc())
        with self.session() as session:
            rows = session.execute(stmt).scalars().all()
            data = [row.to_frame_dict() for row in rows]
        return pd.DataFrame(data)

    def get_esg_metrics(
        self,
        ticker: str,
        provider: str | None = None,
        metric: str | None = None,
    ) -> pd.DataFrame:
        stmt = select(ESGMetricEntry).where(ESGMetricEntry.ticker == ticker)
        if provider:
            stmt = stmt.where(ESGMetricEntry.provider == provider)
        if metric:
            stmt = stmt.where(ESGMetricEntry.metric == metric)
        stmt = stmt.order_by(ESGMetricEntry.as_of.desc())
        with self.session() as session:
            rows = session.execute(stmt).scalars().all()
            data = [row.to_frame_dict() for row in rows]
        return pd.DataFrame(data)

    def upsert_research_artifact(self, record: dict[str, Any]) -> dict[str, Any]:
        payload = dict(record)
        payload.setdefault("source", "crawler")
        payload.setdefault("captured_at", datetime.utcnow())
        metadata_payload = payload.pop("metadata", None)
        payload["metadata_json"] = self._dump_json(metadata_payload)
        with self.session() as session:
            existing = (
                session.execute(
                    select(ResearchArtifact).where(
                        ResearchArtifact.url == payload["url"],
                        ResearchArtifact.content_hash == payload["content_hash"],
                    )
                )
                .scalars()
                .one_or_none()
            )
            if existing is None:
                entry = ResearchArtifact(**payload)
                session.add(entry)
                session.flush()
                session.refresh(entry)
                return entry.to_dict()

            existing.captured_at = payload["captured_at"]
            existing.source = payload.get("source", existing.source)
            existing.raw_content = payload.get("raw_content", existing.raw_content)
            existing.extractive_summary = payload.get("extractive_summary")
            existing.abstractive_summary = payload.get("abstractive_summary")
            existing.sentiment_label = payload.get("sentiment_label")
            existing.sentiment_score = payload.get("sentiment_score")
            existing.metadata_json = payload.get("metadata_json")
            session.add(existing)
            session.flush()
            session.refresh(existing)
            return existing.to_dict()

    def get_research_artifacts(self, limit: int = 50) -> list[dict[str, Any]]:
        if limit <= 0:
            raise ValueError("limit must be a positive integer")
        stmt = (
            select(ResearchArtifact)
            .order_by(ResearchArtifact.captured_at.desc())
            .limit(int(limit))
        )
        with self.session() as session:
            rows = session.execute(stmt).scalars().all()
            return [row.to_dict() for row in rows]

    def get_ownership_flows(
        self,
        ticker: str,
        holder_type: str | None = None,
        metric: str | None = None,
    ) -> pd.DataFrame:
        stmt = select(OwnershipFlow).where(OwnershipFlow.ticker == ticker)
        if holder_type:
            stmt = stmt.where(OwnershipFlow.holder_type == holder_type)
        if metric:
            stmt = stmt.where(OwnershipFlow.metric == metric)
        stmt = stmt.order_by(OwnershipFlow.as_of.desc())
        with self.session() as session:
            rows = session.execute(stmt).scalars().all()
            data = [row.to_frame_dict() for row in rows]
        return pd.DataFrame(data)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def set_refresh_timestamp(
        self,
        ticker: str,
        interval: str,
        category: str,
        timestamp: datetime | None = None,
    ) -> None:
        timestamp = timestamp or datetime.utcnow()
        key = self._refresh_key(ticker, interval, category)
        self.set_meta(key, timestamp.isoformat())

    def get_refresh_timestamp(self, ticker: str, interval: str, category: str) -> datetime | None:
        key = self._refresh_key(ticker, interval, category)
        raw = self.get_meta(key)
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            LOGGER.debug("Unable to parse refresh timestamp for key %s", key)
            return None

    @staticmethod
    def _refresh_key(ticker: str, interval: str, category: str) -> str:
        return f"{ticker}:{interval}:{category}:last_refresh"

    @staticmethod
    def _coerce_date(value: Any) -> date:
        if value is None:
            raise ValueError("Date value cannot be None when storing records.")
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        dt_value = pd.to_datetime(value, errors="coerce")
        if pd.isna(dt_value):
            raise ValueError(f"Unable to parse date value: {value!r}")
        return dt_value.date()

    @staticmethod
    def _coerce_datetime(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        dt_value = pd.to_datetime(value, errors="coerce")
        if pd.isna(dt_value):
            raise ValueError(f"Unable to parse datetime value: {value!r}")
        return dt_value.to_pydatetime()

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _dump_json(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, default=str)
        except (TypeError, ValueError):
            return str(value)


@dataclass(slots=True)
class ExperimentTracker:
    """Persist experiment metadata to the configured database."""

    config: "PredictorConfig"
    database: Database | None = None

    def __post_init__(self) -> None:
        if self.database is None:
            self.database = Database(self.config.database_url)

    def log_run(
        self,
        target: str,
        run_type: str,
        parameters: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        if self.database is None:  # pragma: no cover - defensive guard
            raise RuntimeError("ExperimentTracker is not initialised with a database instance.")

        record = ExperimentLog(
            ticker=self.config.ticker,
            target=target,
            run_type=run_type,
            parameters=self._dump_json(parameters),
            metrics=self._dump_json(metrics),
            context=self._dump_json(context),
        )
        with self.database.session() as session:
            session.add(record)

    @staticmethod
    def _dump_json(payload: dict[str, Any] | None) -> str | None:
        if payload is None:
            return None
        return json.dumps(payload, default=str)


__all__ = [
    "Database",
    "MetaEntry",
    "Price",
    "Indicator",
    "Fundamental",
    "NewsArticle",
    "CorporateEvent",
    "OptionSurfacePoint",
    "SentimentSignal",
    "ESGMetricEntry",
    "OwnershipFlow",
    "ResearchArtifact",
    "ExperimentLog",
    "ExperimentTracker",
]

