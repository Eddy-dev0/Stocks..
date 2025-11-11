"""Database utilities for persisting market data."""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

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


@dataclass(slots=True)
class Database:
    """High level database helper built on top of SQLAlchemy."""

    url: str

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
        with self.session() as session:
            result = session.execute(select(MetaEntry).where(MetaEntry.key == key)).scalar_one_or_none()
        if not result:
            return None
        value = result.value
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return value

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
        stmt = insert(Price).values(records)
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
        with self.session() as session:
            session.execute(stmt)
        return len(records)

    def upsert_indicators(
        self,
        ticker: str,
        interval: str,
        records: Iterable[dict[str, Any]],
    ) -> int:
        payload = []
        for record in records:
            payload.append(
                {
                    "ticker": ticker,
                    "interval": interval,
                    "date": self._coerce_date(record.get("Date")),
                    "name": record.get("Indicator") or record.get("name"),
                    "category": record.get("Category", "technical"),
                    "value": self._coerce_float(record.get("Value")),
                    "extra": json.dumps(record.get("Extra")) if record.get("Extra") is not None else None,
                }
            )
        if not payload:
            return 0
        stmt = insert(Indicator).values(payload)
        stmt = stmt.on_conflict_do_update(
            index_elements=[Indicator.ticker, Indicator.interval, Indicator.date, Indicator.name],
            set_={
                "category": stmt.excluded.category,
                "value": stmt.excluded.value,
                "extra": stmt.excluded.extra,
            },
        )
        with self.session() as session:
            session.execute(stmt)
        return len(payload)

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
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame([row.to_frame_dict() for row in rows])
        return frame

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
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([row.to_frame_dict() for row in rows])

    def get_fundamentals(self, ticker: str) -> pd.DataFrame:
        stmt = select(Fundamental).where(Fundamental.ticker == ticker)
        with self.session() as session:
            rows = session.execute(stmt).scalars().all()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([row.to_frame_dict() for row in rows])

    def get_news(self, ticker: str) -> pd.DataFrame:
        stmt = select(NewsArticle).where(NewsArticle.ticker == ticker).order_by(
            NewsArticle.published_at.desc()
        )
        with self.session() as session:
            rows = session.execute(stmt).scalars().all()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([row.to_frame_dict() for row in rows])

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


__all__ = [
    "Database",
    "MetaEntry",
    "Price",
    "Indicator",
    "Fundamental",
    "NewsArticle",
]

