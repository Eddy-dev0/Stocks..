"""Example demonstrating safe batched UPSERTs with SQLAlchemy 2.x.

The typical error displayed in the question occurs when a single INSERT ..
ON CONFLICT statement binds more parameters than the database allows.  For
instance, SQLite only allows 999 host parameters per statement and PostgreSQL's
`max_stack_depth` / `max_expr_depth` settings impose similar limits.  When we
try to send thousands of rows at once (each row containing seven columns),
SQLAlchemy renders one gigantic statement with far more bound values than the
server will accept, raising ``sqlalchemy.exc.ProgrammingError`` with the
``Too many SQL variables``/``too many bind parameters`` message.

The :func:`bulk_upsert_metrics` helper below automatically chunks the incoming
rows so that each INSERT stays within the limit reported by the active
SQLAlchemy dialect (falling back to a conservative default when the dialect
does not specify a limit).  The helper retains the ``ON CONFLICT .. DO UPDATE``
logic so duplicates are updated with the latest ``category``, ``value`` and
``extra`` fields.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date
from itertools import islice
from typing import Iterable, Iterator, Mapping, MutableMapping, Sequence

from sqlalchemy import Date, Float, PrimaryKeyConstraint, String, Text, create_engine
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class Base(DeclarativeBase):
    """Declarative base for the demo model."""


class Metric(Base):
    """Technical indicator or metric associated with a stock ticker."""

    __tablename__ = "metrics"

    ticker: Mapped[str] = mapped_column(String(32), primary_key=True)
    interval: Mapped[str] = mapped_column(String(16), primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    category: Mapped[str] = mapped_column(String(32))
    value: Mapped[float] = mapped_column(Float(asdecimal=False))
    extra: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        PrimaryKeyConstraint("ticker", "interval", "date", "name", name="pk_metrics"),
    )


@dataclass(slots=True)
class MetricRow:
    """Convenience container for incoming metric rows."""

    ticker: str
    interval: str
    date: date
    name: str
    category: str
    value: float
    extra: str | None = None

    def asdict(self) -> MutableMapping[str, object]:
        return asdict(self)


def _chunked(iterable: Iterable[Mapping[str, object]], size: int) -> Iterator[list[Mapping[str, object]]]:
    """Yield ``size``-sized lists from *iterable* until it is exhausted."""

    iterator = iter(iterable)
    while True:
        batch = list(islice(iterator, size))
        if not batch:
            return
        yield batch


def _determine_chunk_size(session: Session, columns_per_row: int, requested_chunk_size: int | None) -> int:
    """Determine an INSERT batch size that respects the dialect's parameter limit."""

    if requested_chunk_size is not None and requested_chunk_size > 0:
        return requested_chunk_size

    bind = session.get_bind()
    max_params = getattr(bind.dialect, "max_parameters", None)
    if max_params is None or max_params <= 0:
        # Fallback to a conservative default that works for SQLite (999 params).
        max_params = 900

    chunk_size = max_params // columns_per_row
    return max(chunk_size, 1)


def bulk_upsert_metrics(
    session: Session,
    rows: Sequence[Mapping[str, object]] | Iterable[MetricRow],
    *,
    chunk_size: int | None = None,
) -> None:
    """Insert or update *rows* in batches using PostgreSQL's ON CONFLICT.

    Parameters
    ----------
    session:
        An active :class:`~sqlalchemy.orm.Session` bound to a PostgreSQL engine.
    rows:
        Either dictionaries compatible with the :class:`Metric` columns or
        :class:`MetricRow` instances.
    chunk_size:
        Optional manual override to force a particular batch size.
    """

    # Normalise input rows into dictionaries understood by ``session.execute``.
    if isinstance(rows, Sequence):
        normalized_rows: list[Mapping[str, object]] = [r.asdict() if isinstance(r, MetricRow) else r for r in rows]
    else:
        normalized_rows = [r.asdict() if isinstance(r, MetricRow) else r for r in rows]

    if not normalized_rows:
        return

    columns_per_row = len(normalized_rows[0])
    effective_chunk_size = _determine_chunk_size(session, columns_per_row, chunk_size)

    insert_stmt = insert(Metric)
    update_mapping = {
        "category": insert_stmt.excluded.category,
        "value": insert_stmt.excluded.value,
        "extra": insert_stmt.excluded.extra,
    }

    for batch in _chunked(normalized_rows, effective_chunk_size):
        session.execute(
            insert_stmt.on_conflict_do_update(
                index_elements=[Metric.ticker, Metric.interval, Metric.date, Metric.name],
                set_=update_mapping,
            ),
            batch,
        )

    session.commit()


def main() -> None:
    """Run the demo inserting a large set of rows in a safe, batched manner."""

    # Replace the connection string with the real PostgreSQL DSN for production use.
    engine = create_engine("postgresql+psycopg://username:password@localhost/mydatabase", echo=True)

    Base.metadata.create_all(engine)

    sample_rows = [
        MetricRow("NVDA", "1d", date(2024, 11, 14), "Return_1d", "technical", -0.032570175512470656),
        MetricRow("NVDA", "1d", date(2024, 11, 15), "Return_1d", "technical", -0.012340000000000001),
        MetricRow("NVDA", "1d", date(2024, 11, 18), "Return_1d", "technical", 0.0854, extra="gap-up"),
    ]

    with Session(engine) as session:
        bulk_upsert_metrics(session, sample_rows)


if __name__ == "__main__":
    main()
