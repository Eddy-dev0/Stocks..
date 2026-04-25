"""Pattern screener tab for the Streamlit frontend."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Callable, Dict

import pandas as pd
import streamlit as st

from stock_predictor.screener.market_data.provider import FrontendAPIMarketDataProvider
from stock_predictor.screener.pattern_engine.types import PatternType
from stock_predictor.screener.services.screener_service import ScreenerFilters, ScreenerService

PATTERN_CHOICES: tuple[PatternType, ...] = (
    "Double Bottom",
    "Double Top",
    "Triple Bottom",
    "Triple Top",
    "Head and Shoulders",
    "Inverted Head and Shoulders",
    "Ascending Triangle",
    "Descending Triangle",
    "Pennant",
    "Flag",
    "Bearish Flag",
    "Channel",
    "Channel Up",
    "Channel Down",
    "Cup and Handle",
    "Diamond",
)


def _market_label(value: str) -> str:
    return {"all": "Alle", "stock": "Aktien", "future": "Futures"}.get(value, value)


def _render_table(rows: list[dict[str, object]]) -> pd.DataFrame:
    table_rows = []
    for row in rows:
        tq = row["tradeQuality"]
        table_rows.append(
            {
                "Aktienname / Future-Name": row["name"],
                "Symbol": row["symbol"],
                "Chartpattern": row["patternType"],
                "Status": row["status"],
                "Trade-Qualität": f"{tq['rating']} {tq['successes']}/{tq['occurrences']}",
                "Ø Move danach": f"{tq['averageMovePercent']:+.2f}%",
                "Median Move": f"{tq['medianMovePercent']:+.2f}%",
                "Treffer historisch": tq["occurrences"],
                "Letztes Signal": row["detectedAt"],
                "Volumen": int(row["volume"]),
                "Score": row["score"],
                "Aktion": f"Chart öffnen ({row['symbol']})",
            }
        )
    return pd.DataFrame(table_rows)


def render_screener(
    request_fn: Callable[..., Dict[str, Any] | None],
    *,
    default_end_date: date,
) -> None:
    st.subheader("Screener")
    st.caption("1h-Markt-Scanner für Aktien/Futures. Statistische Setups, keine Garantie auf zukünftige Gewinne.")

    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        pattern = st.selectbox("Pattern auswählen", options=list(PATTERN_CHOICES), index=0)
    with col2:
        market = st.selectbox("Markt", options=["all", "stock", "future"], format_func=_market_label)
    with col3:
        min_score = st.slider("Mindest-Trade-Qualität (Score)", min_value=40, max_value=95, value=60)
    with col4:
        min_occ = st.slider("Minimum Sample Size", min_value=10, max_value=120, value=20)

    fcol1, fcol2, fcol3, fcol4 = st.columns(4)
    with fcol1:
        min_volume = st.number_input("Mindestvolumen", min_value=0.0, value=0.0, step=1000.0)
    with fcol2:
        status_filter = st.selectbox("Pattern-Status", options=["all", "forming", "confirmed", "failed", "expired"])
    with fcol3:
        lookback_days = st.slider("Lookback Tage", 90, 730, 365, step=30)
    with fcol4:
        sort_by = st.selectbox("Sortierung", options=["Trade-Qualität", "Trefferanzahl", "Volumen", "Aktualität"])

    provider = FrontendAPIMarketDataProvider(request_fn=request_fn)
    service = ScreenerService(provider)

    st.info("Live-Provider: UI API /data Endpoint. Wenn kein externer Anbieter konfiguriert ist, sind Ergebnisse ggf. eingeschränkt.")

    if not st.button("Scan starten", type="primary"):
        return

    with st.spinner("Scanning 1h market data..."):
        rows = service.scan_market(
            pattern,
            market,
            start_date=default_end_date - timedelta(days=lookback_days),
            end_date=default_end_date,
            filters=ScreenerFilters(
                min_score=float(min_score),
                min_occurrences=int(min_occ),
                min_volume=float(min_volume),
                status=status_filter,
            ),
        )

    if not rows:
        st.warning("Keine Treffer für den gewählten Filter gefunden.")
        return

    if sort_by == "Volumen":
        rows.sort(key=lambda x: x["volume"], reverse=True)
    elif sort_by == "Trefferanzahl":
        rows.sort(key=lambda x: x["tradeQuality"]["occurrences"], reverse=True)
    elif sort_by == "Aktualität":
        rows.sort(key=lambda x: x["detectedAt"], reverse=True)

    table = _render_table(rows)
    st.caption(f"Last updated: {rows[0]['lastUpdated']} UTC")
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption("Hinweis: Confirmed-Setups werden intern priorisiert. Walk-Forward-Backtesting aktiv, sofern ausreichende Historie vorhanden ist.")

    csv = table.to_csv(index=False).encode("utf-8")
    st.download_button("CSV Export", data=csv, file_name="screener_results.csv", mime="text/csv")
