"""Streamlit dashboard for exploring Stock Predictor outputs."""

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from typing import Any, Dict, List

import altair as alt
import pandas as pd
import requests
import streamlit as st

DEFAULT_API_URL = os.getenv("STOCK_PREDICTOR_API_URL", "http://localhost:8000")
DEFAULT_TICKER = os.getenv("STOCK_PREDICTOR_DEFAULT_TICKER", "AAPL")
DEFAULT_API_KEY = os.getenv("STOCK_PREDICTOR_UI_API_KEY", "")

st.set_page_config(page_title="Stock Predictor Dashboard", layout="wide")

for key, value in {
    "api_base": DEFAULT_API_URL,
    "api_key": DEFAULT_API_KEY,
    "data_response": None,
    "forecast_response": None,
    "backtest_response": None,
    "research_response": None,
}.items():
    st.session_state.setdefault(key, value)


def _parse_targets(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _request(
    path: str,
    *,
    method: str = "GET",
    params: Dict[str, Any] | None = None,
    json_payload: Dict[str, Any] | None = None,
) -> Dict[str, Any] | None:
    base_url = (st.session_state.get("api_base") or DEFAULT_API_URL).rstrip("/")
    api_key = st.session_state.get("api_key") or ""
    url = f"{base_url}{path}"
    headers: Dict[str, str] = {}
    if api_key:
        headers["X-API-Key"] = api_key

    try:
        response = requests.request(
            method,
            url,
            params=params,
            json=json_payload,
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network error path
        detail = getattr(getattr(exc, "response", None), "text", str(exc))
        st.error(f"Request to {url} failed: {detail}")
        return None
    return response.json()


def _coerce_dataframe(payload: Any) -> pd.DataFrame | None:
    if payload is None:
        return None
    if isinstance(payload, pd.DataFrame):
        return payload
    if isinstance(payload, dict):
        if "prices" in payload:
            return _coerce_dataframe(payload["prices"])
        for key in ("data", "values", "records", "download_data"):
            if key in payload:
                frame = _coerce_dataframe(payload[key])
                if frame is not None and not frame.empty:
                    return frame
        if payload and all(isinstance(value, (list, tuple)) for value in payload.values()):
            try:
                frame = pd.DataFrame(payload)
                if not frame.empty:
                    return frame
            except ValueError:
                pass
        if payload and all(not isinstance(value, (dict, list, tuple)) for value in payload.values()):
            return pd.DataFrame([payload])
    if isinstance(payload, list):
        if payload and isinstance(payload[0], dict):
            return pd.DataFrame(payload)
        if payload:
            return pd.DataFrame({"value": payload})
    return None


def _normalise_timeseries(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in ("date", "datetime", "timestamp"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
            frame = frame.set_index(column)
            break
    frame = frame.sort_index()
    return frame


def _extract_forecast_low(payload: Any) -> float | None:
    frame = _coerce_dataframe(payload)
    if frame is not None and not frame.empty:
        numeric_cols = frame.select_dtypes("number").columns
        for candidate in ("close", "price", "prediction", "yhat"):
            if candidate in frame.columns:
                return float(frame[candidate].min())
        if not numeric_cols.empty:
            return float(frame[numeric_cols].min().min())
    if isinstance(payload, dict):
        for key in ("forecasts", "forecast", "predictions", "values"):
            if key in payload:
                return _extract_forecast_low(payload[key])
    if isinstance(payload, list):
        for item in payload:
            nested = _extract_forecast_low(item)
            if nested is not None:
                return nested
    return None


def _derive_buy_zone(frame: pd.DataFrame, forecast_low: float | None) -> Dict[str, Any] | None:
    if "close" not in frame.columns:
        return None

    working = frame.copy()
    close = working["close"].astype(float)

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    working["RSI14"] = 100 - (100 / (1 + rs))

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    working["MACD"] = ema_12 - ema_26
    working["MACD_signal"] = working["MACD"].ewm(span=9, adjust=False).mean()
    working["MACD_cross_up"] = (working["MACD"] > working["MACD_signal"]) & (
        working["MACD"].shift(1) <= working["MACD_signal"].shift(1)
    )

    working["BB_mid"] = close.rolling(window=20, min_periods=20).mean()
    rolling_std = close.rolling(window=20, min_periods=20).std()
    working["BB_lower"] = working["BB_mid"] - 2 * rolling_std
    working["BB_upper"] = working["BB_mid"] + 2 * rolling_std

    conditions = (
        (working["RSI14"] < 30)
        & working["MACD_cross_up"].fillna(False)
        & (close <= working["BB_lower"])
    )
    candidates = working[conditions]
    if candidates.empty:
        return None

    base_low = float(candidates["close"].min())
    price_low = min(base_low, forecast_low) if forecast_low is not None else base_low
    price_high = max(float(candidates["close"].max()), price_low * 1.02)
    return {
        "start": candidates.index.min(),
        "end": candidates.index.max(),
        "price_low": price_low,
        "price_high": price_high,
        "reason": (
            "RSI < 30 with bullish MACD crossover and lower Bollinger-band touch"
            "; anchored near forecasted lows" if forecast_low is not None else "RSI < 30 with bullish MACD crossover and lower Bollinger-band touch"
        ),
    }


def _build_price_chart(
    frame: pd.DataFrame,
    *,
    buy_zone: Dict[str, Any] | None = None,
    show_buy_zone: bool = True,
) -> alt.Chart:
    time_index = frame.index.name or "timestamp"
    data_frame = frame.reset_index().rename(columns={frame.index.name or "index": "time"})
    value_vars = ["close"]
    for candidate in ("MA20", "MA50"):
        if candidate in data_frame.columns:
            value_vars.append(candidate)
    melted = data_frame.melt("time", value_vars=value_vars, var_name="series", value_name="value")

    line_chart = (
        alt.Chart(melted)
        .mark_line()
        .encode(
            x=alt.X("time:T", title=time_index.capitalize()),
            y=alt.Y("value:Q", title="Price"),
            color=alt.Color("series:N", title="Series"),
            tooltip=["time:T", "series:N", "value:Q"],
        )
    )

    overlays: List[alt.Chart] = [line_chart]
    if show_buy_zone and buy_zone:
        rect_frame = pd.DataFrame([buy_zone])
        overlays.append(
            alt.Chart(rect_frame)
            .mark_rect(opacity=0.15, color="steelblue")
            .encode(
                x=alt.X("start:T", title=time_index.capitalize()),
                x2="end:T",
                y="price_low:Q",
                y2="price_high:Q",
                tooltip=["reason"],
            )
        )

    return alt.layer(*overlays).resolve_scale(color="independent")


def _download_button(label: str, data: Any, file_name: str, mime: str) -> None:
    if data is None:
        return
    st.download_button(label, data=data, file_name=file_name, mime=mime)


with st.sidebar:
    st.header("Configuration")
    st.text_input("API base URL", key="api_base")
    st.text_input("API key", key="api_key")

    st.header("Query Parameters")
    ticker_input = st.text_input("Ticker symbol", value=DEFAULT_TICKER)
    ticker = (ticker_input or DEFAULT_TICKER).upper().strip()
    start_default = date.today() - timedelta(days=365)
    start_date = st.date_input("Start date", value=start_default)
    end_date = st.date_input("End date", value=date.today())
    interval = st.selectbox("Interval", options=["1d", "1wk", "1mo"], index=0)
    targets_raw = st.text_input("Prediction targets", help="Comma separated list of targets")
    force_refresh_market = st.checkbox("Refresh market data on fetch", value=False)
    refresh_before_forecast = st.checkbox("Refresh data before forecasting", value=False)
    use_custom_horizon = st.checkbox("Specify forecast horizon", value=False)
    horizon_value = None
    if use_custom_horizon:
        horizon_value = st.number_input("Forecast horizon", min_value=1, max_value=365, value=5, step=1)

st.title("Stock Predictor Dashboard")
st.caption("Explore model forecasts, historical indicators, and curated research notes.")

col_data, col_forecast = st.columns(2)

with col_data:
    st.subheader("Market Data & Indicators")
    if st.button("Fetch market data", type="primary"):
        with st.spinner("Loading market data..."):
            response = _request(
                f"/data/{ticker}",
                params={
                    "refresh": bool(force_refresh_market),
                    "start_date": start_date.isoformat() if isinstance(start_date, date) else None,
                    "end_date": end_date.isoformat() if isinstance(end_date, date) else None,
                    "interval": interval,
                },
            )
            if response is not None:
                st.session_state["data_response"] = response
                st.success("Market data loaded")

    data_response = st.session_state.get("data_response") or {}
    payload = data_response.get("data") if isinstance(data_response, dict) else data_response
    frame = _coerce_dataframe(payload)
    if frame is not None and not frame.empty:
        frame = _normalise_timeseries(frame)
        forecast_low = _extract_forecast_low(st.session_state.get("forecast_response"))
        buy_zone = _derive_buy_zone(frame, forecast_low)
        if "close" in frame.columns and "close" in frame.select_dtypes("number").columns:
            frame["MA20"] = frame["close"].rolling(window=20, min_periods=1).mean()
            frame["MA50"] = frame["close"].rolling(window=50, min_periods=1).mean()
        show_buy_zone = st.checkbox(
            "Show buy-zone overlay",
            value=bool(buy_zone),
            help="Highlight windows where RSI < 30, MACD crosses up, and price hugs the lower Bollinger band (anchored near forecast lows when available).",
        )
        st.altair_chart(
            _build_price_chart(frame, buy_zone=buy_zone, show_buy_zone=show_buy_zone),
            use_container_width=True,
        )
        with st.expander("How is the buy-zone derived?"):
            st.markdown(
                "The shaded region captures sessions where:\n"
                "- 14-day RSI dips below 30 (oversold)\n"
                "- The MACD line crosses above its signal (bullish momentum shift)\n"
                "- Closing price touches/under-cuts the 20-day lower Bollinger band\n"
                "If forecast data is present, the price floor leans on forecasted lows."
            )
            if buy_zone:
                st.write(
                    "Start:",
                    buy_zone["start"],
                    "| End:",
                    buy_zone["end"],
                    "| Price window:",
                    f"${buy_zone['price_low']:.2f} - ${buy_zone['price_high']:.2f}",
                )
            else:
                st.caption("No qualifying buy-zone window detected for the current view.")
        st.dataframe(frame.tail(20), use_container_width=True)
        _download_button(
            "Download data (CSV)",
            frame.to_csv().encode("utf-8"),
            file_name=f"{ticker}_data.csv",
            mime="text/csv",
        )
        _download_button(
            "Download data (JSON)",
            json.dumps(payload, indent=2, default=str).encode("utf-8"),
            file_name=f"{ticker}_data.json",
            mime="application/json",
        )
    elif data_response:
        st.info("No tabular data available for the selected configuration.")

with col_forecast:
    st.subheader("Forecasts & Backtests")
    if st.button("Run forecast"):
        with st.spinner("Generating forecasts..."):
            response = _request(
                f"/forecasts/{ticker}",
                method="POST",
                json_payload={
                    "targets": _parse_targets(targets_raw),
                    "refresh": bool(refresh_before_forecast),
                    "horizon": horizon_value,
                },
            )
            if response is not None:
                st.session_state["forecast_response"] = response
                st.success("Forecasts updated")

    forecast_response = st.session_state.get("forecast_response") or {}
    if forecast_response:
        st.write("**Forecast summary**")
        st.json(forecast_response.get("forecasts", forecast_response))
        _download_button(
            "Download forecast JSON",
            json.dumps(forecast_response, indent=2, default=str).encode("utf-8"),
            file_name=f"{ticker}_forecast.json",
            mime="application/json",
        )

    if st.button("Run backtest"):
        with st.spinner("Running backtest..."):
            response = _request(
                f"/backtests/{ticker}",
                method="POST",
                json_payload={"targets": _parse_targets(targets_raw)},
            )
            if response is not None:
                st.session_state["backtest_response"] = response
                st.success("Backtest complete")

    backtest_response = st.session_state.get("backtest_response") or {}
    if backtest_response:
        st.write("**Backtest performance**")
        st.json(backtest_response.get("backtest", backtest_response))
        _download_button(
            "Download backtest JSON",
            json.dumps(backtest_response, indent=2, default=str).encode("utf-8"),
            file_name=f"{ticker}_backtest.json",
            mime="application/json",
        )

st.subheader("Research Notes")
if st.button("Refresh research feed"):
    with st.spinner("Fetching research artifacts..."):
        response = _request("/research")
        if response is not None:
            st.session_state["research_response"] = response
            st.success("Research feed refreshed")

research_response = st.session_state.get("research_response") or {}
items = research_response.get("items", []) if isinstance(research_response, dict) else []
if items:
    research_frame = pd.DataFrame(items)
    if not research_frame.empty:
        st.dataframe(research_frame, use_container_width=True)
    _download_button(
        "Download research JSON",
        json.dumps(items, indent=2, default=str).encode("utf-8"),
        file_name="research_feed.json",
        mime="application/json",
    )
else:
    st.info("Click 'Refresh research feed' to load the latest artifacts.")
