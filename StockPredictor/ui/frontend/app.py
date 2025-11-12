"""Streamlit dashboard for exploring Stock Predictor outputs."""

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from typing import Any, Dict, List

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
        numeric_cols = frame.select_dtypes("number").columns
        if "close" in frame.columns and "close" in numeric_cols:
            frame["MA20"] = frame["close"].rolling(window=20, min_periods=1).mean()
            frame["MA50"] = frame["close"].rolling(window=50, min_periods=1).mean()
            numeric_cols = list(dict.fromkeys(list(numeric_cols) + ["MA20", "MA50"]))
        st.line_chart(frame[numeric_cols])
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
