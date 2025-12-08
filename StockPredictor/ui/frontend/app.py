"""Streamlit dashboard for exploring Stock Predictor outputs."""

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from typing import Any, Dict, List, Mapping

import altair as alt
import pandas as pd
import requests
import streamlit as st

from stock_predictor.core.features import (
    FEATURE_REGISTRY,
    FeatureToggles,
    default_feature_toggles,
)

DEFAULT_API_URL = os.getenv("STOCK_PREDICTOR_API_URL", "http://localhost:8000")
DEFAULT_TICKER = os.getenv("STOCK_PREDICTOR_DEFAULT_TICKER", "AAPL")
DEFAULT_API_KEY = os.getenv("STOCK_PREDICTOR_UI_API_KEY", "")
IMPLEMENTED_FEATURE_GROUPS = {
    name for name, spec in FEATURE_REGISTRY.items() if getattr(spec, "implemented", False)
}
DEFAULT_FEATURE_TOGGLES = FeatureToggles.from_any(
    {
        name: enabled
        for name, enabled in default_feature_toggles().items()
        if name in IMPLEMENTED_FEATURE_GROUPS
    },
    defaults={name: False for name in IMPLEMENTED_FEATURE_GROUPS},
)

st.set_page_config(page_title="Stock Predictor Dashboard", layout="wide")

for key, value in {
    "api_base": DEFAULT_API_URL,
    "api_key": DEFAULT_API_KEY,
    "data_response": None,
    "forecast_response": None,
    "backtest_response": None,
    "train_response": None,
    "research_response": None,
    "buy_zone_response": None,
    "insights_response": None,
    "live_price_response": None,
    "accuracy_response": None,
    "feature_toggles": DEFAULT_FEATURE_TOGGLES.copy(),
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


def _with_feature_toggles(payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Attach the active feature toggles from session state to a payload."""

    updated = dict(payload or {})
    toggles = st.session_state.get("feature_toggles") or DEFAULT_FEATURE_TOGGLES
    if isinstance(toggles, FeatureToggles):
        updated["feature_toggles"] = toggles.asdict()
    else:
        updated["feature_toggles"] = dict(toggles)
    return updated


def _render_feature_toggle_summary(block: Mapping[str, Any] | None) -> None:
    """Display which feature groups were configured and executed for a run."""

    if not isinstance(block, Mapping):
        return

    configured = block.get("feature_toggles")
    executed = (
        block.get("feature_groups_used")
        or block.get("executed_feature_groups")
        or block.get("used_feature_groups")
    )
    if not configured and not executed:
        return

    rows: list[dict[str, object]] = []
    if isinstance(configured, Mapping):
        for name, enabled in sorted(configured.items()):
            rows.append(
                {
                    "Feature group": name,
                    "Configured": bool(enabled),
                    "Executed": bool(executed and name in executed),
                }
            )
    elif executed:
        rows = [
            {"Feature group": name, "Configured": None, "Executed": True}
            for name in sorted(executed)
        ]

    if rows:
        st.markdown("**Feature group usage**")
        st.table(pd.DataFrame(rows))


def _extract_feature_usage(
    forecast_block: Mapping[str, Any] | None,
) -> dict[str, object]:
    """Normalise feature usage details from a PredictionResult payload."""

    feature_groups = []
    indicators = []
    summary_text = None

    if isinstance(forecast_block, Mapping):
        raw_groups = forecast_block.get("feature_groups_used")
        if isinstance(raw_groups, (list, tuple, set)):
            feature_groups = sorted({str(item) for item in raw_groups})

        raw_indicators = forecast_block.get("indicators_used")
        if isinstance(raw_indicators, (list, tuple, set)):
            indicators = sorted({str(item) for item in raw_indicators})

        summary_text = forecast_block.get("feature_usage_summary")

    return {
        "feature_groups": feature_groups,
        "indicators": indicators,
        "summary": summary_text,
    }


def _render_feature_usage(
    forecast_block: Mapping[str, Any] | None,
    *,
    heading: str = "Features used in this prediction",
) -> None:
    """Render feature usage details from a PredictionResult block."""

    usage = _extract_feature_usage(forecast_block)
    feature_groups: list[str] = usage["feature_groups"]
    indicators: list[str] = usage["indicators"]
    summary_text = usage["summary"]

    if not (feature_groups or indicators or summary_text):
        return

    st.markdown(f"**{heading}**")

    if summary_text:
        st.markdown(summary_text)

    if feature_groups:
        st.table(pd.DataFrame({"Feature group": feature_groups}))

    if indicators:
        st.table(pd.DataFrame({"Indicators": indicators}))


def _render_model_input_details(forecast_block: Mapping[str, Any] | None) -> None:
    """Show feature groups, latest feature snapshot, and data sources."""

    if not isinstance(forecast_block, Mapping):
        return

    snapshot = forecast_block.get("latest_features_snapshot")
    feature_groups = forecast_block.get("feature_groups")
    target_validation = forecast_block.get("target_validation")
    data_sources = forecast_block.get("data_sources") or forecast_block.get("sources")

    with st.expander("Model inputs & data sources", expanded=False):
        if snapshot:
            st.caption("Latest feature vector sent to the model (scroll to view all inputs).")
            st.dataframe(
                pd.DataFrame(snapshot).T,
                use_container_width=True,
                height=320,
            )

        if feature_groups:
            rows: list[dict[str, object]] = []
            for name, details in sorted(feature_groups.items()):
                if not isinstance(details, Mapping):
                    continue
                rows.append(
                    {
                        "Feature group": name,
                        "Executed": details.get("executed"),
                        "Status": details.get("status"),
                        "Columns": len(details.get("columns", [])) if isinstance(details.get("columns"), (list, tuple, set)) else None,
                    }
                )
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, height=220)

        if target_validation:
            validation_rows: list[dict[str, object]] = []
            if isinstance(target_validation, Mapping):
                for horizon, details in sorted(target_validation.items()):
                    if not isinstance(details, Mapping):
                        continue
                    validation_rows.append(
                        {
                            "Horizon": horizon,
                            "Close aligned": details.get("close_aligned"),
                            "Close max error": details.get("close_alignment_error"),
                            "Return aligned": details.get("return_aligned"),
                            "Return max error": details.get("return_alignment_error"),
                        }
                    )
            if validation_rows:
                st.dataframe(pd.DataFrame(validation_rows), use_container_width=True, height=200)

        if data_sources:
            sources = data_sources if isinstance(data_sources, list) else [data_sources]
            st.table(pd.DataFrame({"Data source": sources}))


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


def _format_currency(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return str(value)


def _format_percentage(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return str(value)


def _build_live_price_table(snapshot: Dict[str, Any]) -> pd.DataFrame:
    probabilities = snapshot.get("probabilities") or {}
    rows = [
        {"Metric": "Ticker", "Value": snapshot.get("ticker")},
        {"Metric": "Market time", "Value": snapshot.get("market_time")},
        {"Metric": "Last price", "Value": _format_currency(snapshot.get("last_price"))},
        {
            "Metric": "Predicted close",
            "Value": _format_currency(snapshot.get("predicted_close")),
        },
        {
            "Metric": "Expected change %",
            "Value": _format_percentage(snapshot.get("expected_change_pct")),
        },
        {
            "Metric": "Expected low",
            "Value": _format_currency(snapshot.get("expected_low")),
        },
        {"Metric": "Stop-loss", "Value": _format_currency(snapshot.get("stop_loss"))},
        {
            "Metric": "Probability up",
            "Value": _format_percentage(probabilities.get("up")),
        },
        {
            "Metric": "Probability down",
            "Value": _format_percentage(probabilities.get("down")),
        },
    ]
    return pd.DataFrame(rows)


def _beta_band_description(level: str) -> str:
    normalized = str(level).lower()
    if normalized == "high":
        return "high volatility"
    if normalized == "defensive":
        return "defensive / low sensitivity"
    return "market-like sensitivity"


def _summarise_beta_guidance(beta_block: Dict[str, Any]) -> List[str]:
    summaries: List[str] = []
    for key, payload in sorted(beta_block.items()):
        if not isinstance(payload, dict):
            continue
        label = payload.get("label") or key.upper()
        value = payload.get("value")
        if value is None:
            continue
        window = payload.get("window")
        risk_level = payload.get("risk_level")
        band_text = _beta_band_description(risk_level) if risk_level else "market-like sensitivity"
        window_text = f" ({int(window)}-day window)" if isinstance(window, (int, float)) else ""
        summaries.append(f"{label} beta {float(value):.2f}{window_text} – {band_text}")
    return summaries


def _extract_risk_guidance(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    if "risk_guidance" in payload and isinstance(payload.get("risk_guidance"), dict):
        return payload["risk_guidance"]
    recommendation = payload.get("recommendation")
    if isinstance(recommendation, dict) and isinstance(
        recommendation.get("risk_guidance"), dict
    ):
        return recommendation["risk_guidance"]
    return {}


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


def _extract_probabilities(payload: Any) -> tuple[float | None, float | None]:
    forecast_block = payload.get("forecasts", payload) if isinstance(payload, dict) else {}
    monte_carlo_prob = None
    model_pred_prob = None
    if isinstance(forecast_block, dict):
        monte_carlo_prob = forecast_block.get("monte_carlo_target_hit_probability")
        if monte_carlo_prob is None:
            event_probs = forecast_block.get("event_probabilities", {})
            if isinstance(event_probs, dict):
                monte_carlo_prob = (event_probs.get("monte_carlo_target_hit", {}) or {}).get(
                    "probability"
                )
        prob_block = forecast_block.get("probabilities") or forecast_block.get(
            "prediction_probabilities"
        )
        if isinstance(prob_block, dict):
            for candidate in ("direction", "close", "target"):
                if candidate in prob_block and isinstance(prob_block[candidate], dict):
                    values = prob_block[candidate]
                    up_prob = values.get("up") or values.get("bullish") or values.get("positive")
                    down_prob = values.get("down") or values.get("bearish") or values.get("negative")
                    if isinstance(up_prob, (int, float)) and isinstance(down_prob, (int, float)):
                        model_pred_prob = max(float(up_prob), float(down_prob))
                        break
                    numeric = [float(value) for value in values.values() if isinstance(value, (int, float))]
                    if numeric:
                        model_pred_prob = max(numeric)
                        break
    return monte_carlo_prob, model_pred_prob


def _risk_level(value: float | None, *, thresholds: tuple[float, float]) -> str:
    if value is None:
        return "unknown"
    low, high = thresholds
    if value >= high:
        return "high"
    if value >= low:
        return "moderate"
    return "low"


def _sentiment_label(score: float | None) -> str:
    if score is None:
        return "neutral"
    if score > 0.1:
        return "positive"
    if score < -0.1:
        return "negative"
    return "neutral"


def _enrich_indicators(frame: pd.DataFrame) -> pd.DataFrame:
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
    working["MACD_hist"] = working["MACD"] - working["MACD_signal"]
    working["MACD_cross_up"] = (working["MACD"] > working["MACD_signal"]) & (
        working["MACD"].shift(1) <= working["MACD_signal"].shift(1)
    )

    working["BB_mid"] = close.rolling(window=20, min_periods=20).mean()
    rolling_std = close.rolling(window=20, min_periods=20).std()
    working["BB_lower"] = working["BB_mid"] - 2 * rolling_std
    working["BB_upper"] = working["BB_mid"] + 2 * rolling_std
    working["BB_width"] = working["BB_upper"] - working["BB_lower"]

    if {"high", "low"}.issubset(working.columns):
        high = working["high"].astype(float)
        low = working["low"].astype(float)
        previous_close = close.shift(1)
        tr = pd.concat([high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1).max(axis=1)
        working["ATR14"] = tr.rolling(window=14, min_periods=1).mean()

    if "volume" in working.columns:
        pv = close * working["volume"].astype(float)
        cumulative_pv = pv.cumsum()
        cumulative_volume = working["volume"].astype(float).cumsum()
        working["VWAP"] = cumulative_pv / cumulative_volume.replace(0, pd.NA)

    working["VOL30"] = close.pct_change().rolling(window=30, min_periods=20).std() * (252**0.5)

    if "close" in working.select_dtypes("number").columns:
        working["MA20"] = close.rolling(window=20, min_periods=1).mean()
        working["MA50"] = close.rolling(window=50, min_periods=1).mean()
        working["MA200"] = close.rolling(window=200, min_periods=1).mean()

    return working


def _find_buy_zone(frame: pd.DataFrame, forecast_low: float | None) -> Dict[str, Any] | None:
    if "close" not in frame.columns:
        return None

    working = _enrich_indicators(frame)
    close = working["close"].astype(float)

    conditions = (
        (working["RSI14"] < 30)
        & working["MACD_cross_up"].fillna(False)
        & (close <= working["BB_lower"])
    )
    candidates = working[conditions]
    if candidates.empty:
        return None

    base_row = candidates.loc[candidates["close"].idxmin()]
    base_low = float(base_row["close"])
    price_low = min(base_low, forecast_low) if forecast_low is not None else base_low
    price_high = max(float(candidates["close"].max()), price_low * 1.02)

    volatility = float(base_row.get("VOL30", float("nan")))
    atr = float(base_row.get("ATR14", float("nan")))
    reason = (
        "RSI < 30, bullish MACD crossover, and lower Bollinger touch; "
        f"volatility (30d) ~ {volatility:.2%}"
    )
    if not pd.isna(atr):
        reason += f"; ATR14 ${atr:.2f}"
    if forecast_low is not None:
        reason += "; anchored near forecasted lows"

    return {
        "start": candidates.index.min(),
        "end": candidates.index.max(),
        "price_low": price_low,
        "price_high": price_high,
        "reason": reason,
    }


def _build_price_chart(
    frame: pd.DataFrame,
    *,
    buy_zone: Dict[str, Any] | None = None,
    show_buy_zone: bool = True,
    show_ma50: bool = True,
    show_ma200: bool = False,
    show_bollinger: bool = True,
    show_vwap: bool = False,
    show_volume: bool = True,
) -> alt.VConcatChart:
    time_index = frame.index.name or "timestamp"
    data_frame = frame.reset_index().rename(columns={frame.index.name or "index": "time"})

    price_series = ["close"]
    if show_ma50 and "MA50" in data_frame.columns:
        price_series.append("MA50")
    if show_ma200 and "MA200" in data_frame.columns:
        price_series.append("MA200")
    melted = data_frame.melt("time", value_vars=price_series, var_name="series", value_name="value")

    price_layers: List[alt.Chart] = [
        alt.Chart(melted)
        .mark_line()
        .encode(
            x=alt.X("time:T", title=time_index.capitalize()),
            y=alt.Y("value:Q", title="Price"),
            color=alt.Color("series:N", title="Series"),
            tooltip=["time:T", "series:N", "value:Q"],
        )
    ]

    if show_bollinger and {"BB_lower", "BB_upper", "BB_mid"}.issubset(data_frame.columns):
        bollinger_source = data_frame
        price_layers.append(
            alt.Chart(bollinger_source)
            .mark_area(opacity=0.12, color="gray")
            .encode(
                x="time:T",
                y="BB_lower:Q",
                y2="BB_upper:Q",
                tooltip=[
                    "time:T",
                    alt.Tooltip("BB_lower:Q", title="BB Lower"),
                    alt.Tooltip("BB_upper:Q", title="BB Upper"),
                    alt.Tooltip("BB_width:Q", title="BB Width"),
                ],
            )
        )
        price_layers.append(
            alt.Chart(bollinger_source)
            .mark_line(strokeDash=[4, 4], color="gray")
            .encode(x="time:T", y="BB_mid:Q", tooltip=["time:T", alt.Tooltip("BB_mid:Q", title="BB Mid")])
        )

    if show_vwap and "VWAP" in data_frame.columns:
        price_layers.append(
            alt.Chart(data_frame)
            .mark_line(color="orange")
            .encode(x="time:T", y="VWAP:Q", tooltip=["time:T", alt.Tooltip("VWAP:Q", title="VWAP")])
        )

    if show_buy_zone and buy_zone:
        rect_frame = pd.DataFrame([buy_zone])
        price_layers.append(
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

    price_chart = alt.layer(*price_layers).resolve_scale(color="independent")

    panels: List[alt.Chart] = [price_chart.properties(height=320)]

    if show_volume and "volume" in data_frame.columns:
        volume_chart = (
            alt.Chart(data_frame)
            .mark_bar(color="#4c78a8", opacity=0.6)
            .encode(
                x=alt.X("time:T", title=""),
                y=alt.Y("volume:Q", title="Volume"),
                tooltip=["time:T", alt.Tooltip("volume:Q", title="Volume")],
            )
            .properties(height=120)
        )
        panels.append(volume_chart)

    if {"MACD", "MACD_signal", "MACD_hist"}.issubset(data_frame.columns):
        macd_base = alt.Chart(data_frame)
        macd_chart = alt.layer(
            macd_base.mark_bar(color="#9ecae9").encode(x="time:T", y="MACD_hist:Q"),
            macd_base.mark_line(color="#1f77b4").encode(x="time:T", y="MACD:Q", tooltip=["time:T", "MACD:Q"]),
            macd_base.mark_line(color="#ff7f0e", strokeDash=[4, 3]).encode(
                x="time:T", y="MACD_signal:Q", tooltip=[alt.Tooltip("MACD_signal:Q", title="Signal")]
            ),
        ).resolve_scale(y="shared").properties(height=140, title="MACD")
        panels.append(macd_chart)

    if "RSI14" in data_frame.columns:
        rsi_source = data_frame
        rsi_chart = alt.layer(
            alt.Chart(rsi_source)
            .mark_line(color="#2ca02c")
            .encode(x="time:T", y=alt.Y("RSI14:Q", title="RSI"), tooltip=["time:T", alt.Tooltip("RSI14:Q", title="RSI14")]),
            alt.Chart(pd.DataFrame({"y": [30, 70]}))
            .mark_rule(strokeDash=[3, 3], color="gray")
            .encode(y="y:Q"),
        ).properties(height=140, title="RSI (14)")
        panels.append(rsi_chart)

    return alt.vconcat(*panels, spacing=16).resolve_scale(x="shared")


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
    force_refresh_market = st.checkbox("Refresh market data on fetch", value=True)
    refresh_before_forecast = st.checkbox("Refresh data before forecasting", value=True)
    refresh_before_buy_zone = st.checkbox(
        "Refresh data before buy-zone computation", value=True
    )
    use_custom_horizon = st.checkbox("Specify forecast horizon", value=False)
    horizon_value = None
    if use_custom_horizon:
        horizon_value = st.number_input("Forecast horizon", min_value=1, max_value=365, value=5, step=1)

    st.header("Validation & tuning")
    evaluation_strategy = st.selectbox(
        "Evaluation strategy",
        options=["time_series", "rolling", "holdout"],
        index=0,
        help="Use time-series aware cross-validation by default to avoid leakage.",
    )
    evaluation_folds = st.number_input(
        "Cross-validation folds",
        min_value=2,
        max_value=10,
        value=5,
        step=1,
        help="Number of rolling folds for time-series evaluation.",
    )
    tuning_enabled = st.checkbox("Enable hyperparameter tuning", value=True)
    tuning_iterations = st.slider(
        "Tuning iterations",
        min_value=1,
        max_value=50,
        value=10,
        help="Randomly sample parameter sets before final training.",
    )

    st.header("Live model inputs")
    expected_change_pct_model = st.number_input(
        "Expected change % (decimal)",
        value=0.0,
        step=0.001,
        format="%.4f",
        help="Model-predicted percentage change expressed as a decimal (e.g., 0.02 for +2%).",
    )
    expected_low_pct_model = st.number_input(
        "Expected low % (decimal)",
        value=-0.02,
        step=0.001,
        format="%.4f",
        help="Expected downside move from the latest price (negative decimals allowed).",
    )
    stop_loss_pct = st.number_input(
        "Stop-loss % (decimal)",
        value=0.05,
        min_value=0.0,
        step=0.001,
        format="%.4f",
        help="Risk threshold relative to the latest price (e.g., 0.05 for 5% below).",
    )
    prob_up = st.number_input(
        "Probability up (0-1)",
        value=0.5,
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        format="%.2f",
        help="Directional probability from the model output.",
    )

    st.header("Feature groups")
    feature_toggles = FeatureToggles.from_any(
        st.session_state.get("feature_toggles") or DEFAULT_FEATURE_TOGGLES,
        defaults=DEFAULT_FEATURE_TOGGLES.asdict(),
    )
    for name in sorted(IMPLEMENTED_FEATURE_GROUPS):
        label = name.replace("_", " ").title()
        default_value = DEFAULT_FEATURE_TOGGLES.get(name, True)
        feature_toggles[name] = st.checkbox(
            label, value=feature_toggles.get(name, default_value)
        )
    st.session_state["feature_toggles"] = feature_toggles

    st.header("Chart overlays")
    show_all_indicators = st.checkbox(
        "Show all indicators",
        value=False,
        help="Master toggle to enable every overlay (moving averages, Bollinger bands, VWAP, volume, and more).",
    )
    show_ma50 = st.checkbox("Show 50-day MA", value=True, disabled=show_all_indicators)
    show_ma200 = st.checkbox("Show 200-day MA", value=False, disabled=show_all_indicators)
    show_bollinger = st.checkbox(
        "Show Bollinger bands", value=True, disabled=show_all_indicators
    )
    show_volume = st.checkbox(
        "Show volume histogram", value=True, disabled=show_all_indicators
    )
    show_vwap = st.checkbox(
        "Show VWAP", value=False, help="Requires volume data", disabled=show_all_indicators
    )

    overlay_flags = {
        "show_ma50": show_ma50,
        "show_ma200": show_ma200,
        "show_bollinger": show_bollinger,
        "show_vwap": show_vwap,
        "show_volume": show_volume,
    }

    if show_all_indicators:
        overlay_flags = {key: True for key in overlay_flags}

    show_ma50 = overlay_flags["show_ma50"]
    show_ma200 = overlay_flags["show_ma200"]
    show_bollinger = overlay_flags["show_bollinger"]
    show_vwap = overlay_flags["show_vwap"]
    show_volume = overlay_flags["show_volume"]

    st.header("Insights & Risk Panel")
    st.caption("Aggregated probabilities, beta, volatility, and sentiment signals.")
    insights_refresh = st.checkbox(
        "Refresh sentiment", value=False, key="insights_refresh_toggle"
    )
    if st.button("Load insights", type="secondary"):
        with st.spinner("Fetching sentiment insights..."):
            response = _request(
                f"/insights/{ticker}", params={"refresh": bool(insights_refresh)}
            )
            if response is not None:
                st.session_state["insights_response"] = response
                st.success("Insights updated")

    probability_block = _extract_probabilities(forecast_response or {})
    monte_carlo_prob, model_pred_prob = probability_block
    risk_guidance = _extract_risk_guidance(forecast_block)
    beta_block = risk_guidance.get("beta") if isinstance(risk_guidance, dict) else None
    vol_value = risk_guidance.get("volatility") if isinstance(risk_guidance, dict) else None
    insights_payload = st.session_state.get("insights_response") or {}
    sentiment_block = insights_payload.get("sentiment", {})
    sentiment_score = sentiment_block.get("latest_score")
    sentiment_state = _sentiment_label(sentiment_score)

    metric_rows = st.container()
    with metric_rows:
        prob_cols = st.columns(2)
        prob_cols[0].metric(
            "Monte Carlo target hit", f"{float(monte_carlo_prob) * 100:.2f}%" if monte_carlo_prob is not None else "—",
            help="Probability the simulated price path reaches the target by the forecast horizon."
        )
        prob_cols[1].metric(
            "Model prediction", f"{float(model_pred_prob) * 100:.2f}%" if model_pred_prob is not None else "—",
            help="Highest directional probability returned by the model (bullish vs. bearish)."
        )
        risk_cols = st.columns(2)
        if beta_block and isinstance(beta_block, dict):
            for idx, (name, detail) in enumerate(sorted(beta_block.items())):
                if not isinstance(detail, dict):
                    continue
                value = detail.get("value")
                if value is None:
                    continue
                level = _risk_level(float(value), thresholds=(0.9, 1.2))
                risk_cols[idx % 2].metric(
                    f"Beta ({detail.get('label') or name})",
                    f"{float(value):.2f}",
                    delta=f"{level.title()} risk",
                    delta_color="inverse" if level == "high" else "normal",
                    help="Beta above 1 implies amplified moves vs. the market; below 1 dampens swings.",
                )
        risk_level = _risk_level(vol_value if isinstance(vol_value, (int, float)) else None, thresholds=(0.25, 0.5))
        risk_cols[1].metric(
            "Volatility (forecast)",
            f"{float(vol_value):.2%}" if isinstance(vol_value, (int, float)) else "—",
            delta=f"{risk_level.title()} risk" if vol_value is not None else "",
            delta_color="inverse" if risk_level == "high" else "normal",
            help="Annualised volatility derived from forecast dispersion.",
        )
        sentiment_col = st.container()
        with sentiment_col:
            st.metric(
                "News sentiment", f"{float(sentiment_score):+.2f}" if sentiment_score is not None else "—",
                delta=sentiment_state.title(),
                delta_color="normal" if sentiment_state == "positive" else "inverse" if sentiment_state == "negative" else "off",
                help="Rolling news polarity; positive skews bullish, negative hints at downside risk.",
            )

    with st.expander("How to interpret these metrics"):
        st.markdown(
            "- **Monte Carlo target hit**: likelihood simulated price paths touch the configured target.\n"
            "- **Model prediction**: most confident directional probability from the model outputs.\n"
            "- **Beta**: sensitivity vs. the market (1.2+ = high risk).\n"
            "- **Volatility**: annualised dispersion; higher values increase drawdown risk.\n"
            "- **News sentiment**: rolling polarity from recent headlines; sustained negatives can weigh on returns."
        )

    st.header("Risk & Volatility")
    st.caption("Computed from visible market data")
    risk_metrics_placeholder = st.container()

feature_toggles = FeatureToggles.from_any(
    st.session_state.get("feature_toggles") or DEFAULT_FEATURE_TOGGLES,
    defaults=DEFAULT_FEATURE_TOGGLES.asdict(),
)

forecast_response = st.session_state.get("forecast_response") or {}
forecast_block = forecast_response.get("forecasts", forecast_response)

feature_usage = _extract_feature_usage(forecast_block)

st.title("Stock Predictor Dashboard")
st.caption("Explore model forecasts, historical indicators, and curated research notes.")

st.subheader("Live price snapshot")
col_live_action, col_live_table = st.columns([1, 3])
with col_live_action:
    if st.button("Refresh live snapshot", type="primary"):
        with st.spinner("Fetching live pricing..."):
            response = _request(
                f"/live-price/{ticker}",
                method="POST",
                json_payload={
                    "expected_change_pct_model": expected_change_pct_model,
                    "expected_low_pct_model": expected_low_pct_model,
                    "stop_loss_pct": stop_loss_pct,
                    "prob_up": prob_up,
                },
            )
            if response is not None:
                st.session_state["live_price_response"] = response
                st.success("Live snapshot updated")

with col_live_table:
    live_payload = (st.session_state.get("live_price_response") or {}).get("price")
    if live_payload:
        st.dataframe(
            _build_live_price_table(live_payload),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("Click 'Refresh live snapshot' to load the latest intraday metrics.")

col_data, col_forecast = st.columns(2)

with col_data:
    st.subheader("Market Data & Indicators")
    if feature_usage.get("indicators"):
        st.caption("Indicators used by the latest forecast run:")
        st.table(pd.DataFrame({"Indicator": feature_usage["indicators"]}))
    _render_model_input_details(forecast_block)
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

    if st.button("Compute tactical buy zone", type="secondary"):
        with st.spinner("Evaluating buy zone..."):
            response = _request(
                f"/buy-zone/{ticker}",
                method="POST",
                json_payload={
                    "refresh": bool(refresh_before_buy_zone),
                    "feature_toggles": feature_toggles.asdict(),
                },
            )
            if response is not None:
                st.session_state["buy_zone_response"] = response
                st.success("Buy zone analysis updated")

    data_response = st.session_state.get("data_response") or {}
    payload = data_response.get("data") if isinstance(data_response, dict) else data_response
    frame = _coerce_dataframe(payload)
    if frame is not None and not frame.empty:
        frame = _normalise_timeseries(frame)
        frame = _enrich_indicators(frame)
        forecast_low = _extract_forecast_low(st.session_state.get("forecast_response"))
        buy_zone = _find_buy_zone(frame, forecast_low)
        show_buy_zone = st.checkbox(
            "Show buy-zone overlay",
            value=bool(buy_zone),
            help="Highlight windows where RSI < 30, MACD crosses up, and price hugs the lower Bollinger band (anchored near forecast lows when available).",
        )
        st.altair_chart(
            _build_price_chart(
                frame,
                buy_zone=buy_zone,
                show_buy_zone=show_buy_zone,
                show_ma50=show_ma50,
                show_ma200=show_ma200,
                show_bollinger=show_bollinger,
                show_vwap=show_vwap,
                show_volume=show_volume,
            ),
            use_container_width=True,
        )
        with st.expander("How is the buy-zone derived?"):
            st.markdown(
                "The shaded region captures sessions where:\n"
                "- 14-day RSI dips below 30 (oversold)\n"
                "- The MACD line crosses above its signal (bullish momentum shift)\n"
                "- Closing price touches/under-cuts the 20-day lower Bollinger band\n"
                "- Volatility context (ATR/30d vol) is captured in the tooltip\n"
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

        buy_zone_response = st.session_state.get("buy_zone_response") or {}
        buy_zone_payload = (
            buy_zone_response.get("buy_zone")
            if isinstance(buy_zone_response, dict)
            else None
        )
        if buy_zone_payload:
            st.markdown("**Tactical buy-zone confirmations (API)**")
            bounds = buy_zone_payload.get("price_bounds", {})
            window = buy_zone_payload.get("window", {})
            col_left, col_right = st.columns(2)
            lower_bound = bounds.get("lower")
            upper_bound = bounds.get("upper")
            support_level = bounds.get("support")
            last_close_value = bounds.get("last_close")
            with col_left:
                st.write(
                    "Window:",
                    window.get("start"),
                    "to",
                    window.get("end"),
                )
                st.write(
                    "Bounds:",
                    f"${lower_bound:.2f}" if lower_bound is not None else "—",
                    "-",
                    f"${upper_bound:.2f}" if upper_bound is not None else "—",
                )
            with col_right:
                st.write(
                    "Support:",
                    f"${support_level:.2f}" if support_level is not None else "—",
                )
                st.write(
                    "Last close:",
                    f"${last_close_value:.2f}" if last_close_value is not None else "—",
                )

            confirmations = buy_zone_payload.get("confirmations", {})
            if confirmations:
                rows = []
                for name, detail in confirmations.items():
                    rows.append(
                        {
                            "signal": name,
                            "confirmed": "✅" if detail.get("confirmed") else "❌",
                            "value": detail.get("value"),
                            "threshold": detail.get("threshold"),
                            "detail": detail.get("detail"),
                        }
                    )
                st.dataframe(pd.DataFrame(rows))
            else:
                st.caption("No confirmation signals returned by the API.")
        with risk_metrics_placeholder:
            latest_valid = frame.dropna(how="all")
            if latest_valid.empty:
                st.caption("Volatility metrics unavailable for the current selection.")
            else:
                latest = latest_valid.iloc[-1]
                cols_risk = st.columns(2)
                atr_value = latest.get("ATR14")
                vol_value = latest.get("VOL30")
                cols_risk[0].metric("ATR (14)", f"${atr_value:.2f}" if pd.notna(atr_value) else "—")
                cols_risk[1].metric(
                    "Volatility (30d ann.)",
                    f"{vol_value:.2%}" if pd.notna(vol_value) else "—",
                )
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
    if st.button("Daten aktualisieren & Modell neu trainieren"):
        with st.spinner("Aktualisiere Daten und trainiere Modelle..."):
            response = _request(
                f"/train/{ticker}",
                method="POST",
                json_payload={
                    "targets": _parse_targets(targets_raw),
                    "horizon": horizon_value,
                    "feature_toggles": feature_toggles.asdict(),
                    "evaluation_strategy": evaluation_strategy,
                    "evaluation_folds": int(evaluation_folds) if evaluation_strategy == "time_series" else None,
                    "tuning_enabled": bool(tuning_enabled),
                    "tuning_iterations": int(tuning_iterations),
                },
            )
            if response is not None:
                st.session_state["train_response"] = response
                st.success("Training abgeschlossen")

    train_response = st.session_state.get("train_response") or {}
    if train_response:
        st.write("**Training & Refresh Ergebnisse**")
        st.json(train_response)
        _download_button(
            "Download Training JSON",
            json.dumps(train_response, indent=2, default=str).encode("utf-8"),
            file_name=f"{ticker}_training.json",
            mime="application/json",
        )

    if st.button("Run forecast"):
        with st.spinner("Generating forecasts..."):
            response = _request(
                f"/forecasts/{ticker}",
                method="POST",
                json_payload={
                    "targets": _parse_targets(targets_raw),
                    "refresh": bool(refresh_before_forecast),
                    "horizon": horizon_value,
                    "feature_toggles": feature_toggles.asdict(),
                },
            )
            if response is not None:
                st.session_state["forecast_response"] = response
                st.success("Forecasts updated")

    if forecast_response:
        st.write("**Forecast summary**")
        monte_carlo_prob = None
        if isinstance(forecast_block, dict):
            monte_carlo_prob = forecast_block.get("monte_carlo_target_hit_probability")
            if monte_carlo_prob is None:
                event_probs = forecast_block.get("event_probabilities", {})
                if isinstance(event_probs, dict):
                    monte_carlo_prob = (
                        event_probs.get("monte_carlo_target_hit", {}) or {}
                    ).get("probability")
        if isinstance(monte_carlo_prob, (int, float)):
            st.metric(
                "Monte Carlo probability of target hit",
                f"{float(monte_carlo_prob) * 100:.2f}%",
            )
        risk_guidance = _extract_risk_guidance(forecast_block)
        if risk_guidance:
            st.markdown("**Risk guidance**")
            risk_cols = st.columns(2)
            vol_value = risk_guidance.get("volatility")
            uncert_value = risk_guidance.get("uncertainty_std")
            if vol_value is not None:
                risk_cols[0].metric("Forecast volatility", f"{float(vol_value):.2%}")
            if uncert_value is not None:
                risk_cols[1].metric("Prediction uncertainty", f"{float(uncert_value):.3f}")

            beta_block = risk_guidance.get("beta")
            beta_summaries = _summarise_beta_guidance(beta_block) if beta_block else []
            if beta_summaries:
                st.markdown("**Beta sensitivity**")
                for summary in beta_summaries:
                    st.write(summary)
        _render_feature_usage(forecast_block)
        _render_feature_toggle_summary(forecast_block)
        _download_button(
            "Download forecast JSON",
            json.dumps(forecast_response, indent=2, default=str).encode("utf-8"),
            file_name=f"{ticker}_forecast.json",
            mime="application/json",
        )

    accuracy_cols = st.columns([1, 3])
    with accuracy_cols[0]:
        if st.button("Load accuracy summary"):
            with st.spinner("Retrieving accuracy metrics..."):
                params: Dict[str, Any] = {}
                if horizon_value:
                    params["horizon"] = int(horizon_value)
                response = _request(f"/accuracy/{ticker}", params=params)
                if response is not None:
                    st.session_state["accuracy_response"] = response
                    st.success("Accuracy summary refreshed")

    with accuracy_cols[1]:
        accuracy_response = st.session_state.get("accuracy_response") or {}
        accuracy_payload = (
            accuracy_response.get("accuracy")
            if isinstance(accuracy_response, Mapping)
            else None
        )
        if accuracy_payload:
            st.markdown("**Directional accuracy**")
            correct_pct = accuracy_payload.get("correct_pct")
            incorrect_pct = accuracy_payload.get("incorrect_pct")
            total_predictions = accuracy_payload.get("total_predictions")
            runs_considered = accuracy_payload.get("runs_considered")
            stats_cols = st.columns(3)
            stats_cols[0].metric(
                "Correct",
                f"{float(correct_pct) * 100:.1f}%" if correct_pct is not None else "—",
                help="Share of predictions that matched actual direction across stored runs.",
            )
            stats_cols[1].metric(
                "Incorrect",
                f"{float(incorrect_pct) * 100:.1f}%" if incorrect_pct is not None else "—",
                help="Share of predictions that missed the observed direction.",
            )
            stats_cols[2].metric(
                "Total predictions",
                f"{int(total_predictions):,}" if isinstance(total_predictions, (int, float)) else "—",
                help="Combined out-of-sample predictions evaluated in backtests.",
            )
            st.caption(
                f"Runs considered: {runs_considered or 0} | Horizon: {accuracy_payload.get('horizon') or 'active'}"
            )
        else:
            st.info("Run a backtest and refresh accuracy to see directional hit rates.")

    if st.button("Run backtest"):
        with st.spinner("Running backtest..."):
            response = _request(
                f"/backtests/{ticker}",
                method="POST",
                json_payload={
                    "targets": _parse_targets(targets_raw),
                    "feature_toggles": feature_toggles.asdict(),
                },
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
