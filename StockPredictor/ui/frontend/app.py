"""Screener-only Streamlit frontend."""

from __future__ import annotations

import os
from datetime import date
from typing import Any, Dict

import requests
import streamlit as st

from ui.frontend.screener import render_screener

DEFAULT_API_URL = os.getenv("STOCK_PREDICTOR_API_URL", "http://localhost:8000")
DEFAULT_API_KEY = os.getenv("STOCK_PREDICTOR_UI_API_KEY", "")

st.set_page_config(page_title="Stock Predictor Screener", layout="wide")

for key, value in {
    "api_base": DEFAULT_API_URL,
    "api_key": DEFAULT_API_KEY,
}.items():
    st.session_state.setdefault(key, value)


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
    except requests.RequestException as exc:  # pragma: no cover
        detail = getattr(getattr(exc, "response", None), "text", str(exc))
        st.error(f"Request to {url} failed: {detail}")
        return None
    return response.json()


with st.sidebar:
    st.header("Configuration")
    st.text_input("API base URL", key="api_base")
    st.text_input("API key", key="api_key")

st.title("Stock Predictor Screener")
st.caption("Nur der Screener ist aktiv. Alle anderen UI-Bereiche wurden entfernt.")

render_screener(_request, default_end_date=date.today())
