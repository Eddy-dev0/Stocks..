## Manual beta guidance rendering check

1. Run the Streamlit app via `streamlit run ui/frontend/app.py` with a forecast-capable backend.
2. Trigger a forecast that returns `risk_guidance.beta` entries (e.g., ticker with computed betas).
3. Confirm the forecast panel shows the "Risk guidance" block with volatility and uncertainty metrics.
4. Verify the "Beta sensitivity" section lists each benchmark with label, window, value, and risk band phrasing (for example, "S&P 500 beta 1.65 (21-day window) – high volatility").
5. Download the forecast JSON and confirm the beta details in the UI align with the API payload.
