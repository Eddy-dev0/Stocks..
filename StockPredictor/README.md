# Stock Predictor

The Stock Predictor project provides an end-to-end machine learning pipeline for
downloading financial market data, engineering features, training a predictive
model and generating forecasts. The modernised architecture centres around the
``StockPredictorApplication`` orchestrator which wires together the analytical
core, external data providers, research extensions and user interfaces. A
dedicated launcher (`main.py`) now focuses on starting the interactive
interfaces—Tkinter desktop, FastAPI backend and Streamlit dashboard—while the
underlying orchestration APIs remain available for automation or scripting
scenarios.

## Features

- Download historical price data via [yfinance](https://github.com/ranaroussi/yfinance).
- Optional download of recent company news headlines from Financial Modeling
  Prep's public API (requires an API key, falls back to demo limits).
- Automatic sentiment scoring for news articles using VADER.
- Feature engineering with common technical indicators and aggregated
  sentiment information.
- Volume-aware price features (OBV, volume-weighted momentum, orderflow
  imbalance) plus optional macro series merges for benchmarks like VIX, DXY
  or Treasury yields, all controllable via feature toggles.
- Machine learning model (Random Forest Regressor) with persisted metrics and
  trained model artefacts.
- Optional classical time-series baselines (ARIMA, Holt-Winters, Prophet) with
  seasonal controls for quick benchmarking against the ML stack.
- Local SQLite database that stores prices, indicators, fundamentals and news
  for fast reuse across runs.
- Extended database coverage for corporate events, options analytics,
  sentiment-derived feeds, ESG metrics and ownership/flow datasets (provider
  responses are cached when available and fall back to documented placeholders
  when upstream APIs return no data).
- FastAPI backend (`ui/api`) secured with API keys and a Streamlit dashboard (`ui/frontend`) for
  interactive exploration of data, forecasts, backtests and research artefacts.

## Monte Carlo target-hit simulation

The predictor reports a Monte Carlo estimate of the probability that the configured
target price will be touched within the forecast horizon. The simulation assumes a
geometric Brownian motion with constant drift and volatility, normally distributed
log-returns, and no jumps or regime shifts. Paths start from the latest observed
price with drift/volatility inferred from recent log returns, and the reported
probability reflects the share of simulated paths that hit the target at any step.
By default the system simulates 500,000 scenarios to sharpen the probability
estimate; set ``STOCK_PREDICTOR_MONTE_CARLO_PATHS`` to increase or decrease the
computational depth as needed. For even tighter estimates, provide
``STOCK_PREDICTOR_MONTE_CARLO_PRECISION`` alongside
``STOCK_PREDICTOR_MONTE_CARLO_MAX_PATHS`` to enable adaptive batching that keeps
running until the simulated hit probability's standard error drops below the
requested precision (or the path budget is exhausted).

## Project layout

```
StockPredictor/
├── main.py                # Unified launcher for desktop, API and dashboard interfaces
├── stock_predictor/
│   ├── app.py             # StockPredictorApplication orchestrator
│   ├── core/              # Pipelines, modelling and feature engineering
│   ├── providers/         # External data adapters (database, APIs, sentiment)
│   ├── research/          # Experimental modules (e.g. Elliott waves)
│   ├── ui/                # FastAPI entry points and UI integrations
│   └── docs/              # Helpers pointing to the documentation sources
├── data/                  # Cached datasets (created automatically)
└── models/                # Trained models & metrics (created automatically)
```

## Installation

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) create a `.env` file in the project root and provide the API key
   used for downloading news headlines:

   ```
   FINANCIALMODELINGPREP_API_KEY=demo
   ```

   If no key is provided the application continues to run but skips news and
   sentiment integration.

## Usage

All commands are executed from the project root (`StockPredictor/`).

### Desktop interface (Tkinter)

- **Quick start:** run `python main.py` to open the Tkinter desktop experience.
  Close the window (or press `Ctrl+C` in the terminal) to exit. Use
  `--no-train` or `--no-refresh` to disable the corresponding controls when
  launching automated kiosks or demos.
- **Combined launch:** run `python main.py --mode both` to start the Tkinter UI
  while simultaneously exposing the FastAPI service. This is helpful when the
  desktop interface should share data refreshes or predictions with other
  processes on the same machine.

### Web dashboard (Streamlit)

Start the Streamlit dashboard without the desktop client:

```bash
python main.py --mode dash --dash-port 8501
```

The command blocks until the dashboard process exits. Pass additional
environment variables (e.g. `STOCK_PREDICTOR_UI_API_KEY`) to configure access to
remote APIs.

### API service (FastAPI/Uvicorn)

To run only the REST API:

```bash
python main.py --mode api --host 0.0.0.0 --port 8000
```

The service is served by Uvicorn and exposes OpenAPI docs at `/docs`. Combine
all services with:

```bash
python main.py --mode full
```

This launches the Tkinter UI, the background API thread and the Streamlit
dashboard. Closing the Tkinter window gracefully terminates the dashboard
process.

Run `python main.py --help` for the full list of options and defaults.

### Headless live analysis loop

Keep predictions refreshed during the trading day without opening any user
interfaces:

```bash
python main.py --mode live-loop --interval 10m
```

The interval defaults to 300 seconds (5 minutes) and accepts plain seconds or
values suffixed with `s`/`m` (for example `45s`, `2m`). Each iteration refreshes
data as needed, runs the live prediction pipeline and logs the timestamped
results. To run the loop on a schedule with cron every 15 minutes during market
hours, add an entry such as:

```
*/15 9-16 * * 1-5 cd /path/to/StockPredictor && /usr/bin/env python main.py --mode live-loop --interval 15m >> /var/log/stockpredictor-live.log 2>&1
```

Adjust the cron window and log path to suit your environment.

## Data Splitting

Model evaluation defaults to a chronological holdout split: the earliest
``(1 - test_size)`` portion of the data is used for training and the most recent
rows form the test set. Shuffling is disabled for holdout to preserve the time
ordering of samples; attempting to enable shuffling alongside holdout raises a
configuration error. Time-series cross-validation and rolling backtests likewise
maintain chronological order when generating their splits.

### Time-series baselines

Enable classical baselines for the regression targets by providing the
`STOCK_PREDICTOR_TIME_SERIES_BASELINES` environment variable (comma-separated):

```bash
export STOCK_PREDICTOR_TIME_SERIES_BASELINES=arima,holt_winters,prophet
# Optional seasonal tuning shared across baselines and model-specific overrides
export STOCK_PREDICTOR_TIME_SERIES_PARAMS='{"global": {"seasonal_periods": 5}, "prophet": {"fourier_order": 7}}'
```

The training/evaluation pipeline will fit these univariate forecasters on the
historical price targets using the same holdout, time-series CV, or rolling
splits as the ML models. Metrics for each baseline (RMSE/MAE/MAPE) are stored
alongside the existing model evaluation payloads under the `baselines` key in
the persisted JSON metrics, making it easy to compare classical forecasts with
the feature-driven models.

## Database configuration

All market data is cached inside a relational database managed through
SQLAlchemy. By default the application uses SQLite with the database stored at
`data/market_data.sqlite`. The location can be changed by passing the
`--database-url` CLI argument or by setting the
`STOCK_PREDICTOR_DATABASE_URL` environment variable. When pointing at a remote
database the required driver must be installed (see
[`requirements.txt`](requirements.txt)).

### Extended datasets & placeholders

The ETL pipeline now captures additional datasets alongside prices and
fundamentals:

- Corporate actions (dividends, splits, earnings dates)
- Option surface metrics (mid price, implied volatility, volume/open interest)
- Daily aggregated sentiment signals derived from cached news headlines
- ESG sustainability scores
- Ownership and fund flow statistics

Data is sourced from `yfinance` when available. When an upstream provider does
not return data the refresh tasks still populate the corresponding tables with a
single placeholder row that documents the missing feed. These placeholders make
it easy to distinguish “no data yet” from “not refreshed” states and allow the
schema bootstrap to remain idempotent.

## Yahoo Finance rate limiting

The default provider registry now throttles Yahoo Finance requests to a
conservative pace of roughly one request every 40 seconds. This keeps automated
refresh jobs within the unofficial limits enforced by Yahoo and prevents long
cooldown periods triggered by HTTP 429 responses.

Override the default throttle by providing
`PredictorConfig.yahoo_rate_limit_per_second` or
`PredictorConfig.yahoo_rate_limit_per_minute` before constructing the
application/registry. The same values can be supplied via environment variables
(`YAHOO_RATE_LIMIT_PER_SECOND`, `YAHOO_RATE_LIMIT_PER_SEC`, or
`YAHOO_RATE_LIMIT_PER_MINUTE`). Cooldown behaviour can also be tuned through
`PredictorConfig.yahoo_cooldown_seconds` or the `YAHOO_COOLDOWN_SECONDS`
environment variable.

## Requirements

See [`requirements.txt`](requirements.txt) for the complete dependency list.
All packages are available on PyPI and can be installed with `pip`.

## License

This project is distributed under the terms of the MIT License. See the
[`LICENSE`](LICENSE) file for details.
