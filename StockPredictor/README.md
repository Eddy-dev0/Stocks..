# Stock Predictor

The Stock Predictor project provides an end-to-end machine learning pipeline for
downloading financial market data, engineering features, training a predictive
model and generating forecasts. The modernised architecture centres around the
``StockPredictorApplication`` orchestrator which wires together the analytical
core, external data providers, research extensions and user interfaces. A
backwards-compatible (but deprecated) command line entry point remains available
for automation scenarios while new FastAPI helpers power web or Streamlit
front-ends.

## Features

- Download historical price data via [yfinance](https://github.com/ranaroussi/yfinance).
- Optional download of recent company news headlines from Financial Modeling
  Prep's public API (requires an API key, falls back to demo limits).
- Automatic sentiment scoring for news articles using VADER.
- Feature engineering with common technical indicators and aggregated
  sentiment information.
- Machine learning model (Random Forest Regressor) with persisted metrics and
  trained model artefacts.
- Local SQLite database that stores prices, indicators, fundamentals and news
  for fast reuse across runs.
- Extended database coverage for corporate events, options analytics,
  sentiment-derived feeds, ESG metrics and ownership/flow datasets (provider
  responses are cached when available and fall back to documented placeholders
  when upstream APIs return no data).
- CLI modes for data collection, training and inference.

## Project layout

```
StockPredictor/
├── main.py                # Deprecated CLI shim using the new orchestrator
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

### Download data

```bash
python main.py --mode download-data --ticker TSLA --start-date 2022-01-01 --refresh-data
```

This downloads price data (plus indicators, fundamentals, macro metrics and
news when available) and stores them in the SQLite database at
`data/market_data.sqlite`. The `--refresh-data` flag forces a re-download even
when the requested window already exists in the database.

### Train a model

```bash
python main.py --mode train --ticker TSLA --start-date 2022-01-01 --news-limit 100
```

This prepares features, trains a Random Forest model and saves both the model
(`models/TSLA_random_forest.joblib`) and metrics
(`models/TSLA_random_forest_metrics.json`).

### Generate a prediction

```bash
python main.py --mode predict --ticker TSLA
```

The command loads the most recent trained model, refreshes features if needed
and returns a JSON blob with the predicted closing price for the next trading
day, the absolute and percentage change relative to the latest observed close
and metadata about the run.

Running `python main.py` without arguments uses defaults. Set
`STOCK_PREDICTOR_DEFAULT_MODE` and `STOCK_PREDICTOR_DEFAULT_TICKER` in your
environment (or `.env`) to customise the implicit `mode` and `ticker`. The
application falls back to `predict` for the mode and `AAPL` for the ticker when
no overrides are provided.

### Additional options

- `--no-sentiment` disables sentiment processing even when news is available.
- `--data-dir` and `--models-dir` allow custom storage locations.
- `--database-url` overrides the default SQLite connection string. Any
  SQLAlchemy-compatible URL is accepted, e.g.
  `sqlite:////tmp/stock_predictor.sqlite` or `postgresql://user:pass@host/db`.
- `--log-level DEBUG` enables more verbose logging for troubleshooting.

Run `python main.py --help` for the full set of options.

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

## Requirements

See [`requirements.txt`](requirements.txt) for the complete dependency list.
All packages are available on PyPI and can be installed with `pip`.

## License

This project is distributed under the terms of the MIT License. See the
[`LICENSE`](LICENSE) file for details.
