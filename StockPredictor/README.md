# Stock Predictor

The Stock Predictor project provides an end-to-end machine learning pipeline for
downloading financial market data, engineering features, training a predictive
model and generating forecasts. All functionality is exposed through a single
command line interface so the system can be automated or integrated into other
workflows easily.

## Features

- Download historical price data via [yfinance](https://github.com/ranaroussi/yfinance).
- Optional download of recent company news headlines from Financial Modeling
  Prep's public API (requires an API key, falls back to demo limits).
- Automatic sentiment scoring for news articles using VADER.
- Feature engineering with common technical indicators and aggregated
  sentiment information.
- Machine learning model (Random Forest Regressor) with persisted metrics and
  trained model artefacts.
- CLI modes for data collection, training and inference.

## Project layout

```
StockPredictor/
├── main.py                # Command line interface
├── stock_predictor/
│   ├── __init__.py
│   ├── config.py          # Runtime configuration helpers
│   ├── data_fetcher.py    # Internet data acquisition
│   ├── model.py           # StockPredictorAI implementation
│   ├── preprocessing.py   # Feature engineering helpers
│   └── sentiment.py       # Sentiment scoring utilities
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

This downloads price data (and news if available) and stores them inside the
`data/` directory. The `--refresh-data` flag forces re-download even when cached
files already exist.

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
- `--log-level DEBUG` enables more verbose logging for troubleshooting.

Run `python main.py --help` for the full set of options.

## Requirements

See [`requirements.txt`](requirements.txt) for the complete dependency list.
All packages are available on PyPI and can be installed with `pip`.

## License

This project is distributed under the terms of the MIT License. See the
[`LICENSE`](LICENSE) file for details.
