# FinTexTS Volatility Forecasting Pilot

This pilot checks whether text fields in `EXAONE-BI/FinTexTS` improve next-day stock volatility forecasts over time-series-only baselines.

The target is next-day log Garman-Klass volatility computed from OHLC prices. The study compares naive/HAR/time-series lag baselines against time-series features augmented with SentenceTransformer text embeddings.

No paid APIs are used. The default embedding model runs locally on CPU:

`sentence-transformers/all-MiniLM-L6-v2`

## Install

```bash
pip install -r requirements.txt
```

`lightgbm` is listed as an optional preferred model backend. If it is not importable, the code falls back to scikit-learn's `HistGradientBoostingRegressor`, and then to `RandomForestRegressor` if needed.

## Run

```bash
python -m src.run_pilot --num_tickers 10
```

Fast smoke run:

```bash
python -m src.run_pilot --num_tickers 3 --pca_dim 32
```

Useful options:

```bash
python -m src.run_pilot \
  --num_tickers 10 \
  --embedding_model sentence-transformers/all-MiniLM-L6-v2 \
  --pca_dim 64 \
  --forecast_horizon 1 \
  --max_lag 22 \
  --output_dir outputs
```

## Outputs

The pipeline writes:

- `outputs/results.csv`
- `outputs/predictions.csv`
- `outputs/error_by_ticker.csv`
- `outputs/event_day_analysis.csv`
- `outputs/model_comparison_rmse.png`
- `outputs/model_comparison_mae.png`
- `outputs/example_predictions_<ticker>.png`

## Reading Results

If TS + text models reduce MAE or RMSE compared with `TS_only_lags`, FinTexTS text likely contains useful volatility signal.

If `all_text` performs worse than `target_sector_text`, broad text context may be noisy and future work should filter text more aggressively.

If text does not improve results, it is probably premature to build a full Reasoning Agent or Reflection Agent around this dataset before improving alignment, filtering, or target construction.

## Data And Column Mapping

The loader automatically detects ticker, date, open, high, low, close, and text columns. If detection fails, it prints available columns and raises an error. Override mappings in `src/config.py`.

## Validation Notes

Run syntax checks with:

```bash
python -m py_compile src/*.py
```

The full pilot requires internet access to download the Hugging Face dataset and the embedding model unless both are already cached locally. It also requires the packages in `requirements.txt`. GPU is not required.
