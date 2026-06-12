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

## Detailed News Signal Analysis

The detailed mode checks whether FinTexTS news has predictive association in specific tickers, event days, text levels, targets, and horizons. It keeps the original pilot outputs and adds ticker-level metrics, text-level ablations, event/spike diagnostics, placebo tests, alignment sensitivity, and an automatic report.

Quick 5-ticker run:

```bash
python -m src.run_pilot \
  --num_tickers 5 \
  --analysis_mode detailed_news_signal \
  --forecast_horizons 1 3 \
  --targets log_gk total_volatility \
  --quick_mode \
  --run_placebo_tests \
  --run_event_keyword_filter \
  --output_dir outputs/detailed_signal
```

Fuller diagnostic run:

```bash
python -m src.run_pilot \
  --num_tickers 5 \
  --analysis_mode detailed_news_signal \
  --forecast_horizons 1 2 3 5 \
  --targets log_gk total_volatility log_abs_return \
  --evaluation_mode fixed \
  --run_placebo_tests \
  --run_event_keyword_filter \
  --run_statistical_tests \
  --output_dir outputs/detailed_signal
```

Detailed mode writes stable-schema outputs including:

- `outputs/detailed_signal/news_signal_by_ticker.csv`
- `outputs/detailed_signal/ticker_improvement_summary.csv`
- `outputs/detailed_signal/text_level_ablation.csv`
- `outputs/detailed_signal/event_day_forecast_comparison.csv`
- `outputs/detailed_signal/event_day_volatility_statistics.csv`
- `outputs/detailed_signal/volatility_spike_analysis.csv`
- `outputs/detailed_signal/news_lead_lag_analysis.csv`
- `outputs/detailed_signal/text_signal_regression.csv`
- `outputs/detailed_signal/event_keyword_filter_results.csv`
- `outputs/detailed_signal/text_intensity_analysis.csv`
- `outputs/detailed_signal/text_placebo_tests.csv`
- `outputs/detailed_signal/text_alignment_sensitivity.csv`
- `outputs/detailed_signal/statistical_tests.csv`
- `outputs/detailed_signal/walk_forward_results.csv`
- `outputs/detailed_signal/news_signal_report.md`

The report gives a conservative recommendation: `proceed_with_agents`, `run_larger_pilot`, `revise_target_or_alignment`, or `stop_agent_development`. It does not claim that news helps unless the diagnostics support that.

## Outputs

The pipeline writes:

- `outputs/results.csv`
- `outputs/predictions.csv`
- `outputs/error_by_ticker.csv`
- `outputs/event_day_analysis.csv`
- `outputs/model_comparison_rmse.png`
- `outputs/model_comparison_mae.png`
- `outputs/example_predictions_<ticker>.png`

Detailed mode also creates:

- `news_signal_by_ticker_mae.png`
- `text_level_ablation_heatmap.png`
- `event_day_error_comparison.png`
- `spike_error_comparison.png`
- `spike_rate_by_text_level.png`
- `spike_prediction_metrics.png`
- `news_lead_lag_heatmap.png`
- `correct_vs_placebo_text.png`
- `target_comparison.png`
- `example_event_predictions_<ticker>.png`

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
