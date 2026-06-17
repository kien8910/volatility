"""Look-back sequence plus controlled news embedding tests for spike days.

This module is intentionally separate from the existing pilot and controlled
embedding scripts. It switches the time-series side from loose HAR/lag features
to an explicit fixed look-back window, then adds selected text embeddings on top
of that same window.

Example:

    python -m src.spike_lookback_embedding \
      --num_tickers 25 \
      --tickers BAC C AMGN AXP AMD \
      --look_back 22 \
      --targets log_abs_return log_gk \
      --forecast_horizons 1 3 \
      --spike_percentiles 80 85 90 95 \
      --embedding_variants target_sector_weighted target_sector_event_weighted target_delta_20d \
      --alignment_modes shifted_1_day_text same_day_text \
      --require_target_event_keyword \
      --output_dir outputs/spike_lookback_embedding_candidate
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import DEFAULT_CACHE_DIR, DEFAULT_EMBEDDING_MODEL, RANDOM_SEED
from .detailed_news_signal import split_by_time
from .models import make_ridge_model
from .spike_controlled_embedding import (
    build_controlled_embedding,
    summarize_placebo,
)
from .spike_news_embedding_signal import (
    PLACEBO_VARIANTS,
    prepare_data,
    train_only_pca,
)
from .text_features import add_event_keyword_features, add_text_presence_flags, normalize_text_value


DEFAULT_LOOKBACK_VARIANTS = [
    "target_sector_weighted",
    "target_sector_event_weighted",
    "target_delta_20d",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run look-back window plus text embedding tests on volatility spike days.")
    parser.add_argument("--num_tickers", type=int, default=25)
    parser.add_argument("--tickers", nargs="*", default=["BAC", "C", "AMGN", "AXP", "AMD"])
    parser.add_argument("--look_back", type=int, default=22)
    parser.add_argument("--forecast_horizons", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--targets", nargs="+", default=["log_abs_return", "log_gk"])
    parser.add_argument("--spike_percentiles", nargs="+", type=float, default=[80.0, 85.0, 90.0, 95.0])
    parser.add_argument("--alignment_modes", nargs="+", default=["shifted_1_day_text", "same_day_text"])
    parser.add_argument("--embedding_variants", nargs="+", default=DEFAULT_LOOKBACK_VARIANTS)
    parser.add_argument("--placebo_variants", nargs="+", default=["correct_text"])
    parser.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding_device", default="auto")
    parser.add_argument("--pca_dim", type=int, default=32)
    parser.add_argument("--delta_window", type=int, default=20)
    parser.add_argument("--target_weight", type=float, default=2.0)
    parser.add_argument("--sector_weight", type=float, default=1.0)
    parser.add_argument("--max_lag", type=int, default=None, help="Internal lag build; defaults to look_back.")
    parser.add_argument("--cache_dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--output_dir", default="outputs/spike_lookback_embedding")
    parser.add_argument("--run_placebo", action="store_true", help="Append shuffled/cross-ticker/stale placebo variants.")
    parser.add_argument(
        "--require_target_event_keyword",
        action="store_true",
        help="Evaluate only test rows where target-company text contains at least one event keyword.",
    )
    return parser.parse_args()


def _add_lookback_window(df: pd.DataFrame, ticker_col: str, date_col: str, look_back: int) -> tuple[pd.DataFrame, list[str]]:
    work = df.sort_values([ticker_col, date_col]).copy()
    grouped = work.groupby(ticker_col, sort=False)
    window_cols: list[str] = []
    for offset in range(look_back):
        col = f"lookback_logGKVol_t_minus_{offset}"
        work[col] = grouped["logGKVol"].shift(offset)
        window_cols.append(col)
    work["lookback_complete"] = work[window_cols].notna().all(axis=1)
    return work.sort_index(), window_cols


def _fit_predict_ridge(train: pd.DataFrame, test: pd.DataFrame, features: list[str], ticker_col: str, target_col: str) -> pd.Series:
    model = make_ridge_model(features, [ticker_col])
    model.fit(train[features + [ticker_col]], train[target_col])
    return pd.Series(model.predict(test[features + [ticker_col]]), index=test.index)


def _classification_metrics(y_true: pd.Series, score: pd.Series) -> dict[str, float]:
    if y_true.nunique() < 2:
        return {"precision": np.nan, "recall": np.nan, "f1": np.nan, "roc_auc": np.nan, "pr_auc": np.nan}
    pred = score >= 0.5
    out = {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": np.nan,
        "pr_auc": np.nan,
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, score))
        out["pr_auc"] = float(average_precision_score(y_true, score))
    except Exception:
        pass
    return out


def _classifier_score(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> pd.Series | None:
    if train["is_spike"].nunique() < 2 or test["is_spike"].nunique() < 2:
        return None
    clf = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logit",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                    solver="liblinear",
                ),
            ),
        ]
    )
    clf.fit(train[features], train["is_spike"].astype(int))
    return pd.Series(clf.predict_proba(test[features])[:, 1], index=test.index)


def _text_preview(df: pd.DataFrame, text_columns: dict[str, list[str]], idx: int) -> str:
    parts: list[str] = []
    for group in ["target", "sector"]:
        for column in text_columns.get(group, []):
            if column in df.columns:
                text = normalize_text_value(df.loc[idx, column])
                if text:
                    parts.append(text.replace("\n", " ")[:180])
    return " | ".join(parts)[:360]


def evaluate_lookback_variant(
    work: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    specs,
    lookback_cols: list[str],
    embedding_cols: list[str],
    args: argparse.Namespace,
    embedding_variant: str,
    alignment_mode: str,
    placebo_variant: str,
    text_columns: dict[str, list[str]],
) -> tuple[list[dict], list[dict], list[dict]]:
    metric_rows: list[dict] = []
    classifier_rows: list[dict] = []
    event_rows: list[dict] = []
    text_feature_cols = lookback_cols + embedding_cols

    for spec in specs:
        tr = train.dropna(subset=lookback_cols + text_feature_cols + [spec.column]).copy()
        te = test.dropna(subset=lookback_cols + text_feature_cols + [spec.column]).copy()
        if args.require_target_event_keyword:
            te = te[te["target_event_keyword_count"].fillna(0).gt(0)]
        if tr.empty or te.empty:
            continue

        lookback_pred = _fit_predict_ridge(tr, te, lookback_cols, args._ticker_col, spec.column)
        text_pred = _fit_predict_ridge(tr, te, text_feature_cols, args._ticker_col, spec.column)

        for pct in args.spike_percentiles:
            thresholds = tr.groupby(args._ticker_col)[spec.column].quantile(pct / 100.0).rename("threshold").to_frame()
            scored = te[[args._ticker_col, args._date_col, spec.column]].copy()
            scored["lookback_only_prediction"] = lookback_pred
            scored["lookback_text_prediction"] = text_pred
            scored = scored.merge(thresholds, left_on=args._ticker_col, right_index=True, how="left")
            scored["is_spike"] = scored[spec.column] > scored["threshold"]
            scored["lookback_abs_error"] = (scored["lookback_only_prediction"] - scored[spec.column]).abs()
            scored["lookback_text_abs_error"] = (scored["lookback_text_prediction"] - scored[spec.column]).abs()
            scored["abs_error_improvement"] = scored["lookback_abs_error"] - scored["lookback_text_abs_error"]
            scored["underprediction_reduction"] = (
                (scored[spec.column] - scored["lookback_only_prediction"]).clip(lower=0)
                - (scored[spec.column] - scored["lookback_text_prediction"]).clip(lower=0)
            )

            tr_cls = tr.merge(thresholds, left_on=args._ticker_col, right_index=True, how="left").copy()
            te_cls = te.merge(thresholds, left_on=args._ticker_col, right_index=True, how="left").copy()
            tr_cls["is_spike"] = tr_cls[spec.column] > tr_cls["threshold"]
            te_cls["is_spike"] = te_cls[spec.column] > te_cls["threshold"]
            for feature_set_name, feature_set in {
                "lookback_only": lookback_cols,
                "lookback_plus_text": text_feature_cols,
            }.items():
                score = _classifier_score(tr_cls, te_cls, feature_set)
                if score is None:
                    continue
                for ticker, idx in te_cls.groupby(args._ticker_col, sort=False).groups.items():
                    idx_list = list(idx)
                    classifier_rows.append(
                        {
                            "ticker": ticker,
                            "target_name": spec.target_name,
                            "horizon": spec.horizon,
                            "spike_percentile": pct,
                            "feature_set": feature_set_name,
                            "embedding_variant": embedding_variant,
                            "alignment_mode": alignment_mode,
                            "placebo_variant": placebo_variant,
                            "evaluation_filter": "target_event_keyword" if args.require_target_event_keyword else "all_test_rows",
                            "num_test_spikes": int(te_cls.loc[idx_list, "is_spike"].sum()),
                            "num_test_samples": int(len(idx_list)),
                            **_classification_metrics(te_cls.loc[idx_list, "is_spike"].astype(int), score.loc[idx_list]),
                        }
                    )

            for segment, mask in {
                "all": pd.Series(True, index=scored.index),
                "spike": scored["is_spike"],
                "non_spike": ~scored["is_spike"],
            }.items():
                subset = scored[mask]
                if subset.empty:
                    continue
                for ticker, group in subset.groupby(args._ticker_col, sort=False):
                    base_mae = float(group["lookback_abs_error"].mean())
                    text_mae = float(group["lookback_text_abs_error"].mean())
                    metric_rows.append(
                        {
                            "ticker": ticker,
                            "target_name": spec.target_name,
                            "horizon": spec.horizon,
                            "spike_percentile": pct,
                            "segment": segment,
                            "embedding_variant": embedding_variant,
                            "alignment_mode": alignment_mode,
                            "placebo_variant": placebo_variant,
                            "evaluation_filter": "target_event_keyword" if args.require_target_event_keyword else "all_test_rows",
                            "num_samples": int(len(group)),
                            "lookback_only_mae": base_mae,
                            "lookback_text_mae": text_mae,
                            "mae_improvement_pct": (base_mae - text_mae) / base_mae * 100 if base_mae else np.nan,
                            "underprediction_reduction": float(group["underprediction_reduction"].mean()),
                        }
                    )

            if placebo_variant == "correct_text":
                for idx, row in scored[scored["is_spike"]].iterrows():
                    event_rows.append(
                        {
                            "ticker": row[args._ticker_col],
                            "date": row[args._date_col],
                            "target_name": spec.target_name,
                            "horizon": spec.horizon,
                            "spike_percentile": pct,
                            "embedding_variant": embedding_variant,
                            "alignment_mode": alignment_mode,
                            "evaluation_filter": "target_event_keyword" if args.require_target_event_keyword else "all_test_rows",
                            "actual_target": row[spec.column],
                            "threshold": row["threshold"],
                            "lookback_only_prediction": row["lookback_only_prediction"],
                            "lookback_text_prediction": row["lookback_text_prediction"],
                            "abs_error_improvement": row["abs_error_improvement"],
                            "underprediction_reduction": row["underprediction_reduction"],
                            "target_event_keyword_count": int(work.loc[idx].get("target_event_keyword_count", 0)),
                            "text_preview": _text_preview(work, text_columns, idx),
                        }
                    )
    return metric_rows, classifier_rows, event_rows


def _lookback_placebo_ranks(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    spike = metrics[metrics["segment"].eq("spike")].copy()
    if spike.empty:
        return pd.DataFrame()
    rank_input = spike.rename(
        columns={
            "lookback_text_mae": "embedding_mae",
            "num_samples": "num_spike_samples",
        }
    )
    return summarize_placebo(rank_input)


def write_summary(output_dir: Path, metrics: pd.DataFrame, ranks: pd.DataFrame, classifier: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = ["# Look-Back Spike Embedding Summary", ""]
    lines.append(f"- look_back: {args.look_back}")
    lines.append(f"- evaluation_filter: {'target_event_keyword' if args.require_target_event_keyword else 'all_test_rows'}")
    if not metrics.empty:
        correct = metrics[(metrics["segment"].eq("spike")) & (metrics["placebo_variant"].eq("correct_text"))]
        lines.extend(["", "## Spike Regression", ""])
        lines.append(f"- Rows: {len(correct)}")
        lines.append(f"- Positive MAE improvement rate: {(correct['mae_improvement_pct'] > 0).mean() * 100:.2f}%")
        lines.append(f"- Mean MAE improvement: {correct['mae_improvement_pct'].mean():.2f}%")
        lines.append(f"- Median MAE improvement: {correct['mae_improvement_pct'].median():.2f}%")
        lines.extend(["", "## By Variant", ""])
        by_variant = correct.groupby("embedding_variant")["mae_improvement_pct"].agg(["count", "mean", "median"]).sort_values("median", ascending=False)
        for name, row in by_variant.iterrows():
            lines.append(f"- {name}: mean={row['mean']:.2f}%, median={row['median']:.2f}% over {int(row['count'])} cases")
        lines.extend(["", "## By Target", ""])
        by_target = correct.groupby(["target_name", "horizon"])["mae_improvement_pct"].agg(["count", "mean", "median"]).sort_values("median", ascending=False)
        for (target, horizon), row in by_target.iterrows():
            lines.append(f"- {target} h={horizon}: mean={row['mean']:.2f}%, median={row['median']:.2f}% over {int(row['count'])} cases")
    if not ranks.empty:
        lines.extend(["", "## Placebo Rank", ""])
        lines.append(f"- Correct text ranked best in {(ranks['correct_text_rank'] == 1).sum()} of {len(ranks)} comparable spike cases.")
    if not classifier.empty:
        correct_cls = classifier[
            (classifier["placebo_variant"].eq("correct_text"))
            & (classifier["feature_set"].eq("lookback_plus_text"))
        ]
        lines.extend(["", "## Spike Classifier", ""])
        lines.append(f"- Mean PR-AUC with text: {correct_cls['pr_auc'].mean():.4f}")
        lines.append(f"- Median PR-AUC with text: {correct_cls['pr_auc'].median():.4f}")
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- The look-back window contains current-day logGKVol and prior observations only.",
            "- Spike thresholds are computed from train rows per ticker and then applied to test rows.",
            "- PCA is fit on train rows only.",
            "- This script compares look-back-only against look-back-plus-text; it does not gate non-spike rows to a separate production model.",
        ]
    )
    (output_dir / "lookback_embedding_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    args.max_lag = args.max_lag or args.look_back
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, _, _, specs, column_map, _, text_columns = prepare_data(args)
    if args.tickers:
        keep = set(args.tickers)
        df = df[df[column_map.ticker].isin(keep)].copy()
    df = add_text_presence_flags(df, text_columns, ticker_col=column_map.ticker)
    df = add_event_keyword_features(df, text_columns)
    df, lookback_cols = _add_lookback_window(df, column_map.ticker, column_map.date, args.look_back)
    df = df[df["lookback_complete"]].copy().reset_index(drop=True)

    train_part, val_part, test = split_by_time(df, column_map.ticker, column_map.date)
    train = pd.concat([train_part, val_part], ignore_index=False)
    args._ticker_col = column_map.ticker
    args._date_col = column_map.date

    variants = list(args.placebo_variants)
    if args.run_placebo:
        variants = sorted(set(variants + PLACEBO_VARIANTS), key=(variants + PLACEBO_VARIANTS).index)

    metric_rows: list[dict] = []
    classifier_rows: list[dict] = []
    event_rows: list[dict] = []
    for embedding_variant in args.embedding_variants:
        for alignment_mode in args.alignment_modes:
            for placebo_variant in variants:
                raw = build_controlled_embedding(df, text_columns, embedding_variant, alignment_mode, placebo_variant, args)
                emb_df, emb_cols = train_only_pca(raw, train.index, args.pca_dim, f"{embedding_variant}_{alignment_mode}_{placebo_variant}")
                work = pd.concat([df.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)
                train_work = work.loc[train.index]
                test_work = work.loc[test.index]
                rows, cls_rows, ev_rows = evaluate_lookback_variant(
                    work,
                    train_work,
                    test_work,
                    specs,
                    lookback_cols,
                    emb_cols,
                    args,
                    embedding_variant,
                    alignment_mode,
                    placebo_variant,
                    text_columns,
                )
                metric_rows.extend(rows)
                classifier_rows.extend(cls_rows)
                event_rows.extend(ev_rows)

    metrics = pd.DataFrame(metric_rows)
    classifier = pd.DataFrame(classifier_rows)
    events = pd.DataFrame(event_rows)
    ranks = _lookback_placebo_ranks(metrics)

    metrics.to_csv(output_dir / "lookback_embedding_metrics.csv", index=False)
    classifier.to_csv(output_dir / "lookback_embedding_classifier_metrics.csv", index=False)
    ranks.to_csv(output_dir / "lookback_embedding_placebo_ranks.csv", index=False)
    if not events.empty:
        events.to_csv(output_dir / "lookback_embedding_event_examples.csv", index=False)
        events.sort_values("abs_error_improvement", ascending=False).head(300).to_csv(output_dir / "lookback_embedding_top_improvements.csv", index=False)
        events.sort_values("abs_error_improvement", ascending=True).head(300).to_csv(output_dir / "lookback_embedding_top_failures.csv", index=False)
    write_summary(output_dir, metrics, ranks, classifier, args)
    print(f"Look-back spike embedding outputs written to: {output_dir.resolve()}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
