# Controlled Spike Embedding Summary

## Correct Text Regression

- Rows: 436
- Positive MAE improvement rate: 83.72%
- Mean MAE improvement: 4.77%
- Median MAE improvement: 2.99%

## By Embedding Variant

- target_sector_weighted: mean=5.36%, median=3.64% over 218 cases
- target_sector_event_weighted: mean=4.17%, median=2.29% over 218 cases

## Placebo Rank

- Correct text ranked best in 238 of 436 cases.
- target_sector_weighted: correct-best rate 62.84%
- target_sector_event_weighted: correct-best rate 46.33%

## Classifier

- Mean PR-AUC: 0.1783
- Median PR-AUC: 0.1483

## Decision Rule

- Treat a variant as promising only if correct text improves spike MAE, reduces underprediction, and beats placebo in at least half of comparable ticker/configuration cases.
- Prefer shifted alignment when it remains competitive with same-day alignment.
- Do not claim causal effect; this is predictive signal analysis.
