# Drift experiment summary

All numbers below were produced by `uv run python -m automl.experiments.run_drift`.

- Crops: ['Cabbage', 'Onion', 'Potato', 'Tomato']
- Seeds: [0, 1, 2, 3, 4]
- Windows: [1, 2, 3, 4, 5]
- Trial budget per search: 15

## Facts: test error by strategy (windows 2 onward)

| strategy     |   mae_mean |   mae_std |   rmse_mean |   rmse_std |   mape_mean |   trials_mean |   seconds_mean |   n_runs |
|:-------------|-----------:|----------:|------------:|-----------:|------------:|--------------:|---------------:|---------:|
| no_retrain   |     129.05 |     70.41 |      187.34 |      97.28 |        6.82 |             0 |           0    |       80 |
| refit        |     129.52 |     74.44 |      187.85 |      99.93 |        6.91 |             0 |           0.1  |       80 |
| cold_bo      |     129.29 |     71.29 |      187.43 |      97.08 |        6.91 |            15 |           2.37 |       80 |
| warm_bo_all  |     127.5  |     71.27 |      185.26 |      96.96 |        6.85 |            15 |           4.69 |       80 |
| warm_bo_topk |     127.76 |     70.79 |      185.94 |      96.87 |        6.85 |            15 |           2.49 |       80 |

## Facts: the same thing, split by crop

Pooling hides disagreement between crops, so this comes first.

| crop    |   no_retrain |   refit |   cold_bo |   warm_bo_all |   warm_bo_topk | best        |
|:--------|-------------:|--------:|----------:|--------------:|---------------:|:------------|
| Cabbage |         92.8 |    92.7 |      94.5 |          94.4 |           93.3 | refit       |
| Onion   |        144.5 |   137.9 |     135.6 |         131   |          132.4 | warm_bo_all |
| Potato  |         69.8 |    70   |      71.2 |          71.3 |           71.4 | no_retrain  |
| Tomato  |        209.2 |   217.6 |     215.9 |         213.2 |          213.9 | no_retrain  |

### Warm start against a plain refit, within each crop

| crop    |   n_pairs | warm_bo_all_wins   |   mean_mae_difference |   p_value | favours     | significant_after_holm   |
|:--------|----------:|:-------------------|----------------------:|----------:|:------------|:-------------------------|
| Onion   |        20 | 14/20              |                 -6.86 |    0.0266 | warm_bo_all | False                    |
| Tomato  |        20 | 14/20              |                 -4.3  |    0.0826 | warm_bo_all | False                    |
| Potato  |        20 | 8/20               |                  1.39 |    0.1893 | refit       | False                    |
| Cabbage |        20 | 9/20               |                  1.71 |    0.2774 | refit       | False                    |

## Facts: validation error reached by each search

This is what the tuner optimises. Test error is a separate question.

| strategy     |   mean |    std |   count |
|:-------------|-------:|-------:|--------:|
| cold_bo      | 141.33 | 113.73 |      80 |
| warm_bo_all  | 142.84 | 115.51 |      80 |
| warm_bo_topk | 141.77 | 115.2  |      80 |

## Facts: mean test MAE per window

|   window_id |   no_retrain |   refit |   cold_bo |   warm_bo_all |   warm_bo_topk |
|------------:|-------------:|--------:|----------:|--------------:|---------------:|
|           1 |        nan   |   nan   |     177.3 |         nan   |          nan   |
|           2 |        125.8 |   127.5 |     129.5 |         126.7 |          128.2 |
|           3 |        169.2 |   173.4 |     170.6 |         171.2 |          168.9 |
|           4 |        127.7 |   123.6 |     121.3 |         118.8 |          120.4 |
|           5 |         93.4 |    93.5 |      95.8 |          93.2 |           93.6 |

## Facts: head-to-head against the cold search

| strategy     | beat_cold_bo   |   mean_mae_difference |
|:-------------|:---------------|----------------------:|
| no_retrain   | 52/80          |                 -0.24 |
| refit        | 44/80          |                  0.22 |
| warm_bo_all  | 48/80          |                 -1.79 |
| warm_bo_topk | 46/80          |                 -1.53 |

## Facts: paired significance tests

Every strategy ran on the same seeds and the same windows, so each (seed, window) is a matched pair. A paired signed-rank test removes the shared difficulty of each window, which is what the large standard deviations above are mostly measuring. `wins` counts the pairs where the left strategy had the lower test MAE; a negative difference favours it.

| comparison                |   n_pairs | wins   |   mean_mae_difference |   median_mae_difference |   p_value | significant_after_holm   |
|:--------------------------|----------:|:-------|----------------------:|------------------------:|----------:|:-------------------------|
| cold_bo vs no_retrain     |        80 | 28/80  |                  0.24 |                    1.31 |    0.0146 | False                    |
| warm_bo_all vs cold_bo    |        80 | 48/80  |                 -1.79 |                   -0.46 |    0.0392 | False                    |
| warm_bo_topk vs cold_bo   |        80 | 46/80  |                 -1.53 |                   -0.98 |    0.0436 | False                    |
| warm_bo_all vs refit      |        80 | 45/80  |                 -2.02 |                   -0.37 |    0.1515 | False                    |
| warm_bo_all vs no_retrain |        80 | 33/80  |                 -1.55 |                    0.94 |    0.4176 | False                    |
| cold_bo vs refit          |        80 | 35/80  |                 -0.22 |                    0.49 |    0.5    | False                    |
| refit vs no_retrain       |        80 | 39/80  |                  0.47 |                    0.14 |    0.5019 | False                    |

Seven comparisons are made, so `significant_after_holm` is the column that matters. A raw p-value below 0.05 that does not survive correction is suggestive, not conclusive.

## Facts: trials needed to match the cold search's final validation score

**Read `cold_bo` with care.** The target *is* the cold search's own final score, so cold reaches it by definition in every run. Only the warm rows carry information; the cold row is there to show how long cold took to arrive at the score the warm runs are being asked to match.

| strategy     | reached_target   |   mean_trials_when_reached |   median_trials_when_reached |
|:-------------|:-----------------|---------------------------:|-----------------------------:|
| cold_bo      | 80/80            |                      10.51 |                         11.5 |
| warm_bo_all  | 26/80            |                       6.96 |                          6   |
| warm_bo_topk | 40/80            |                       7    |                          7   |

## Facts: drift detected at each window boundary

|   window_id |   n_drifted_features |   n_features_tested | input_drift   |   error_ratio | error_drift   |
|------------:|---------------------:|--------------------:|:--------------|--------------:|:--------------|
|           2 |                    9 |                   9 | True          |         1.646 | True          |
|           3 |                    9 |                   9 | True          |         1.289 | True          |
|           4 |                    9 |                   9 | True          |         0.898 | True          |
|           5 |                    9 |                   9 | True          |         0.74  | True          |

## Interpretation

- The lowest mean test MAE across post-drift windows came from **warm_bo_all** at 127.5 (SD 71.3) rupees per quintal.
- The plain refit baseline scored 129.5.
- When **warm_bo_all** did reach the cold search's final validation score, it needed a median of 6 trials, against the 12 the cold search itself took. It reached that bar in 26/80 runs.

Unpaired standard deviations here are roughly a third of the means and every strategy's interval overlaps every other's, which is why the paired table above is the one to read: it removes the window-difficulty variance that those intervals are mostly made of. Treat all of it as evidence about this dataset and this budget, not as a general finding.
