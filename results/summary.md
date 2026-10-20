# Drift experiment summary

All numbers below were produced by `uv run python -m automl.experiments.run_drift`.

- Seeds: [0, 1, 2, 3, 4]
- Windows: [1, 2, 3, 4, 5]
- Trial budget per search: 15

## Facts: test error by strategy (windows 2 onward)

| strategy     |   mae_mean |   mae_std |   rmse_mean |   rmse_std |   mape_mean |   trials_mean |   seconds_mean |   n_runs |
|:-------------|-----------:|----------:|------------:|-----------:|------------:|--------------:|---------------:|---------:|
| no_retrain   |     144.45 |     45.49 |      210.67 |      76.27 |        7.01 |             0 |           0    |       20 |
| refit        |     137.9  |     37.74 |      204.82 |      69.36 |        6.84 |             0 |           0.07 |       20 |
| cold_bo      |     135.63 |     38.01 |      202.28 |      72.3  |        6.7  |            15 |           2.49 |       20 |
| warm_bo_all  |     131.04 |     37.04 |      195.22 |      70.96 |        6.6  |            15 |           7.14 |       20 |
| warm_bo_topk |     132.4  |     37.87 |      198.1  |      73.3  |        6.61 |            15 |           2.68 |       20 |

## Facts: validation error reached by each search

This is what the tuner optimises. Test error is a separate question.

| strategy     |   mean |   std |   count |
|:-------------|-------:|------:|--------:|
| cold_bo      | 121.82 | 41.43 |      20 |
| warm_bo_all  | 123.79 | 42.04 |      20 |
| warm_bo_topk | 121.93 | 41.26 |      20 |

## Facts: mean test MAE per window

|   window_id |   no_retrain |   refit |   cold_bo |   warm_bo_all |   warm_bo_topk |
|------------:|-------------:|--------:|----------:|--------------:|---------------:|
|           1 |        nan   |   nan   |      59.7 |         nan   |          nan   |
|           2 |        168.6 |   167.3 |     171.9 |         170.3 |          170   |
|           3 |        151.1 |   142.3 |     138.3 |         134.2 |          129.6 |
|           4 |        184.7 |   164.4 |     157   |         146.5 |          155.6 |
|           5 |         73.4 |    77.6 |      75.3 |          73.1 |           74.4 |

## Facts: head-to-head against the cold search

| strategy     | beat_cold_bo   |   mean_mae_difference |
|:-------------|:---------------|----------------------:|
| no_retrain   | 8/20           |                  8.82 |
| refit        | 8/20           |                  2.27 |
| warm_bo_all  | 14/20          |                 -4.59 |
| warm_bo_topk | 12/20          |                 -3.23 |

## Facts: paired significance tests

Every strategy ran on the same seeds and the same windows, so each (seed, window) is a matched pair. A paired signed-rank test removes the shared difficulty of each window, which is what the large standard deviations above are mostly measuring. `wins` counts the pairs where the left strategy had the lower test MAE; a negative difference favours it.

| comparison                |   n_pairs | wins   |   mean_mae_difference |   median_mae_difference |   p_value | significant_after_holm   |
|:--------------------------|----------:|:-------|----------------------:|------------------------:|----------:|:-------------------------|
| warm_bo_all vs cold_bo    |        20 | 14/20  |                 -4.59 |                   -3.45 |    0.0266 | False                    |
| warm_bo_all vs refit      |        20 | 14/20  |                 -6.86 |                   -4.83 |    0.0266 | False                    |
| warm_bo_all vs no_retrain |        20 | 12/20  |                -13.41 |                   -4.4  |    0.04   | False                    |
| warm_bo_topk vs refit     |        20 | 12/20  |                 -5.5  |                   -6.21 |    0.0583 | False                    |
| refit vs no_retrain       |        20 | 13/20  |                 -6.55 |                   -1.48 |    0.1231 | False                    |
| warm_bo_topk vs cold_bo   |        20 | 12/20  |                 -3.23 |                   -2.28 |    0.1536 | False                    |
| cold_bo vs refit          |        20 | 11/20  |                 -2.27 |                   -0.88 |    0.4445 | False                    |

Seven comparisons are made, so `significant_after_holm` is the column that matters. A raw p-value below 0.05 that does not survive correction is suggestive, not conclusive.

## Facts: trials needed to match the cold search's final validation score

**Read `cold_bo` with care.** The target *is* the cold search's own final score, so cold reaches it by definition in every run. Only the warm rows carry information; the cold row is there to show how long cold took to arrive at the score the warm runs are being asked to match.

| strategy     | reached_target   |   mean_trials_when_reached |   median_trials_when_reached |
|:-------------|:-----------------|---------------------------:|-----------------------------:|
| cold_bo      | 20/20            |                      11.35 |                           12 |
| warm_bo_all  | 7/20             |                       9    |                           11 |
| warm_bo_topk | 11/20            |                       7.27 |                            9 |

## Facts: drift detected at each window boundary

|   window_id |   n_drifted_features |   n_features_tested | input_drift   |   error_ratio | error_drift   |
|------------:|---------------------:|--------------------:|:--------------|--------------:|:--------------|
|           2 |                    9 |                   9 | True          |         2.824 | True          |
|           3 |                    9 |                   9 | True          |         0.897 | False         |
|           4 |                    7 |                   9 | True          |         1.221 | True          |
|           5 |                    9 |                   9 | True          |         0.402 | False         |

## Interpretation

- The lowest mean test MAE across post-drift windows came from **warm_bo_all** at 131.0 (SD 37.0) rupees per quintal.
- The plain refit baseline scored 137.9.
- When **warm_bo_topk** did reach the cold search's final validation score, it needed a median of 9 trials, against the 12 the cold search itself took. It reached that bar in 11/20 runs.

Unpaired standard deviations here are roughly a third of the means and every strategy's interval overlaps every other's, which is why the paired table above is the one to read: it removes the window-difficulty variance that those intervals are mostly made of. Treat all of it as evidence about this dataset and this budget, not as a general finding.
