# Drift experiment summary

All numbers below were produced by `uv run python -m automl.experiments.run_drift`.

- Seeds: [0, 1, 2, 3, 4]
- Windows: [1, 2, 3, 4, 5]
- Trial budget per search: 15

## Facts: test error by strategy (windows 2 onward)

| strategy     |   mae_mean |   mae_std |   rmse_mean |   rmse_std |   mape_mean |   trials_mean |   seconds_mean |   n_runs |
|:-------------|-----------:|----------:|------------:|-----------:|------------:|--------------:|---------------:|---------:|
| no_retrain   |     157.25 |     45.21 |      228.48 |      83.9  |        6.97 |             0 |           0    |       20 |
| refit        |     153.41 |     43.04 |      223.11 |      82.57 |        6.89 |             0 |           0.03 |       20 |
| cold_bo      |     160.73 |     45.79 |      233.78 |      89.07 |        7.28 |            15 |           1.87 |       20 |
| warm_bo_all  |     160.43 |     47.17 |      235.22 |      92.75 |        7.2  |            15 |           3.39 |       20 |
| warm_bo_topk |     159.61 |     46.59 |      232.81 |      90.19 |        7.15 |            15 |           1.84 |       20 |

## Facts: validation error reached by each search

This is what the tuner optimises. Test error is a separate question.

| strategy     |   mean |   std |   count |
|:-------------|-------:|------:|--------:|
| cold_bo      | 144.75 | 50.16 |      20 |
| warm_bo_all  | 145.38 | 50.95 |      20 |
| warm_bo_topk | 144.63 | 49.59 |      20 |

## Facts: mean test MAE per window

|   window_id |   no_retrain |   refit |   cold_bo |   warm_bo_all |   warm_bo_topk |
|------------:|-------------:|--------:|----------:|--------------:|---------------:|
|           1 |        nan   |   nan   |      72.2 |         nan   |          nan   |
|           2 |        200.9 |   200.5 |     217.2 |         218.8 |          212.8 |
|           3 |        155.9 |   146.9 |     147.9 |         148.2 |          148.7 |
|           4 |        185.5 |   177.5 |     180.7 |         180   |          184.2 |
|           5 |         86.7 |    88.8 |      97.2 |          94.7 |           92.7 |

## Facts: head-to-head against the cold search

| strategy     | beat_cold_bo   |   mean_mae_difference |
|:-------------|:---------------|----------------------:|
| no_retrain   | 12/20          |                 -3.48 |
| refit        | 15/20          |                 -7.32 |
| warm_bo_all  | 13/20          |                 -0.3  |
| warm_bo_topk | 11/20          |                 -1.11 |

## Facts: trials needed to match the cold search's final validation score

**Read `cold_bo` with care.** The target *is* the cold search's own final score, so cold reaches it by definition in every run. Only the warm rows carry information; the cold row is there to show how long cold took to arrive at the score the warm runs are being asked to match.

| strategy     | reached_target   |   mean_trials_when_reached |   median_trials_when_reached |
|:-------------|:-----------------|---------------------------:|-----------------------------:|
| cold_bo      | 20/20            |                      10.65 |                         11   |
| warm_bo_all  | 9/20             |                       7.11 |                          4   |
| warm_bo_topk | 10/20            |                       8.9  |                          8.5 |

## Facts: drift detected at each window boundary

|   window_id |   n_drifted_features |   n_features_tested | input_drift   |   error_ratio | error_drift   |
|------------:|---------------------:|--------------------:|:--------------|--------------:|:--------------|
|           2 |                    9 |                   9 | True          |         2.783 | True          |
|           3 |                    7 |                   9 | True          |         0.776 | False         |
|           4 |                    4 |                   9 | True          |         1.19  | True          |
|           5 |                    9 |                   9 | True          |         0.468 | False         |

## Interpretation

- The lowest mean test MAE across post-drift windows came from **refit** at 153.4 (SD 43.0) rupees per quintal.
- The plain refit baseline scored 153.4.
- When **warm_bo_all** did reach the cold search's final validation score, it needed a median of 4 trials, against the 11 the cold search itself took. It reached that bar in 9/20 runs.

Standard deviations here are roughly a third of the means, and every strategy's interval overlaps every other's. Treat the ordering as weak evidence about this dataset and this budget, not as a general finding.
