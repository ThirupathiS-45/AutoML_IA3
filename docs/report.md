# Drift-Aware AutoML for Indian Crop Price Forecasting

**An empirical evaluation of warm-started Bayesian optimization after real market drift**

*Generated from the committed result CSVs by
`uv run python -m automl.experiments.make_report`. Every number below is interpolated
from `results/`; none is typed by hand.*

---

## Abstract

Deployed forecasting models decay silently when the data they see changes. This project
builds a weekly wholesale crop price forecaster for India, tunes it with a Bayesian
optimizer written from scratch, and asks a single operational question: when the market
drifts, is it worth **reusing** the tuner's past trial history rather than searching
again from nothing?

Across 4 crops, 5 time-ordered windows
and 5 seeds under an equal 15-trial budget, warm-started search beats a
cold-start search modestly and consistently, but does **not** beat the far cheaper
alternatives of refitting the previous hyperparameters or leaving the model alone. No
comparison survives correction for multiple testing. The experiment was run three times
as its scope widened and gave three different answers; that instability is reported as a
finding rather than resolved in favour of the most flattering run.

This is an empirical evaluation and a working system, **not a new algorithm**.

---

## 1. Introduction and problem framing — Module 1

Agricultural price forecasts in India are consumed by people with no capacity to
maintain them. A trader or a cooperative wants next week's likely price; nobody is
monitoring whether the model that produces it still works. This is the ordinary failure
mode of deployed machine learning: the model does not break, it quietly becomes wrong.

The forecasting task here is deliberately small and concrete: **predict the mean weekly
modal price, in rupees per quintal, of a given crop in a given market one week ahead.**

The research question concerns what to do when that model's world changes:

> After real market drift, does warm-started Bayesian search recover forecast accuracy
> in fewer trials than (a) no retraining, (b) refitting the old best settings, and
> (c) a cold-start search, under a small, equal trial budget?

Each of those three comparisons has a different answer, and §7 reports them separately.

### 1.1 Honest positioning

Adapting AutoML to evolving data is established work, and so is transferring tuning
history between related tasks. This project contributes an end-to-end, reproducible
evaluation of those ideas on Indian commodity price data, with the optimizer written
rather than imported, and the whole pipeline wired to a deployable service. It does not
contribute a new method.

| Prior work | Publication type |
| --- | --- |
| Celik & Vanschoren, *Adaptation Strategies for Automated Machine Learning on Evolving Data*, IEEE TPAMI 43(9), 2021 | Peer-reviewed journal |
| Celik, Singh & Vanschoren, *Online AutoML: an adaptive AutoML framework for online learning*, Machine Learning 112, 2023 | Peer-reviewed journal |
| Wu et al., *ChaCha for Online AutoML*, ICML 2021 | Peer-reviewed conference |
| Feurer, Letham, Hutter & Bakshy, *Practical Transfer Learning for Bayesian Optimization*, arXiv:1802.02219 | **Preprint only.** Checked 2026-10-20: four versions (Feb 2018 – Oct 2022), no journal reference and no publisher DOI beyond the arXiv-issued one. Must be cited as a preprint. |

The last row matters for honesty: the mechanism this project tests — reusing a
surrogate's history across related tasks — rests on a preprint, while the three
peer-reviewed works above establish that the *question* is worth asking.

---

## 2. Data — Module 1

### 2.1 Source and provenance

| | |
| --- | --- |
| Portal | CEDA Agmarknet, Centre for Economic Data and Analysis, Ashoka University |
| Endpoint | `POST https://agmarknet.ceda.ashoka.edu.in/api/prices` |
| Underlying data | Agmarknet, Ministry of Agriculture and Farmers Welfare, Government of India |
| Crops | Onion, Potato, Tomato, Cabbage |
| States requested | 8 candidates, 6 kept per crop |
| Granularity | Daily modal price, aggregated to weekly mean (₹/quintal) |
| Range | 2018-01-01 to 2025-10-30 |

Acquisition is scripted, not manual. `automl.data.fetch` sweeps every (crop, state) pair
three requests at a time, skips files already present so an interrupted run resumes, and
writes `data/raw/provenance.json` recording exactly what was asked for and when. The
portal returns HTTP 500 under load; cabbage needed two extra passes.

### 2.2 The entity is a state, not a single mandi

This is the project's most significant compromise and it belongs in the body of the
report, not a footnote. CEDA's **district**-level endpoint returns HTTP 500/504, and the
current `agmarknet.gov.in` requires a captcha, which this project does not attempt to
circumvent. The only working programmatic source is state-level.

So each "market" is a **state-wide aggregate of mandi modal prices**. A state average is
smoother than any individual mandi, which almost certainly makes this forecasting task
*easier* than the mandi-level task it stands in for. Every result should be read with
that in mind.

### 2.3 Cleaning, and what was deliberately not removed

Two rules, both conservative, because a real onion price spike is signal and not noise:

1. **Hard bounds.** A modal price outside ₹100–₹20000 per quintal is treated as a data-entry error and dropped.
2. **Robust z-score.** Within one (crop, market) series, the log price is compared with
   that series' median using the median absolute deviation. Points beyond
   5.0 MADs are **flagged and kept**, so the
   decision can be revisited without re-downloading anything.

Every operation is scoped to a single (crop, market) series: tomato in Bihar and onion in
Gujarat are different price regimes and must never share a median.

### 2.4 Weekly aggregation and the stop rule

Daily records are resampled to weekly buckets labelled by the Monday the week *starts*
on. This detail caused a real bug: pandas labels a `W-MON` bucket by the Monday it
*ends* on by default, which would have shifted every observation forward by one week —
silently, with correct-looking output. It is now pinned by a test.

A week with fewer than 3 observed days is recorded
as missing rather than averaged from almost nothing, and every series is reindexed onto
one shared weekly grid so gaps are explicit.

The project specification sets a stop rule: halt if more than
10% of weeks are missing. With several crops, halting
on one bad series would block every good one, so the implemented behaviour is to
**exclude the offending series and name it in the quality report**
(`on_stop_rule_violation: halt` restores the original behaviour). Bihar is 45–81% missing
for every crop and is never selected.

Coverage for every candidate series is in `results/data_quality.md`.


---

## 3. Features and evaluation design — Module 1

### 3.1 The indexing convention that makes leakage testable

Every row is indexed by its **target week** `T`. The label is the price in week `T`, and
every feature must be computable before `T` begins:

- `lag_k` is the price in week `T − k`, for k ∈ [1, 2, 4, 8]
- `roll_mean_w` and `roll_std_w` cover weeks `T − w` … `T − 1`, for w ∈ [4, 8]
- `month` and `season` describe `T` itself, which is legitimate — next month's name is
  knowable today without observing any price
- `market` enters as a category; `crop` deliberately does **not**, because a per-crop
  model would see one constant value

### 3.2 Target representation: predicting the change, not the level

The model is fitted on the **week-on-week change** and the prediction is added back onto
last week's price. Metrics are always reported on the price scale either way.

This was measured, not assumed. Gradient-boosted trees cannot predict a value outside
their training range, which costs them badly on a spiking series:

| Target form | Mean test MAE (onion, 5 windows, seed 0) |
| --- | --- |
| `level` — predict the price directly | 200.6 |
| `delta` — predict the change | **142.8** |
| naive "same as last week" | 141.6 |

Delta reaches parity with the random walk; level does not. The configuration uses
`target.mode: delta`.

### 3.3 Proving the absence of leakage

Rather than asserting that no future information leaks, the test suite demonstrates it
with a **perturbation test**: change the price of exactly one week, and the feature row
whose target is that week must be bit-for-bit unchanged. A companion test confirms the
perturbation *does* reach the following week's `lag_1`, so the first test cannot pass
trivially on all-NaN features. Further tests confirm that one market's history never
reaches another's, and that one crop's history never reaches another's.

### 3.4 Window layout

The timeline is cut into 5 windows, each with its own
train / validation / test blocks strictly in time order, with
26-week validation and test blocks and an expanding training set.

|   window_id | train_start   | train_end   | valid_start   | valid_end   | test_start   | test_end   |   train_weeks |
|------------:|:--------------|:------------|:--------------|:------------|:-------------|:-----------|--------------:|
|           1 | 2018-01-08    | 2022-10-17  | 2022-10-24    | 2023-04-17  | 2023-04-24   | 2023-10-16 |           250 |
|           2 | 2018-01-08    | 2023-04-17  | 2023-04-24    | 2023-10-16  | 2023-10-23   | 2024-04-15 |           276 |
|           3 | 2018-01-08    | 2023-10-16  | 2023-10-23    | 2024-04-15  | 2024-04-22   | 2024-10-14 |           302 |
|           4 | 2018-01-08    | 2024-04-15  | 2024-04-22    | 2024-10-14  | 2024-10-21   | 2025-04-28 |           328 |
|           5 | 2018-01-08    | 2024-10-14  | 2024-10-21    | 2025-04-28  | 2025-05-05   | 2025-10-27 |           354 |

Twelve-week blocks were tried first and abandoned: across three markets they gave only
36 validation rows, too few to select hyperparameters from. That decision turned out to
drive the headline result (§7.4).


---

## 4. Baselines — Modules 2 and 3

Five reference points, cheapest first. All are scored on the window 1 test block.

| crop    | method               |     mae |    rmse |   mape |   seconds |
|:--------|:---------------------|--------:|--------:|-------:|----------:|
| Onion   | naive_last_week      |   60.63 |   89.32 |   4.91 |      0    |
| Onion   | naive_mean_4w        |  105.31 |  150.77 |   8.06 |      0    |
| Onion   | lightgbm_default     |   66.56 |   91.77 |   5.23 |      0.27 |
| Onion   | autokeras_structured |   56.88 |   84.93 |   4.6  |     11.56 |
| Onion   | kerastuner_mlp       |   59.36 |   86.45 |   4.85 |     58.4  |
| Potato  | naive_last_week      |   30.11 |   43.2  |   2.25 |      0    |
| Potato  | naive_mean_4w        |   43.63 |   57.36 |   3.49 |      0    |
| Potato  | lightgbm_default     |   32.78 |   48.34 |   2.42 |      0.08 |
| Potato  | autokeras_structured |   28.95 |   41.7  |   2.15 |     20.04 |
| Potato  | kerastuner_mlp       |   29.7  |   42.41 |   2.22 |     91.56 |
| Tomato  | naive_last_week      |  598.26 |  937.34 |  18.03 |      0    |
| Tomato  | naive_mean_4w        | 1368.44 | 1940.75 |  45.59 |      0    |
| Tomato  | lightgbm_default     |  483.8  |  775.82 |  14.3  |      0.08 |
| Tomato  | autokeras_structured |  457.02 |  731.45 |  14.18 |     22.16 |
| Tomato  | kerastuner_mlp       |  461.25 |  767.95 |  14.01 |     80.61 |
| Cabbage | naive_last_week      |   99.96 |  138.21 |   7.89 |      0    |
| Cabbage | naive_mean_4w        |  164.09 |  225.98 |  12    |      0    |
| Cabbage | lightgbm_default     |   98.7  |  151.51 |   7.87 |      0.08 |
| Cabbage | autokeras_structured |   93.53 |  134.68 |   7.33 |     21.87 |
| Cabbage | kerastuner_mlp       |   94.69 |  134.35 |   7.48 |     95.41 |

### 4.1 The naive forecast is a real opponent

Repeating last week's price is the bar every learned model has to clear, and on window 1
the best learned model clears it on all four crops — by 3-7% on onion, potato and
cabbage, and by a wide margin on tomato, whose volatility leaves more signal to learn.
The four-week mean is much worse than the one-week lag everywhere, which is itself
informative: averaging away recent information costs more than the noise it removes.

That comparison is window-1 specific, and it does not hold everywhere. The deployed
models' live backtest over the most recent 26 weeks (§9.2) finds the naive forecast
winning for onion and potato. Both figures are real; they describe different periods and
different models, and the honest summary is that the learned models and the random walk
are close enough that which one wins depends on the slice.

### 4.2 AutoKeras (Module 2)

`StructuredDataRegressor` with a small `max_trials`, to fit on CPU. Included as a
reference point, not a competitor: on a few hundred weekly rows a neural network is not
expected to beat gradient-boosted trees.

**AutoKeras 3.0.0 is not the textbook's AutoKeras.** The course book targets the 1.0.x
line; 3.0 is built on Keras 3, has dropped `TimeseriesForecaster`, rejects pandas
DataFrames outright, and hands Keras 3 a NumPy integer where `Dense` requires a Python
`int`. The baseline runs behind a small documented shim
(`src/automl/baselines/autokeras_baseline.py`). The number is real, but it came from a
library combination that does not work out of the box.

### 4.3 KerasTuner with a custom block (Module 3)

A hand-written search space — depth, width, dropout, learning rate, optimiser — plus a
**custom block**: `LagSelection` is a Keras layer that multiplies the input by a binary
mask the tuner chooses, one boolean per lag feature. This turns *feature selection into
a hyperparameter*: "is the 8-week lag worth keeping?" becomes part of the same search as
"how wide should layer two be?". At least one feature is always kept.

Its choice is the most interpretable result in this section: of the four lag features it
could switch on or off, it kept **only `lag_1`**. The search independently rediscovered
that last week's price carries nearly all the signal.


---

## 5. The custom Bayesian optimizer — Module 4

Eight LightGBM hyperparameters are searched: `colsample_bytree`, `learning_rate`, `max_depth`, `min_child_samples`, `n_estimators`, `num_leaves`, `reg_lambda`, `subsample`.

### 5.1 Vectorising hyperparameters (`tuner/space.py`)

A Gaussian Process cannot read a dictionary; it needs a point in a numeric box. Every
dimension is normalised to the unit interval **[0, 1]**, so no hyperparameter dominates
the kernel's distance calculation merely because its raw numbers are large.

Two details matter. **Log axes**: learning rate and regularisation are searched
logarithmically, because the step from 0.005 to 0.01 matters as much as the step from
0.1 to 0.2. **Integers**: parameters such as `num_leaves` are treated as continuous
during the search and rounded only when a real model is built, which keeps the surrogate
smooth while still producing valid arguments.

### 5.2 The surrogate (`tuner/surrogate.py`)

Fitting one LightGBM model takes seconds and the budget is 15 trials, so after each
real trial a much cheaper model is fitted from hyperparameter vector to the validation
error it produced, and *that* model is asked where to look next.

The kernel is **Matérn with ν = 2.5**: it assumes the error surface is smooth but not
absurdly so. A squared-exponential kernel assumes infinite smoothness and tends to make
the surrogate overconfident between observed points.

Scores are standardised before fitting. Without that, a zero-mean prior would make the
surrogate believe unexplored regions have an error of zero rupees and chase them forever.

### 5.3 Expected Improvement (`tuner/acquisition.py`)

Written here rather than imported. For an untried configuration the surrogate gives a
predicted error `mu` and an uncertainty `sigma`. Expected Improvement asks: *on average,
how many rupees of error would we save against the best result so far by trying this
point?*

With `f_best` the lowest error seen so far and `xi` a small exploration margin:

```
improvement = f_best − mu − xi
z           = improvement / sigma
EI          = improvement × Φ(z) + sigma × φ(z)
```

`Φ` is the standard normal cumulative distribution — the probability of landing below a
value — and `φ` is its density. The first term is the size of the improvement weighted
by how likely it is; the second rewards uncertainty on its own. Points therefore win
either by looking good (low `mu`) or by being unexplored (high `sigma`), which is how one
formula balances exploiting and exploring. EI is never negative: a configuration that
turns out worse than the current best is simply discarded, not counted against us.

### 5.4 The loop (`tuner/bo_tuner.py`)

1. Evaluate 5 configurations drawn at random.
2. Fit the surrogate on every trial so far.
3. Score a large random candidate pool with Expected Improvement, keep the best point.
4. Evaluate it for real, record it, return to 2.
5. Stop when the budget is spent.

Maximising the acquisition function over a random pool, rather than by gradient ascent,
is simple enough to be obviously correct and adequate for eight dimensions.

### 5.5 History, resume and warm start (`tuner/history.py`)

Every trial is persisted to JSON with its vector, parameters, score, window and
timestamp. Two things depend on this: a search interrupted after 8 of 15 trials
resumes rather than restarting, and the warm strategies load earlier windows' trials
before their first new evaluation. A test asserts that a resumed search reproduces the
earlier trials exactly.

Two warm variants are evaluated separately rather than merged:

- **`warm_bo_all`** — every past trial is pre-loaded into the surrogate and the whole
  budget goes to informed proposals. Past trials do not count against the budget, and
  their scores are never reported as the new window's result: they were measured on
  different data.
- **`warm_bo_topk`** — only the best `k` past configurations are reused, and they are
  re-evaluated on the new window's data as the initial design. This costs `k` trials but
  every score is measured on current data.

### 5.6 Does the tuner work at all?

Against random search, identical budget, windows and seeds:

| crop    | method        |   valid_mae |   test_mae |   seconds |
|:--------|:--------------|------------:|-----------:|----------:|
| Cabbage | custom_bo     |       94.47 |      94.64 |      2.07 |
| Cabbage | random_search |       94.3  |      93.62 |      1.52 |
| Onion   | custom_bo     |       84.54 |      59.73 |      1.92 |
| Onion   | random_search |       84.76 |      60.57 |      1.39 |
| Potato  | custom_bo     |       60.46 |      29.38 |      2.29 |
| Potato  | random_search |       61.01 |      29.15 |      1.29 |
| Tomato  | custom_bo     |      133.85 |     525.64 |      2.13 |
| Tomato  | random_search |      134.85 |     525.38 |      1.56 |

**Random search is not a straw man, and this is the clearest demonstration of it.**
Averaged per crop, the custom optimizer reaches a slightly lower validation error on
three of the four crops. Run by run it is a coin flip: the optimizer won 10 of
20 crop-seed pairs on validation error, test error is tied everywhere, and it
costs roughly 1.5x the wall clock.

An earlier onion-only run of this same comparison showed the optimizer clearly ahead
(84.5 against 84.8 on validation, with non-overlapping spreads over five seeds). Across
four crops that advantage disappears. On eight dimensions with a 15-trial budget a
Gaussian Process has very little room to earn its keep over uniform sampling - worth
stating plainly, because the surrogate is the centrepiece of Module 4 and the honest
finding is that it is not yet paying for itself at this budget.


---

## 6. Drift detection — Module 1

Two independent signals, both reported, because neither alone is the whole story: inputs
can shift while the model stays accurate, and a model can decay while its inputs look
unchanged.

**Input drift.** For each numeric feature, a two-sample Kolmogorov–Smirnov test compares
the reference window against the current one. The KS test measures the largest gap
between two cumulative distributions and reports how surprising that gap would be if
both samples came from the same source.

Because many features are tested at once, the p-values carry a **Holm–Bonferroni**
correction. Without it, testing ten features at the 5% level raises a false alarm about
forty percent of the time. Holm sorts the p-values and applies a progressively looser
threshold, which controls that risk without being as blunt as dividing alpha by the
number of tests for every feature.

**Error drift.** The deployed model's error on the current window divided by its error on
the previous one, flagged past a 10% rise.

|   window_id |   n_drifted_features |   n_features_tested | input_drift   |   error_ratio | error_drift   |
|------------:|---------------------:|--------------------:|:--------------|--------------:|:--------------|
|           2 |                    9 |                   9 | True          |         1.646 | True          |
|           3 |                    9 |                   9 | True          |         1.289 | True          |
|           4 |                    9 |                   9 | True          |         0.898 | True          |
|           5 |                    9 |                   9 | True          |         0.74  | True          |

Drift was real throughout: input drift fired at **every** boundary, and the deployed
model's error nearly tripled at one of them. The experiment was not run on a quiet
period.

---

## 7. The drift experiment — the core result

**Design.** 4 crops × 5 strategies × 4
post-drift windows × 5 seeds = 80 matched
runs per strategy, 15 trials per search. Window 1 is the shared deployment moment:
one cold search runs, its winner becomes "the model in production", and its trial history
becomes the knowledge the warm strategies inherit. Every strategy therefore starts from
exactly the same place, with the same seeds and the same data — the fairness condition the
specification requires.

Each warm strategy keeps its **own** accumulating history, so one strategy's trials never
help another's.

### 7.1 Facts: pooled

| strategy     |   mae_mean |   mae_std |   rmse_mean |   rmse_std |   mape_mean |   trials_mean |   seconds_mean |   n_runs |
|:-------------|-----------:|----------:|------------:|-----------:|------------:|--------------:|---------------:|---------:|
| no_retrain   |     129.05 |     70.41 |      187.34 |      97.28 |        6.82 |             0 |           0    |       80 |
| refit        |     129.52 |     74.44 |      187.85 |      99.93 |        6.91 |             0 |           0.1  |       80 |
| cold_bo      |     129.29 |     71.29 |      187.43 |      97.08 |        6.91 |            15 |           2.37 |       80 |
| warm_bo_all  |     127.5  |     71.27 |      185.26 |      96.96 |        6.85 |            15 |           4.69 |       80 |
| warm_bo_topk |     127.76 |     70.79 |      185.94 |      96.87 |        6.85 |            15 |           2.49 |       80 |

### 7.2 Facts: split by crop, before any pooling

Pooling hides disagreement, and here the crops genuinely disagree, so this comes first.

| crop    |   no_retrain |   refit |   cold_bo |   warm_bo_all |   warm_bo_topk | best        |
|:--------|-------------:|--------:|----------:|--------------:|---------------:|:------------|
| Cabbage |         92.8 |    92.7 |      94.5 |          94.4 |           93.3 | refit       |
| Onion   |        144.5 |   137.9 |     135.6 |         131   |          132.4 | warm_bo_all |
| Potato  |         69.8 |    70   |      71.2 |          71.3 |           71.4 | no_retrain  |
| Tomato  |        209.2 |   217.6 |     215.9 |         213.2 |          213.9 | no_retrain  |

Warm start against a plain refit, tested within each crop:

| crop    |   n_pairs | warm_bo_all_wins   |   mean_mae_difference |   p_value | favours     | significant_after_holm   |
|:--------|----------:|:-------------------|----------------------:|----------:|:------------|:-------------------------|
| Onion   |        20 | 14/20              |                 -6.86 |    0.0266 | warm_bo_all | False                    |
| Tomato  |        20 | 14/20              |                 -4.3  |    0.0826 | warm_bo_all | False                    |
| Potato  |        20 | 8/20               |                  1.39 |    0.1893 | refit       | False                    |
| Cabbage |        20 | 9/20               |                  1.71 |    0.2774 | refit       | False                    |

### 7.3 Facts: paired significance tests

Every strategy ran on identical (crop, seed, window) combinations, so each is a **matched
pair**, not an independent sample. The unpaired standard deviations in §7.1 are dominated
by how hard each crop and window is, which affects every strategy equally. A paired
signed-rank test removes that shared difficulty and asks only whether one strategy is
consistently better on the same problem. Wilcoxon is used rather than a paired t-test
because 80 differences are too few to lean on a normality assumption.

| comparison                |   n_pairs | wins   |   mean_mae_difference |   median_mae_difference |   p_value | significant_after_holm   |
|:--------------------------|----------:|:-------|----------------------:|------------------------:|----------:|:-------------------------|
| cold_bo vs no_retrain     |        80 | 28/80  |                  0.24 |                    1.31 |    0.0146 | False                    |
| warm_bo_all vs cold_bo    |        80 | 48/80  |                 -1.79 |                   -0.46 |    0.0392 | False                    |
| warm_bo_topk vs cold_bo   |        80 | 46/80  |                 -1.53 |                   -0.98 |    0.0436 | False                    |
| warm_bo_all vs refit      |        80 | 45/80  |                 -2.02 |                   -0.37 |    0.1515 | False                    |
| warm_bo_all vs no_retrain |        80 | 33/80  |                 -1.55 |                    0.94 |    0.4176 | False                    |
| cold_bo vs refit          |        80 | 35/80  |                 -0.22 |                    0.49 |    0.5    | False                    |
| refit vs no_retrain       |        80 | 39/80  |                  0.47 |                    0.14 |    0.5019 | False                    |

Validation error reached by each search — the quantity the tuner actually optimises:

| strategy     |   mean |    std |   count |
|:-------------|-------:|-------:|--------:|
| cold_bo      | 141.33 | 113.73 |      80 |
| warm_bo_all  | 142.84 | 115.51 |      80 |
| warm_bo_topk | 141.77 | 115.2  |      80 |

Trials needed to reach the cold search's final validation score. **Read `cold_bo` with
care**: the target *is* its own final score, so it reaches it by definition in every run;
only the warm rows carry information.

| strategy     | reached_target   |   mean_trials_when_reached |   median_trials_when_reached |
|:-------------|:-----------------|---------------------------:|-----------------------------:|
| cold_bo      | 80/80            |                      10.51 |                         11.5 |
| warm_bo_all  | 26/80            |                       6.96 |                          6   |
| warm_bo_topk | 40/80            |                       7    |                          7   |

Mean test MAE per window:

|   window_id |   no_retrain |   refit |   cold_bo |   warm_bo_all |   warm_bo_topk |
|------------:|-------------:|--------:|----------:|--------------:|---------------:|
|           2 |        125.8 |   127.5 |     129.5 |         126.7 |          128.2 |
|           3 |        169.2 |   173.4 |     170.6 |         171.2 |          168.9 |
|           4 |        127.7 |   123.6 |     121.3 |         118.8 |          120.4 |
|           5 |         93.4 |    93.5 |      95.8 |          93.2 |           93.6 |

### 7.4 Interpretation

The three claims of the research question have three different answers.

**(c) Against a cold-start search - modest yes.** `warm_bo_all` won 48/80 matched
runs (mean -1.79 rupees, raw p=0.039), and warm starting has the lower mean
error in three of the four crops. This is the part of the question that holds up.

**(b) Against refitting the old settings - no.** 45/80 runs, p=0.15.

**(a) Against no retraining at all - no.** 33/80 runs, p=0.42 - fewer than half.

**A result the question did not ask for, and the most practically useful one:**
hyperparameter search barely helps on this data and often hurts. The cold search *lost*
to leaving the model completely alone in 52 of 80 runs (p=0.015 in
no-retrain's favour). Every strategy's pooled mean sits between 127.5 and
129.5 - under two rupees per quintal apart, on series priced in thousands.

**Nothing survives Holm-Bonferroni correction**, pooled or per crop.

One mechanism is worth noting: all three searches reach almost identical **validation**
error (141.3 to 142.8), yet warm starting produces better **test**
error. The warm prior is not finding a better validation score; it is finding
configurations that generalise better from the same score. Warm starting also costs about
2.0x the wall clock of a cold search, because the Gaussian Process is
refitted on a prior that grows with every window. Accuracy was bought with time, not
saved.

### 7.5 The result moved three times, and that is the finding

| Run | Scope | Validation rows | Winner | Warm vs refit |
| --- | --- | --- | --- | --- |
| 2026-10-18 | onion, 3 states | 36 | `refit` (153.4) | no benefit |
| 2026-10-20 | onion, 6 states | 156 | `warm_bo_all` (131.0) | wins 14/20, p=0.027 |
| 2026-10-20 | **4 crops**, 6 states each | 156 each | `warm_bo_all` by a hair (127.5) | wins 45/80, p=0.15 |

The first run was a clean negative result. Widening the same crop from three states to
six quadrupled the validation rows and reversed it. Adding three more crops then
dissolved most of the gain.

The diagnosis written after the first run — *"the bottleneck is validation signal, not
search quality"* — predicted the second flip. The third run shows the limit of that
reasoning: more data made search worth something on onion, but the advantage did not
transfer to potato or cabbage.

The honest conclusion is narrower than the middle row suggested and more useful than
either extreme: **warm starting is a reliable improvement over cold starting; and on this
data, hyperparameter search is a poor use of effort compared with simply refitting or
leaving the model alone.** Quoting the middle run alone would have been wrong.


---

## 8. Scaling — Module 5

Bayesian optimization is sequential by construction: each proposal is meant to learn from
every previous result. To use more than one core, the loop is relaxed into **batch
Bayesian optimization** — each round takes the top `n_workers` points by Expected
Improvement instead of only the best, evaluates them together, then refits.

The trade is explicit: a batch of four proposals is chosen with less information than
four sequential proposals would have been. What is bought is wall-clock time.

| crop    |   n_workers |   seconds |   speedup_vs_1_worker |   best_valid_mae |
|:--------|------------:|----------:|----------------------:|-----------------:|
| Cabbage |           1 |     3.687 |                 1     |           93.929 |
| Cabbage |           2 |     4.191 |                 0.893 |           93.098 |
| Cabbage |           4 |     2.747 |                 1.351 |           94.231 |
| Cabbage |           8 |     3.038 |                 1.218 |           93.269 |
| Onion   |           1 |     3.462 |                 1     |           84.658 |
| Onion   |           2 |     3.612 |                 0.953 |           83.856 |
| Onion   |           4 |     2.693 |                 1.292 |           83.934 |
| Onion   |           8 |     3.031 |                 1.146 |           84.766 |
| Potato  |           1 |     4.921 |                 1     |           59.956 |
| Potato  |           2 |     4.398 |                 1.169 |           59.706 |
| Potato  |           4 |     3.028 |                 1.627 |           60.282 |
| Potato  |           8 |     3.336 |                 1.477 |           60.145 |
| Tomato  |           1 |     4.227 |                 1     |          133.495 |
| Tomato  |           2 |     4.494 |                 0.946 |          134.56  |
| Tomato  |           4 |     2.61  |                 1.626 |          133.531 |
| Tomato  |           8 |     3.038 |                 1.389 |          135.767 |

Validation quality is flat across worker counts, so batching did not cost accuracy. It
simply bought less time than the worker count suggests: one LightGBM fit on this data
takes a fraction of a second, which is the same order as the cost of dispatching a task
to a worker process. **Trial-level parallelism pays when a single trial is expensive; at
this scale it is mostly overhead.**

### 8.1 Why neither form of GPU parallelism fits

**Data parallelism** replicates the model across devices and splits the batch. With
roughly a thousand training rows there is no batch worth splitting, and synchronisation
would cost more than the compute it saves.

**Model parallelism** splits one model across devices. A LightGBM ensemble of a few
hundred shallow trees is orders of magnitude too small to need it.

The axis that actually helps here is *trial* parallelism across CPU cores, which is what
is measured above.

---

## 9. Deployment — Module 6

### 9.1 Registry and model cards

One tuned model per crop in `models/<slug>/`, each with a **model card** recording
version, training window, metrics, feature names, the tuner and budget used, the creation
time and the git commit. A pickle alone is an unlabelled jar; the card is what lets
somebody finding `model.pkl` six months later decide whether to trust it.

| crop    | version   |   test_mae |
|:--------|:----------|-----------:|
| Cabbage | 1.0.0     |      75.59 |
| Onion   | 1.0.0     |      73.93 |
| Potato  | 1.0.0     |      42.55 |
| Tomato  | 1.0.0     |     201.1  |

### 9.2 The service

A FastAPI application serves both the JSON API and the page that consumes it from one
port, started by one command: `uv run web-start`. There is no second server, no CORS
configuration and no Node toolchain.

Every figure the page shows is **computed per request** from the dataset on disk:
`/api/predict` builds the forecast feature row at request time, and `/api/backtest`
re-runs the registered model across recent weeks and scores it against the naive forecast
on exactly the same rows. That last endpoint is what lets the page report honestly, per
crop and per state, whether the model is beating the baseline — and often it is not.

Serving a real forecast required a correctness fix worth recording: the training table's
last row targets a week that has **already happened**, so predicting from it and labelling
the result "next week" is wrong. `build_forecast_rows` appends a genuine future row per
series, anchored on the last observed price.

### 9.3 Container and CI

The image is built with `uv sync --no-default-groups`, so TensorFlow, AutoKeras, Jupyter,
matplotlib and pyarrow never enter it — the served model is LightGBM and needs no
deep-learning runtime. The price table ships as CSV rather than parquet for the same
reason. `Jenkinsfile` runs Checkout → Lint & Test → Docker build → image smoke test →
Push to GHCR → Deploy via the separate `mandi-infra` repository → Verify `/api/health`,
with every credential referenced by Jenkins credential ID.

---

## 10. Generality check — Module 6

If the optimizer is a general tool rather than something quietly shaped around LightGBM,
it should tune an unrelated pipeline with no changes. It was pointed at 20 Newsgroups text
classification (TF-IDF plus a linear classifier); only the search space and the objective
differ, and the space, surrogate, acquisition and loop modules are imported untouched.

| method        |   best_cv_error |   test_accuracy |   seconds |
|:--------------|----------------:|----------------:|----------:|
| custom_bo     |          0.1109 |          0.8621 |   13.4433 |
| random_search |          0.1164 |          0.8623 |   11.2633 |

The tuner transfers: it found better cross-validation scores than random search on a task
with nothing in common with price forecasting. It showed no test-accuracy advantage.

---

## 11. Limitations

1. **The entity is a state, not a mandi**, so the task is probably easier than the one it
   stands in for.
2. **A random walk is competitive with everything here.** Any claim about model quality
   must be read against the naive baseline.
3. **No comparison survives multiplicity correction**, pooled or per crop.
4. **The crops disagree.** Onion and tomato favour warm starting; potato and cabbage
   favour a plain refit.
5. **The experiment gave three different answers** as its scope widened. §7.5 reports all
   three.
6. **Differences are tiny next to the price scale** — under two rupees per quintal between
   pooled strategy means.
7. **Drift was observed, not induced.** Its size and timing were whatever the market did,
   so nothing here describes behaviour under drift of a controlled magnitude.
8. **The 15-trial budget is small by design** and the conclusions do not extend to
   larger budgets.
9. **One country, one horizon (1 week),
   4 crops.**
10. **Coverage is uneven in places**; see `results/data_quality.md` before trusting any
    single series.
11. **AutoKeras needed a compatibility shim** (§4.2).
12. **Model selection used a single validation block.** No nested cross-validation, so the
    reported test metrics still carry selection noise from one split.

---

## 12. Conclusion and future work

The system works end to end: scripted acquisition from a government data portal, a
leakage-tested feature pipeline, a Bayesian optimizer written from scratch with working
save, resume and warm start, a drift check with multiple-testing correction, per-crop
model registration, and a live service that reports its own accuracy honestly.

The research question's answer is partial. Warm starting beats cold starting consistently
but modestly; it does not beat the cheaper alternatives; and on this data the whole
activity of re-searching hyperparameters is hard to justify against simply refitting.
None of it survives correction for multiple comparisons.

The most transferable lesson is methodological rather than about AutoML. The same code,
on the same crop, produced opposite conclusions at 36 and 156 validation rows, and the
conclusion weakened again when three more crops were added. **A single-configuration
experiment on a small validation set can report whatever its sample size happens to
support.** Reporting all three runs is what makes the finding usable.

Future work, in descending order of expected value:

1. **More validation data per window**, since that is what moved the result — longer
   blocks, more markets, or pooling crops into one model.
2. **Nested cross-validation**, to separate selection noise from genuine generalisation.
3. **Induced drift of known magnitude**, to characterise when warm starting helps instead
   of only whether it did here.
4. **Mandi-level data**, if a source without a captcha becomes available, since the state
   aggregate is the biggest threat to external validity.
5. **Larger trial budgets**, to test whether the warm-over-cold advantage grows or
   vanishes when the search has room to converge anyway.
