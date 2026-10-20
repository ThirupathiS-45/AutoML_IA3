# mandi-automl

Drift-aware AutoML for weekly onion price forecasting in Indian wholesale markets.

A LightGBM forecaster tuned by a **Bayesian optimizer written from scratch** in this
repository. When the market drifts, the tuner can **warm-start** from the trial history
of earlier time periods instead of searching from nothing. The question is whether that
memory is worth anything under a small, fixed trial budget.

## Research question

After real market drift, does warm-started Bayesian search recover forecast accuracy in
fewer trials than (a) no retraining, (b) refitting the previous best settings, and
(c) a cold-start search — under a small, equal trial budget?

## Honest framing

This is an **empirical evaluation plus an end-to-end system, not a new algorithm.**
Adapting AutoML to evolving data, and transferring tuning history between related
tasks, are both established ideas:

| Work | Publication type |
| --- | --- |
| Celik & Vanschoren, *Adaptation Strategies for Automated Machine Learning on Evolving Data*, IEEE TPAMI 43(9), 2021 | Peer-reviewed journal |
| Celik, Singh & Vanschoren, *Online AutoML: an adaptive AutoML framework for online learning*, Machine Learning 112, 2023 | Peer-reviewed journal |
| Wu et al., *ChaCha for Online AutoML*, ICML 2021 | Peer-reviewed conference |
| Feurer, Letham, Hutter & Bakshy, *Practical Transfer Learning for Bayesian Optimization*, arXiv:1802.02219 | **Preprint only — verified 2026-10-20.** arXiv lists four versions (v1 Feb 2018 through v4 Oct 2022) with no journal reference and no publisher DOI, only the arXiv-issued `10.48550/arXiv.1802.02219`. Cite it as a preprint. |

What this repository adds is a reproducible evaluation of those ideas on Indian
commodity price data, with the optimizer written rather than imported, and the whole
thing wired to a deployable service.

## Data

| | |
| --- | --- |
| Source | [CEDA Agmarknet portal](https://agmarknet.ceda.ashoka.edu.in/), Centre for Economic Data and Analysis, Ashoka University |
| Underlying data | Agmarknet, Ministry of Agriculture and Farmers Welfare, Government of India |
| Commodity | Onion (`commodity_id=23`) |
| Granularity | Daily, aggregated to weekly mean modal price (₹/quintal) |
| Range | 2018-01-01 to 2025-10-30 (409 weeks) |
| Downloaded | 2026-10-18 |
| Markets kept | Maharashtra, Karnataka, Gujarat |

### ⚠️ The entity is a state, not a single mandi

This is the one substantive deviation from the original project plan, and it should be
stated in any write-up. CEDA's **district**-level endpoint returns HTTP 500/504, and
`agmarknet.gov.in` v2 now requires a captcha, which this project does not attempt to
bypass. The only working programmatic source is state-level, so each "market" is a
**state-wide aggregate of mandi modal prices**. The column is still named `market` so
the code stays generic, but the aggregation is real and matters: a state average is
smoother than any individual mandi, which almost certainly makes this forecasting task
*easier* than the mandi-level task it stands in for.

Raw CSVs are git-ignored. `data/raw/provenance.json` records exactly what was requested
and when.

## Reproducing everything

```bash
uv sync                  # runtime + dev + research (see pyproject [tool.uv])
uv run pytest            # the full test suite
uv run ruff check .      # lint
```

| Phase | Command | Output |
| --- | --- | --- |
| 1. Download | `uv run python -m automl.data.fetch` | `data/raw/*.csv`, `provenance.json` |
| 1. Weekly series | `uv run python -m automl.data.aggregate` | `data/processed/weekly.parquet`, `results/data_quality.md` |
| 3. Baselines | `uv run python -m automl.experiments.run_baselines --with-deep` | `results/baselines.csv` |
| 4. Tuner check | `uv run python -m automl.experiments.run_tuner_check` | `results/tuner_vs_random.csv` |
| 5. Drift experiment | `uv run python -m automl.experiments.run_drift` | `results/drift_experiment.csv`, `drift_traces.csv` |
| 5. Summary | `uv run python -m automl.experiments.summarize` | `results/summary.md`, plots |
| 6. Parallel | `uv run python -m automl.experiments.run_parallel` | `results/parallel.csv` |
| 7. Register model | `uv run python -m automl.registry` | `models/model.pkl`, `model_card.json` |
| 7. Serve | `uv run flask --app app.app run --port 8000` | http://localhost:8000 |
| 9. Generality | `uv run python -m automl.experiments.run_generality` | `results/generality.csv` |

<!-- RESULTS_START -->
## Results

Every number below came from a command in the table above. Nothing is estimated.

### Baselines, window 1 test block (78 rows, 3 markets x 26 weeks)

| method               |    mae |   rmse |   mape |   seconds |
|:---------------------|-------:|-------:|-------:|----------:|
| autokeras_structured |  71.14 | 103.7  |   5.55 |     12.58 |
| naive_last_week      |  76.08 | 109.17 |   5.92 |      0    |
| kerastuner_mlp       |  79.18 | 112.99 |   6.11 |     29.19 |
| lightgbm_default     |  90.36 | 120.47 |   6.96 |      0.07 |
| naive_mean_4w        | 125.97 | 177.4  |   9.38 |      0    |

Our tuned LightGBM scored **72.2** MAE on the same
block (`results/tuner_vs_random.csv`, mean of 5 seeds), which puts it between AutoKeras
and the naive forecast.

**The naive forecast is hard to beat.** Repeating last week's price gives MAE 76.1. Only
AutoKeras (71.1) and our tuned
LightGBM (72.2) edge past it, and not by
much. Weekly onion prices behave close to a random walk, so this is the expected shape of
the result rather than a defect, but it should temper any claim the system makes.

Note the KerasTuner custom block's choice: of the four lag features it was allowed to
switch on or off, it kept **only `lag_1`**. The search independently rediscovered that
last week's price carries nearly all the signal.

### Phase 4: our Bayesian optimizer against random search

Window 1, 15 trials each, 5 seeds, identical data and seeds.

              valid_mae       test_mae       seconds      
                   mean   std     mean   std    mean   std
method                                                    
custom_bo        122.47  2.22    72.22  1.04    1.64  0.41
random_search    128.29  2.33    72.27  2.62    0.98  0.06

The tuner wins clearly on **validation MAE**, which is the quantity it optimises, and the
two intervals do not overlap. On **test MAE** the methods tie on average, though BO's
spread is about 2.5x tighter. Reading those together: the search is working, but at this
budget the extra validation gain does not reliably survive onto held-out weeks.

### Phase 5: the drift experiment (core result)

5 strategies x 4 post-drift windows x 5 seeds, 15 trials per search.

| strategy     |   mae_mean |   mae_std |   rmse_mean |   trials |   seconds |
|:-------------|-----------:|----------:|------------:|---------:|----------:|
| no_retrain   |     157.25 |     45.21 |      228.48 |        0 |      0    |
| refit        |     153.41 |     43.04 |      223.11 |        0 |      0.03 |
| cold_bo      |     160.73 |     45.79 |      233.78 |       15 |      1.87 |
| warm_bo_all  |     160.43 |     47.17 |      235.22 |       15 |      3.39 |
| warm_bo_topk |     159.61 |     46.59 |      232.81 |       15 |      1.84 |

Mean test MAE per window:

|   window_id |   no_retrain |   refit |   cold_bo |   warm_bo_all |   warm_bo_topk |
|------------:|-------------:|--------:|----------:|--------------:|---------------:|
|           2 |        200.9 |   200.5 |     217.2 |         218.8 |          212.8 |
|           3 |        155.9 |   146.9 |     147.9 |         148.2 |          148.7 |
|           4 |        185.5 |   177.5 |     180.7 |         180   |          184.2 |
|           5 |         86.7 |    88.8 |      97.2 |          94.7 |           92.7 |

Head-to-head against the cold search, run by run:

| strategy | beat `cold_bo` | mean MAE difference |
| --- | --- | --- |
| `no_retrain` | 12/20 | -3.48 |
| `refit` | 15/20 | -7.32 |
| `warm_bo_all` | 13/20 | -0.30 |
| `warm_bo_topk` | 11/20 | -1.11 |

Trials needed to reach the cold search's final validation score. `cold_bo` reaches its
own score by definition, so only the warm rows carry information:

| strategy | reached it | median trials |
| --- | --- | --- |
| `cold_bo` | 20/20 | 11 |
| `warm_bo_all` | 9/20 | 4 |
| `warm_bo_topk` | 10/20 | 8 |

Drift detected at each boundary (KS tests, Holm-Bonferroni corrected):

|   window_id |   drifted |   tested |   error_ratio | error_drift   |
|------------:|----------:|---------:|--------------:|:--------------|
|           2 |         9 |        9 |         2.783 | True          |
|           3 |         7 |        9 |         0.776 | False         |
|           4 |         4 |        9 |         1.19  | True          |
|           5 |         9 |        9 |         0.468 | False         |

#### What this means

**The answer to the research question is no.** Warm-started Bayesian search did not beat
the alternatives on forecast accuracy. Plain **refit** -- reuse window 1's hyperparameters,
fit on current data, spend zero trials -- had the lowest mean test MAE at
153.4, and beat the cold search in 15 of 20 runs. All three
searches landed within about one rupee of each other.

The one place warm start did help is **speed of convergence**. When `warm_bo_all` reached
the cold search's final validation score, it took a median of 4 trials against cold's 11.
But it only got there in 9 of 20 runs, so that advantage is unreliable at this budget.

The most likely explanation is visible in the numbers themselves: all three searches reach
nearly identical *validation* MAE (144.6-145.4), yet none of them turns that into better
*test* MAE than simply reusing old hyperparameters. With 78 validation rows, the search is
fitting validation noise. Better search is not the bottleneck; the validation signal is.

Drift was real and detectable -- input drift fired at every boundary, and the deployed
model's error nearly tripled at window 2 -- so the experiment was not run on a quiet period.

### Phase 6: parallel trials

Batch Bayesian optimization, 24 trials, 3 seeds, on a 16-core machine.

|   n_workers |   seconds |   speedup_vs_1_worker |   best_valid_mae |
|------------:|----------:|----------------------:|-----------------:|
|           1 |     3.597 |                 1     |          122.727 |
|           2 |     3.474 |                 1.032 |          122.337 |
|           4 |     2.605 |                 1.37  |          122.391 |
|           8 |     2.948 |                 1.212 |          124.651 |

**Parallelism barely pays here, and the reason is instructive.** One LightGBM fit on this
data takes about 0.15 s, which is the same order as the cost of dispatching a task to a
worker process. Peak speed-up is 1.37x at 4 workers, and
8 workers is *slower* than 4. Trial-level parallelism is worth it when a single trial is
expensive; at 0.15 s per trial it is mostly overhead. Validation quality is flat across
worker counts, so batching did not cost accuracy -- it simply did not buy much time.

### Phase 9: the same tuner on a different problem

20 Newsgroups text classification, TF-IDF plus a linear classifier, 12 trials, 3 seeds.
Not one line of the tuner changed.

| method        |   best_cv_error |   test_accuracy |   seconds |
|:--------------|----------------:|----------------:|----------:|
| custom_bo     |          0.1109 |          0.8621 |   13.4433 |
| random_search |          0.1164 |          0.8623 |   11.2633 |

The tuner transfers: it found better cross-validation scores than random search on a task
with nothing in common with price forecasting. It showed no test-accuracy advantage.

### The registered model

`models/model_card.json`, version 1.0.0, commit `3cbe926`:
trained through 2025-10-27, test MAE **93.7**,
RMSE 114.6, MAPE 6.7% on the held-out
block of the final window.
<!-- RESULTS_END -->

## How the pieces work

### Leakage prevention

Every row is indexed by the **target week** `T`. The label is the price in week `T`,
and every feature must be computable before `T` begins: `lag_k` is the price in week
`T-k`, and `roll_mean_w` averages weeks `T-w` through `T-1`. Calendar features describe
`T` itself, which is legitimate — next month's name is knowable today.

This is enforced by a **perturbation test**: change the price of exactly one week, and
the feature row whose target is that week must be bit-for-bit unchanged. A companion
test confirms the perturbation *does* reach the following week, so the first test
cannot pass trivially.

### Target representation

The model is fitted on the **week-on-week change**, not the price level, and the
predicted change is added back onto last week's price. Metrics are always reported on
the price scale either way. This was measured, not assumed — see `configs/features.yaml`
for the numbers. Gradient-boosted trees cannot predict a value outside their training
range, which costs them badly on a spiking series like onion prices.

### The custom Bayesian optimizer (`src/automl/tuner/`)

- `space.py` — maps hyperparameters onto the unit box, with log axes for learning rate
  and regularisation, and integers rounded only when a real model is built.
- `surrogate.py` — a Gaussian Process with a Matérn(ν=2.5) kernel on standardised
  scores. Standardisation matters: a zero-mean prior would otherwise make the surrogate
  believe unexplored regions have zero error.
- `acquisition.py` — Expected Improvement, written here rather than imported.
- `bo_tuner.py` — random initial design, then fit the surrogate and take the
  Expected-Improvement-maximising point from a large random candidate pool.
- `history.py` — every trial persisted to JSON, enabling exact resume and warm start.

Warm start comes in two flavours. `warm_all` pre-loads every past trial into the
surrogate and spends the whole budget on informed proposals. `warm_topk` re-evaluates
the best `k` past configurations on current data first, costing `k` trials but
measuring every score on data that is actually current.

### Drift detection (`src/automl/drift.py`)

Two independent signals, both reported because neither is sufficient alone:

- **Input drift** — a two-sample Kolmogorov–Smirnov test per numeric feature, with
  Holm–Bonferroni correction. Without correction, testing ten features at the 5% level
  raises a false alarm about forty percent of the time.
- **Error drift** — the deployed model's error this window divided by last window's.

### Why the Docker image excludes the research stack

TensorFlow, AutoKeras, KerasTuner, Jupyter and matplotlib live in a `research`
dependency group. The image is built with `uv sync --no-default-groups`, so none of
them enter it — the served model is LightGBM and needs no deep-learning runtime.
`pyarrow` was also moved out (it alone was 167 MB); serving rows travel as a joblib
pickle instead.

### Parallelism, and why GPUs do not help here

Bayesian optimization is sequential by construction: each proposal is meant to learn
from every previous result. `run_parallel.py` relaxes this into **batch BO** — take the
top `n_workers` points by Expected Improvement instead of just the best one, evaluate
them together, then refit. The trade is explicit: a batch of four is chosen with less
information than four sequential proposals would have had.

Neither form of GPU parallelism fits this task. **Data parallelism** replicates the
model across devices and splits the batch; with roughly a thousand training rows there
is no batch worth splitting, and the synchronisation cost would exceed the compute.
**Model parallelism** splits a single model across devices; a LightGBM ensemble of a few
hundred shallow trees is far too small to need it. The useful axis here is *trial*
parallelism across CPU cores, which is what is measured.

## Repository layout

```
configs/       YAML configuration — no magic numbers in code
src/automl/    data, features, split, drift, tuner, models, baselines, experiments
app/           Flask service
results/       experiment CSVs, plots and reports (committed)
models/        model.pkl, model_card.json, serving_features.pkl (committed)
tests/         leakage, split-ordering, tuner, drift, registry and app tests
docs/          report outline
notebooks/     exploration only; never imported by src
```

Infrastructure (Terraform + Ansible) lives in a separate `mandi-infra` repository. This
repository holds only the `Dockerfile` and `Jenkinsfile` it is deployed with.

## Environment

Python **3.11**, pinned in `.python-version`. Checked against PyPI on 2026-10-18:
`tensorflow` 2.21.0 requires `>=3.10`, while `autokeras` 3.0.0 and `keras-tuner` 1.4.8
publish classifiers only through 3.11. Python 3.11 is the single version all three
support, so `requires-python` is `>=3.11,<3.12`.

Note that **AutoKeras 3.0.0 is not the textbook's AutoKeras**. The course book (Manning,
2022) targets the 1.0.x line; 3.0 is built on Keras 3 and has dropped
`TimeseriesForecaster`. `StructuredDataRegressor`, which this project uses, survives.

<!-- LIMITATIONS_START -->
## Limitations

Read these before quoting any number above.

1. **The entity is a state, not a mandi.** A state-wide average is smoother than any
   individual market, which very likely makes this task easier than the one it stands in
   for. See the data section for why no mandi-level source was reachable.
2. **A random walk is competitive with everything here.** Repeating last week's price
   scores MAE 76 on window 1; the best learned model scores 71. Any claim about model
   quality has to be read against that.
3. **The headline comparison is a negative result.** Warm starting did not beat refitting.
   The sample is 4 windows x 5 seeds, standard deviations are roughly a third of the
   means, and every strategy's interval overlaps every other's.
4. **Drift was observed, not induced.** Its size and timing were whatever the market did
   between 2024 and 2025, so the experiment cannot say how the strategies behave under
   drift of a controlled magnitude.
5. **The budget of 15 trials is small by design** and the conclusions do not extend to
   larger budgets. With 78 validation rows, more search appears to buy more overfitting.
6. **One commodity, one country, one horizon.** Onion, India, one week ahead.
7. **AutoKeras needed a compatibility shim.** AutoKeras 3.0.0 hands Keras 3 a NumPy
   integer where a Python `int` is required; `src/automl/baselines/autokeras_baseline.py`
   patches this in a subclass. The baseline number is real, but it came from a library
   combination that does not work out of the box.
8. **Model selection used the validation block only.** No nested cross-validation, so the
   reported test metrics still carry selection noise from a single validation split.
<!-- LIMITATIONS_END -->
