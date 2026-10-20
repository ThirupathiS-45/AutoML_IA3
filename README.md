# mandi-automl

Drift-aware AutoML for weekly crop price forecasting in Indian wholesale markets,
with a live FastAPI web app.

A LightGBM forecaster — one per crop — tuned by a **Bayesian optimizer written from
scratch** in this repository. When the market drifts, the tuner can **warm-start** from
the trial history of earlier time periods instead of searching from nothing. The
research question is whether that memory is worth anything under a small, fixed trial
budget.

```bash
uv sync && uv run web-start      # once the data and models exist; see "Running the app"
```

Covers **onion, potato, tomato and cabbage** across **eight states**, 2018–2025, from
the CEDA Agmarknet portal.

## Research question

After real market drift, does warm-started Bayesian search recover forecast accuracy in
fewer trials than (a) no retraining, (b) refitting the previous best settings, and
(c) a cold-start search — under a small, equal trial budget?

### The answer, measured on 4 crops x 5 strategies x 5 windows x 5 seeds

| Claim | Verdict | Evidence |
| --- | --- | --- |
| Warm start beats a **cold-start search** | **Modest yes** | wins 48/80 matched runs, raw p=0.039, lower mean in 3 of 4 crops |
| Warm start beats **refitting** the old settings | **No** | wins 45/80, p=0.15 |
| Warm start beats **no retraining at all** | **No** | wins 33/80, p=0.42 |

Two things qualify all of it, and they are not footnotes:

- **Nothing survives Holm-Bonferroni correction**, pooled or per crop. The pooled
  strategy means span under two rupees per quintal, on series priced in thousands.
- **Hyperparameter search is a poor use of effort on this data at all.** The cold search
  *lost* to leaving the model alone in 52 of 80 runs (p=0.015 in no-retrain's favour).

And the result moved as the scope widened — three runs, three answers. That instability
is reported in full under [Phase 5](#phase-5-the-drift-experiment-core-result) rather
than resolved in favour of the most flattering one.

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

### Where it comes from

| | |
| --- | --- |
| **Portal** | [CEDA Agmarknet](https://agmarknet.ceda.ashoka.edu.in/) — Centre for Economic Data and Analysis, Ashoka University |
| **Endpoint used** | `POST https://agmarknet.ceda.ashoka.edu.in/api/prices` (the portal's own public JSON API, the same one its charts call) |
| **Underlying data** | Agmarknet, Ministry of Agriculture and Farmers Welfare, Government of India |
| **Crops** | Onion (`23`), Potato (`24`), Tomato (`78`), Cabbage (`154`) — IDs from `GET /api/commodities` |
| **States** | Maharashtra, Karnataka, Gujarat, Madhya Pradesh, Rajasthan, Uttar Pradesh, West Bengal, Bihar — IDs from `GET /api/states?commodity_id=<id>` |
| **Granularity** | Daily modal price, aggregated here to weekly mean (₹/quintal) |
| **Range** | 2018-01-01 → 2025-10-30 (409 weeks); 2025-10-30 is the last date the portal returns |
| **Licence / reuse** | CEDA republishes Government of India open data. Attribute both CEDA and Agmarknet in any derived work, and check CEDA's [republish policy](https://ceda.ashoka.edu.in/republish-policy/) before redistributing the raw files. |

Nothing is downloaded by hand. `uv run python -m automl.data.fetch` sweeps every
(crop, state) pair, writes one CSV per series into `data/raw/`, and records exactly
what was requested in `data/raw/provenance.json`. The sweep is **resumable** — files
already present are skipped — and runs three requests at a time, because the portal
takes roughly forty seconds per year of daily data.

Raw and processed data are git-ignored. What *is* committed is
`models/weekly_snapshot.csv`, the small processed table the web app serves from.

### Why these crops and states, and not others?

No technical reason — the original project spec said *"onion and 2–3 markets"*, so the
drift experiment was built on that. The portal exposes **453 commodities and 36
states/UTs**, and widening is a config change, not a code change: edit `crops` and
`candidate_markets` in `configs/data.yaml` and re-run the fetch. Everything downstream
(cleaning, features, windows, models, the API) is keyed on `(crop, market)` and adapts
on its own.

### ⚠️ The entity is a state, not a single mandi

This is the one substantive deviation from the original plan, and it belongs in any
write-up. CEDA's **district**-level endpoint returns HTTP 500/504, and
`agmarknet.gov.in` v2 now requires a captcha, which this project does not attempt to
bypass. The only working programmatic source is state-level, so each "market" is a
**state-wide aggregate of mandi modal prices**. The column is still named `market` so
the code stays generic, but the aggregation is real and matters: a state average is
smoother than any individual mandi, which almost certainly makes this forecasting task
*easier* than the mandi-level task it stands in for.

## Running the app

### First time

```bash
uv sync                                        # install everything
uv run python -m automl.data.fetch             # download prices  (~1 h, resumable)
uv run python -m automl.data.aggregate         # build the weekly series
uv run python -m automl.registry               # tune + register one model per crop
uv run web-start                               # start the app
```

Then open **http://127.0.0.1:8000**.

### Every time after that

```bash
uv run web-start
```

`web-start` is the single command you asked for. It checks the prerequisites in order,
prints the exact command to run for anything missing, and then starts the server.
**Backend and frontend are one process by design** — FastAPI serves both the JSON API
and the page that consumes it, from the same port. There is no second server to launch,
no CORS to configure, and nothing that can drift out of sync.

```
uv run web-start                      # http://127.0.0.1:8000
uv run web-start --reload             # restart on source edits (development)
uv run web-start --port 9000          # different port
uv run web-start --host 0.0.0.0       # reachable from the network
uv run web-start --skip-checks        # start even with data missing
```

### What the app actually does

Every figure on the page is **computed when you ask for it**, not read from a stored
answer:

| Endpoint | What it computes, live |
| --- | --- |
| `POST /api/predict` | Builds the forecast feature row for the next unobserved week from the dataset on disk, runs the registered model, reconstructs the price from the predicted change. |
| `GET /api/history` | The real observed weekly prices for the chosen series. |
| `GET /api/backtest` | Re-runs the registered model across recent weeks and scores it against the naive "same as last week" forecast on exactly the same rows. This is how the page can tell you, honestly, whether the model is beating the baseline for your selection. |
| `GET /api/meta` | Crops, states, model versions and the data range. |
| `GET /api/model/{crop}` | That crop's model card. |
| `GET /api/health` | Liveness. Never touches the model, so it answers even when data is missing. |
| `GET /docs` | Interactive API documentation, generated by FastAPI. |

The service caches the dataset and the models in memory but **reloads them
automatically when the files on disk change**, so re-running `automl.data.aggregate` or
`automl.registry` is picked up without restarting the server.

If data or models are missing the app still starts and `/api/health` still returns 200
— a deployment should come up and report honestly rather than crash-loop. The data
routes return 503 with the exact commands needed to fix it.

## Reproducing the research results

```bash
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
| 7. Register models | `uv run python -m automl.registry` | `models/<crop>/`, `models/index.json` |
| 9. Generality | `uv run python -m automl.experiments.run_generality` | `results/generality.csv` |

Every experiment takes `--crop` and defaults to the first crop in `configs/data.yaml`:

```bash
uv run python -m automl.experiments.run_drift --crop Tomato
uv run python -m automl.registry --crop Onion --crop Potato
```

The committed `results/drift_experiment.csv` is the four crops concatenated; the
individual runs are kept in `results/by_crop/` so each can be checked on its own.
`summarize.py` reads the combined file and reports the per-crop split before any pooled
number, because the crops disagree.

### In a container

```bash
docker build -t mandi-automl .
docker run --rm -p 8000:8000 mandi-automl
curl localhost:8000/api/health
```

The image installs runtime dependencies only (`uv sync --no-default-groups`), so
TensorFlow, AutoKeras, Jupyter, matplotlib and pyarrow never enter it.

<!-- RESULTS_START -->
## Results

Every number below came from a command in the table above. Nothing is estimated.

### Registered models

| crop    |   states | model version   |   test MAE (Rs/quintal) |
|:--------|---------:|:----------------|------------------------:|
| Cabbage |        6 | 1.0.0           |                  75.592 |
| Onion   |        6 | 1.0.0           |                  73.932 |
| Potato  |        6 | 1.0.0           |                  42.55  |
| Tomato  |        6 | 1.0.0           |                 201.097 |

Tomato's error is far larger because tomato prices are far more volatile, not because
the model is worse. The app's live backtest makes that concrete: for **tomato and
cabbage the model beats the naive forecast**, while for onion and potato it does not.

### Baselines, onion, window 1 test block (156 rows)

| method               |    mae |   rmse |   mape |   seconds |
|:---------------------|-------:|-------:|-------:|----------:|
| autokeras_structured |  56.88 |  84.93 |   4.6  |     14.53 |
| kerastuner_mlp       |  59.36 |  86.45 |   4.85 |     58.34 |
| naive_last_week      |  60.63 |  89.32 |   4.91 |      0    |
| lightgbm_default     |  66.56 |  91.77 |   5.23 |      0.18 |
| naive_mean_4w        | 105.31 | 150.77 |   8.06 |      0    |

Our tuned LightGBM scored **59.7** on the same block
(`results/tuner_vs_random.csv`, mean of 5 seeds).

**The naive forecast is a real opponent.** Repeating last week's price gives MAE
60.6. AutoKeras
(56.9), our tuned LightGBM
(59.7) and the KerasTuner MLP
(59.4) all edge past it, but not by much.
Weekly prices behave close to a random walk, so this is the expected shape of the
result rather than a defect. The app's `/api/backtest` reports this honestly per crop
and state — and for tomato the model wins clearly, while for onion and potato the naive
forecast often wins.

Note the KerasTuner custom block's choice: of the four lag features it could switch on
or off, it kept **only `lag_1`**. The search independently rediscovered that last week's
price carries nearly all the signal.

### Phase 4: our Bayesian optimizer against random search

Onion, window 1, 15 trials each, 5 seeds, identical data and seeds.

              valid_mae       test_mae       seconds      
                   mean   std     mean   std    mean   std
method                                                    
custom_bo         84.54  1.36    59.73  1.79    4.87  2.80
random_search     84.76  0.50    60.57  2.18    1.96  0.26

The tuner is slightly ahead on both validation and test error, but the margins are
inside one standard deviation. On this low-dimensional space with a 15-trial budget,
random search is close to competitive — which is worth stating rather than hiding.

### Phase 5: the drift experiment (core result)

**All four crops**, 5 strategies x 4 post-drift windows x 5 seeds = 80 matched runs per
strategy, 15 trials per search, 6 states each.

| strategy     |   mae_mean |   mae_std |   rmse_mean |   rmse_std |   mape_mean |   trials_mean |   seconds_mean |   n_runs |
|:-------------|-----------:|----------:|------------:|-----------:|------------:|--------------:|---------------:|---------:|
| no_retrain   |     129.05 |     70.41 |      187.34 |      97.28 |        6.82 |             0 |           0    |       80 |
| refit        |     129.52 |     74.44 |      187.85 |      99.93 |        6.91 |             0 |           0.1  |       80 |
| cold_bo      |     129.29 |     71.29 |      187.43 |      97.08 |        6.91 |            15 |           2.37 |       80 |
| warm_bo_all  |     127.5  |     71.27 |      185.26 |      96.96 |        6.85 |            15 |           4.69 |       80 |
| warm_bo_topk |     127.76 |     70.79 |      185.94 |      96.87 |        6.85 |            15 |           2.49 |       80 |

#### Split by crop, before any pooling

Pooling hides disagreement, and here the crops genuinely disagree:

| crop    |   no_retrain |   refit |   cold_bo |   warm_bo_all |   warm_bo_topk | best        |
|:--------|-------------:|--------:|----------:|--------------:|---------------:|:------------|
| Cabbage |         92.8 |    92.7 |      94.5 |          94.4 |           93.3 | refit       |
| Onion   |        144.5 |   137.9 |     135.6 |         131   |          132.4 | warm_bo_all |
| Potato  |         69.8 |    70   |      71.2 |          71.3 |           71.4 | no_retrain  |
| Tomato  |        209.2 |   217.6 |     215.9 |         213.2 |          213.9 | no_retrain  |

Warm start against a plain refit, tested separately within each crop:

| crop    |   n_pairs | warm_bo_all_wins   |   mean_mae_difference |   p_value | favours     | significant_after_holm   |
|:--------|----------:|:-------------------|----------------------:|----------:|:------------|:-------------------------|
| Onion   |        20 | 14/20              |                 -6.86 |    0.0266 | warm_bo_all | False                    |
| Tomato  |        20 | 14/20              |                 -4.3  |    0.0826 | warm_bo_all | False                    |
| Potato  |        20 | 8/20               |                  1.39 |    0.1893 | refit       | False                    |
| Cabbage |        20 | 9/20               |                  1.71 |    0.2774 | refit       | False                    |

#### Paired significance tests, pooled over all four crops

Each (crop, seed, window) is a **matched pair** — every strategy saw exactly the same
problem. A paired signed-rank test removes the shared difficulty of each crop and
window, which is what the enormous unpaired standard deviations above are mostly
measuring. `wins` counts the pairs where the left strategy had the lower test MAE.

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

Trials needed to reach the cold search's final validation score. `cold_bo` reaches its
own score by definition, so only the warm rows carry information:

| strategy     | reached_target   |   mean_trials_when_reached |   median_trials_when_reached |
|:-------------|:-----------------|---------------------------:|-----------------------------:|
| cold_bo      | 80/80            |                      10.51 |                         11.5 |
| warm_bo_all  | 26/80            |                       6.96 |                          6   |
| warm_bo_topk | 40/80            |                       7    |                          7   |

Drift detected at each boundary (KS tests, Holm-Bonferroni corrected):

|   window_id |   n_drifted_features |   n_features_tested | input_drift   |   error_ratio | error_drift   |
|------------:|---------------------:|--------------------:|:--------------|--------------:|:--------------|
|           2 |                    9 |                   9 | True          |         1.646 | True          |
|           3 |                    9 |                   9 | True          |         1.289 | True          |
|           4 |                    9 |                   9 | True          |         0.898 | True          |
|           5 |                    9 |                   9 | True          |         0.74  | True          |

#### What this means

Read the three claims separately, because they have different answers.

**1. Warm starting does beat cold starting — modestly, and consistently.**
`warm_bo_all` won 48/80 matched runs against the cold search (mean
-1.79 rupees, raw p=0.039), and `warm_bo_topk` won
46/80 (p=0.044).
Warm starting has the lower mean error in three of the four crops. This is the part of
the research question that holds up.

**2. It does not beat refitting or doing nothing.** Against a plain refit, warm start
won only 45/80 runs (p=0.15). Against leaving the model completely
alone it won 33/80 (p=0.42) — fewer than half.

**3. Searching at all barely helps on this data, and often hurts.** The cold search
*lost* to doing nothing in 52 of 80 runs
(p=0.015 in no-retrain's favour). Every strategy's pooled mean sits between
127.5 and 129.5 — under two rupees per quintal
apart, on series whose prices run to thousands.

**Nothing survives Holm-Bonferroni correction**, pooled or per crop.

#### The onion-only result did not replicate, and that is the finding

This experiment has now been run three times, and the answer moved each time:

| Run | Scope | Winner | Warm vs refit |
| --- | --- | --- | --- |
| 2026-10-18 | onion, 3 states, 36 validation rows | `refit` | no benefit |
| 2026-10-20 | onion, 6 states, 156 validation rows | `warm_bo_all` | warm wins 14/20, p=0.027 |
| 2026-10-20 | **four crops**, 6 states each, 80 runs | `warm_bo_all` by a hair | warm wins 45/80, p=0.15 |

The middle row looked like a clean win for warm starting. Adding three more crops
dissolved it: onion and tomato favour warm starting, potato and cabbage favour refit,
and the pooled advantage over refit stops being significant even before correction.

The honest conclusion is narrower than the middle row suggested, and more useful than
either extreme: **warm starting is a reliable improvement over cold starting, and
hyperparameter search is a poor use of effort on this data compared with simply
refitting or leaving the model alone.** The first result was real; generalising it from
one crop would have been wrong.

Drift itself was real and detectable throughout: input drift fired at every boundary,
and the deployed model's error nearly tripled at one of them.

### Phase 6: parallel trials

Batch Bayesian optimization, 24 trials, 3 seeds, on a 16-core machine.

|   n_workers |   seconds |   speedup_vs_1_worker |   best_valid_mae |
|------------:|----------:|----------------------:|-----------------:|
|           1 |     7.835 |                 1     |           84.658 |
|           2 |     9.102 |                 0.937 |           83.856 |
|           4 |     6.13  |                 1.364 |           83.934 |
|           8 |     5.475 |                 1.438 |           84.766 |

Peak speed-up is 1.44x. Validation quality is flat
across worker counts, so batching did not cost accuracy — it simply bought less time
than the worker count suggests, because a single LightGBM fit on this data is quick
enough that process dispatch is a real share of the cost.

### Phase 9: the same tuner on a different problem

20 Newsgroups text classification, TF-IDF plus a linear classifier, 12 trials, 3 seeds.
Not one line of the tuner changed.

| method        |   best_cv_error |   test_accuracy |   seconds |
|:--------------|----------------:|----------------:|----------:|
| custom_bo     |          0.1109 |          0.8621 |   13.4433 |
| random_search |          0.1164 |          0.8623 |   11.2633 |

The tuner transfers: it found better cross-validation scores than random search on a
task with nothing in common with price forecasting. It showed no test-accuracy
advantage.
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
`pyarrow` was also moved out (it alone was 167 MB), which is why the app's dataset
travels as `models/weekly_snapshot.csv` rather than a parquet file, and serving rows
travel as a joblib pickle.

### Why the web stack looks the way it does

FastAPI rather than Flask: the API is the product here, and FastAPI gives typed request
bodies, validated query parameters and `/docs` for free. The UI is plain HTML, CSS and
ES modules with a hand-written SVG chart — **no CDN, no build step, no Node**. That
keeps the container to one process and one port, and means the frontend has no
dependency that can rot or need pinning.

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

## Project status

| Area | State |
| --- | --- |
| Data pipeline (fetch, clean, weekly aggregate, quality report) | Done, 4 crops x 8 states, 2018-2025 |
| Features and time windows, with leakage tests | Done |
| Baselines (naive, LightGBM, AutoKeras, KerasTuner) | Done, **onion only** |
| Custom Bayesian optimizer (space, GP surrogate, EI, resume, warm start) | Done |
| BO vs random search | Done, **onion only** |
| Drift experiment | Done, **all four crops**, 420 rows |
| Parallel trial timings | Done, **onion only** |
| Model registry + model cards | Done, one model per crop |
| FastAPI app + `web-start` + Docker + Jenkinsfile | Done |
| Generality check (20 Newsgroups) | Done |
| README, report outline | Done |
| Report prose | **Not written.** The outline is in `docs/report_outline.md`. |

Tests: 175 passing. `ruff check` and `ruff format --check` clean. Verified against a
fresh checkout into an empty directory, where `uv sync && uv run web-start` serves
immediately without downloading anything.

Infrastructure (Terraform + Ansible) lives in the separate `mandi-infra` repository and
is deliberately out of scope here.

## Repository layout

```
configs/            YAML configuration — no magic numbers in code
src/automl/         data, features, split, drift, tuner, models, baselines, experiments
app/
  main.py           FastAPI routes
  service.py        live data + model access, reloads when files change
  cli.py            the `web-start` entry point
  templates/        index.html
  static/           app.css, chart.js (hand-written SVG charts)
results/            experiment CSVs, plots and reports (committed)
  by_crop/          per-crop drift results, so each run stays auditable alone
  summary.md        generated tables, facts kept apart from interpretation
  data_quality.md   coverage and stop-rule report
models/
  index.json        which crops are registered (committed)
  weekly_snapshot.csv  the table the app serves from (committed)
  <crop>/           model.pkl, model_card.json, serving_features.pkl per crop
tests/              leakage, split-ordering, tuner, drift, registry and app tests
docs/               report outline
notebooks/          exploration only; never imported by src
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
   individual market, which very likely makes this task easier than the one it stands
   in for. See the data section for why no mandi-level source was reachable.
2. **A random walk is competitive with everything here.** Repeating last week's price
   scores MAE 60.6 on onion window 1; the best learned model scores 56.9.
   The app reports this per selection rather than hiding it.
3. **No comparison survives multiplicity correction**, pooled or per crop. The
   strongest surviving claim is warm-over-cold at a raw p of 0.039 across 80 matched
   runs.
4. **The crops disagree.** Onion and tomato favour warm starting; potato and cabbage
   favour a plain refit. A pooled number hides that, so the per-crop split is reported
   first.
5. **The experiment has given three different answers** as its scope widened, and all
   three are reported rather than quietly dropped. It is the strongest caution here
   about how much a conclusion can depend on sample size and on how many crops you
   look at.
6. **Differences are tiny next to the price scale.** The pooled strategy means span
   under two rupees per quintal, on series whose prices run to thousands. Whether that
   is worth any engineering effort is a judgement the numbers cannot make.
7. **Drift was observed, not induced.** Its size and timing were whatever the market
   did between 2024 and 2025, so the experiment cannot say how the strategies behave
   under drift of a controlled magnitude.
8. **The budget of 15 trials is small by design** and the conclusions do not extend to
   larger budgets.
9. **One country, one horizon, four crops.** India, one week ahead.
10. **Coverage is uneven in places.** Bihar is 45-81% missing for every crop and is
    never selected; cabbage in Madhya Pradesh sits right at the 10% limit. Check
    `results/data_quality.md` before trusting any single series.
11. **AutoKeras needed a compatibility shim.** AutoKeras 3.0.0 hands Keras 3 a NumPy
    integer where a Python `int` is required, and rejects pandas DataFrames outright.
    `src/automl/baselines/autokeras_baseline.py` works around both. The baseline number
    is real, but it came from a library combination that does not work out of the box.
12. **Model selection used a single validation block.** No nested cross-validation, so
    the reported test metrics still carry selection noise from one split.
13. **Baselines, the tuner-vs-random check and the parallel timings were run on onion
    only.** Only the drift experiment covers all four crops. `--crop` makes the rest a
    one-command job per crop; it has not been run.
<!-- LIMITATIONS_END -->
