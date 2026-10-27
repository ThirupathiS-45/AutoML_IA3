<!--
  Skill-based evaluation report, in the institute's format.

  Every number here was read out of results/ on 2026-10-27. This file is
  HAND-MAINTAINED and is not the generated report: docs/report.md is produced by
  `uv run python -m automl.experiments.make_report` and is the technical
  reference. If you change an experiment, regenerate that one and re-check the
  figures here against it.

  Replace every <FILL: ...> marker before submitting. Those are the only things
  the repository cannot tell you.
-->

<div align="center">

# DIVISION OF DATA SCIENCE AND CYBER SECURITY

## SCHOOL OF ENGINEERING AND TECHNOLOGY

### A SKILL BASED EVALUATION REPORT

# Drift-Aware AutoML for Weekly Crop Price Forecasting with a Hand-Written Bayesian Optimisation Tuner

**SUBMITTED BY**

<FILL: YOUR NAME> (<FILL: URK__AI____>)

**COURSE CODE**
<FILL: 24DC2005>

**COURSE NAME**
AutoML

<FILL: OCTOBER 2026>

---

**Submitted by**

<FILL: YOUR NAME> (<FILL: register number>)
<FILL: TEAMMATE NAME> (<FILL: register number>)
<FILL: TEAMMATE NAME> (<FILL: register number>)

**DIVISION OF DATA SCIENCE AND CYBER SECURITY**

**<FILL: KARUNYA INSTITUTE OF TECHNOLOGY AND SCIENCES>**
(Declared as Deemed-to-be-under Sec-3 of the UGC Act, 1956)
Karunya Nagar, Coimbatore - 641 114. INDIA

<FILL: OCTOBER 2026>

</div>

---

## ABSTRACT

Machine learning models that are deployed and then left alone become quietly
less accurate as the world they predict changes. This phenomenon, known as
drift, is well documented, but the practical question of what a small team
should actually do about it is less settled. Retraining costs effort; a full
hyperparameter search costs more; and it is not obvious that either is worth
the expense.

This project builds an end-to-end AutoML system that forecasts the following
week's wholesale price of four crops — onion, potato, tomato and cabbage —
across six Indian states each, and uses that system to answer one measurable
question. After real market drift, does a Bayesian hyperparameter search that
is warm-started from its own past trials recover forecast accuracy in fewer
trials than taking no action, than refitting the previous best settings on new
data, or than starting a fresh search, when all of them are given the same
budget of fifteen trials?

The Bayesian optimiser is written from first principles rather than taken from
a library: hyperparameters are vectorised into a unit box with logarithmic axes
where appropriate, a Gaussian process with a Matern kernel serves as the
surrogate, and the Expected Improvement acquisition function is implemented by
hand. Trial history is persisted, which makes both resumption and warm starting
possible. Drift is detected at each window boundary by two-sample
Kolmogorov-Smirnov tests per feature with Holm-Bonferroni correction, together
with a rise in prediction error.

The experiment ran 420 configurations: four crops, five strategies, five
expanding time-ordered windows and five random seeds. Warm starting from all
past trials produced the lowest pooled mean test error, at 127.50 rupees per
quintal against 129.52 for a plain refit and 129.05 for no retraining at all.
That ordering, however, does not hold up. The four crops disagree about which
strategy is best; the pooled means span less than two rupees per quintal; and
across seven paired comparisons on eighty matched runs each, **no result
survives Holm-Bonferroni correction**. A cold search was beaten by doing
nothing in fifty-two of eighty runs. The answer to the research question is
therefore a qualified one: warm starting is modestly better than a cold search,
is indistinguishable from a refit, and does not beat leaving the model alone.

This report presents an empirical evaluation together with a working system,
**not a new algorithm**. Its most informative outcome is methodological. The
same question was measured three times as the scope widened, and the answer
moved each time — from refit winning on one crop and three states, to warm
starting winning at p = 0.027 on one crop and six states, to warm starting
winning by under two rupees and p = 0.15 across four crops. An encouraging
result on a narrow slice did not generalise, and the project reports that
rather than the most flattering of the three runs.

---

## CHAPTER 1 — INTRODUCTION

Machine learning has moved from research laboratories into routine production
use, and with that move a new class of problem has become common. A model that
was accurate when it was trained does not stay accurate. The statistical
relationship it learned between inputs and outputs belongs to a particular
period, and when the underlying process changes — a change referred to as
drift — the model's predictions degrade. The degradation is usually silent. No
error is raised, no alert fires, and the model continues to serve confident
answers that are progressively further from the truth.

Automated Machine Learning, or AutoML, addresses an adjacent problem. It seeks
to reduce the expertise and labour needed to build a model in the first place
by automating preprocessing choices, model selection and hyperparameter tuning.
The two problems meet at an obvious question: if AutoML can build a model
without much human effort, can it also *maintain* one? When drift is detected,
a search could simply be run again. But a search is not free, and a small team
operating under a fixed compute budget needs to know whether that expense buys
anything.

There is a plausible shortcut. A hyperparameter search that has run before has
accumulated knowledge about which regions of the search space are promising.
If that history is carried forward — if the new search is *warm-started* from
the old one — then in principle it should need fewer trials to find a good
configuration than a search beginning from nothing. This is the hypothesis the
project tests.

The forecasting task chosen is the weekly wholesale, or mandi, price of
agricultural commodities in India. The choice is deliberate. Agricultural
prices genuinely drift: they respond to monsoon timing, planting decisions,
storage capacity, export policy and festival demand, none of which are stable
across years. The data is public. And the forecast has an intelligible unit —
rupees per quintal — so an improvement of two rupees can be recognised for what
it is, rather than hidden inside a normalised score.

Four commodities are forecast: onion, potato, tomato and cabbage. Each has its
own model, because their price levels differ by an order of magnitude and a
single pooled model would be dominated by the most expensive. For each
commodity, six states are selected from eight candidates on the basis of data
coverage.

The project has two deliverables that are assessed together. The first is the
experiment: a fair comparison of five adaptation strategies under an equal
trial budget, with its results reported honestly whether or not they are
favourable. The second is the system: a tested Python package, a web
application that computes every figure it displays from live data, a container
image, a continuous-integration pipeline and the cloud infrastructure it
deploys to. The second exists partly to demonstrate the engineering, and partly
because a claim about maintaining deployed models is more credible when made by
something that is itself deployed.

---

## CHAPTER 2 — LITERATURE REVIEW

The automation of machine learning workflows has a well-established history.
Early systems concentrated on hyperparameter optimisation, approaching it as a
search problem to be solved by grid search or random sampling. Random search in
particular proved a surprisingly strong baseline, because in high-dimensional
spaces only a few dimensions usually matter and random sampling explores those
dimensions more efficiently than a grid of the same size. This result remains
relevant to the present project, and is revisited in Chapter 6.

Bayesian optimisation emerged as the dominant refinement. Rather than sampling
blindly, it builds a probabilistic model — a surrogate — of how configurations
map to performance, and uses that surrogate to choose where to evaluate next.
Gaussian processes are the usual surrogate because they supply not only a
prediction but a calibrated uncertainty, and that uncertainty is what allows an
acquisition function such as Expected Improvement to balance exploiting known
good regions against exploring unknown ones. Systems such as Auto-WEKA and
Auto-sklearn demonstrated that this machinery could be extended from
hyperparameters to entire pipelines.

The specific question of AutoML under *changing* data is more recent and more
directly relevant. Celik and Vanschoren, in a peer-reviewed IEEE Transactions
on Pattern Analysis and Machine Intelligence article (2021), compare adaptation
strategies for AutoML on evolving data streams, and find that the best strategy
depends on the nature and rate of the drift rather than being fixed. Celik,
Singh and Vanschoren extend this in the peer-reviewed journal Machine Learning
(2023), proposing an online AutoML framework that adapts its own pipeline as
data arrives. Wu and colleagues, in a peer-reviewed ICML 2021 paper, present
ChaCha, which treats online AutoML as a problem of managing a champion
configuration against a set of challengers under a limited budget.

The mechanism this project uses — carrying trial history from one search into
the next — is described by Feurer, Letham, Hutter and Bakshy. Their work is
available as arXiv:1802.02219. It was checked across four arXiv versions on
2026-10-20 and carries no journal reference and no publisher DOI, so **it must
be cited as a preprint**, not as a peer-reviewed publication. That check is
recorded because the distinction affects how much weight the citation should
carry.

The course textbook, *Automated Machine Learning in Action* by Song, Jin and
Hu (Manning, 2022), supplies the structure the implementation follows: AutoKeras
for end-to-end solutions, KerasTuner for custom search spaces and blocks, and a
hand-built Bayesian optimiser constructed from vectorisation, a surrogate fitted
to history, an acquisition function and a sampling loop.

The gap this project occupies is narrow and worth stating precisely. The prior
work establishes that adaptation strategy matters and that transferring search
history is a reasonable idea. What it does not supply is a measurement on a
small, real, drifting tabular dataset under the kind of austere budget a student
or a small team actually has — fifteen trials, five seeds, commodity CPU. That
measurement is this project's contribution, and it is a measurement, not an
algorithm.

---

## CHAPTER 3 — SYSTEM DESIGN

### 3.1 System Architecture

The system separates cleanly into four layers, each of which can be exercised
without the others.

```
┌──────────────────────────────────────────────────────────────────┐
│  DATA LAYER                                                      │
│  fetch (CEDA API) → load → clean → aggregate to weekly           │
│  → data/processed/weekly.parquet                                 │
└───────────────────────────┬──────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│  MODELLING LAYER                                                  │
│  features.py (past-only lags, rolling means, calendar)            │
│  split.py (5 expanding time-ordered windows)                      │
│  models/lgbm.py (LightGBM regressor)                              │
│  drift.py (per-feature KS tests + error-rise ratio)               │
└───────────────────────────┬──────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│  TUNER LAYER  (written by hand, not a library)                    │
│  space.py       hyperparameters ↔ unit-box vectors                │
│  surrogate.py   Gaussian process, Matern ν=2.5                    │
│  acquisition.py Expected Improvement                              │
│  history.py     persist trials as JSON → resume and warm start    │
│  bo_tuner.py    the search loop, cold or warm                     │
└───────────────────────────┬──────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│  SERVING AND DEPLOYMENT LAYER                                     │
│  registry.py → models/<crop>/ + models/index.json                 │
│  FastAPI (app/) serves JSON API and HTML page, one port           │
│  Docker image → Jenkins → GHCR → Ansible → Azure VM               │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 High-Level Design

**Data pipeline.** Daily modal prices are fetched from the CEDA Agmarknet
portal, deduplicated, screened for outliers, and aggregated to a weekly mean
with a Monday-labelled week. The aggregation is the step most prone to a silent
error: pandas labels a `W-MON` period by the day the week *ends* on, so the
resampling is pinned with `label="left", closed="left"` and that behaviour is
locked by a regression test. Without it every observation would have shifted
forward by one week, undetectably.

**Feature construction.** Lags of 1, 2, 4 and 8 weeks, rolling means over 4 and
8 weeks, month, season and market as a category. Every feature is computed from
past values only. The target is the following week's price, represented
internally as a week-on-week change and reconstructed to a price by adding the
one-week lag back. That representation was chosen on evidence, not preference:
mean test MAE was 200.6 predicting the level directly, 142.8 predicting the
delta, and 141.6 for the naive baseline. Gradient-boosted trees cannot
extrapolate beyond the range of prices they were trained on, which is why the
level formulation loses.

**Window design.** The timeline is divided into five expanding windows, each
with its own training portion followed by a 26-week validation block and a
26-week test block, strictly in time order and never overlapping. A minimum of
104 training weeks is enforced. Blocks are 26 weeks rather than 12 because
12-week blocks yielded only 36 validation rows, too few to select
hyperparameters from.

**Drift detection.** At each window boundary, every numeric feature's
distribution in the new window is compared against the previous window by a
two-sample Kolmogorov-Smirnov test. Because nine features are tested at once,
the p-values are corrected by the Holm-Bonferroni procedure; without correction,
testing nine features would manufacture apparent drift from noise. A separate
signal is the ratio of the deployed model's error in the new window to its error
in the old one.

**The tuner.** Hyperparameters are mapped into a unit hypercube, with
logarithmic axes for learning rate and regularisation and rounding for integer
parameters, so that the surrogate sees a geometry in which distance means
something. Five random initial trials seed a Gaussian process with a Matern
kernel (ν = 2.5); Expected Improvement is then maximised over random candidates
to pick each subsequent trial, until the budget of fifteen is spent. Every trial
is written to JSON with its vector, parameters, score, window identifier and
timestamp, which is what makes resumption and warm starting possible.

**Warm starting** has two variants, reported separately rather than merged:
`warm_bo_all`, which loads all past trials into the surrogate before the first
new trial, and `warm_bo_topk`, which re-evaluates only the best past
configurations as the new initial design.

### 3.3 UML Diagrams

**Class diagram (description).**

- `Config` — loads YAML from `configs/` and resolves paths. No numeric constant
  is written inside a function; everything is configuration.
- `FeatureBuilder` (`features.py`) — builds past-only features, stamps the
  target mode onto the frame, and appends a genuine future week for serving.
- `WindowSplitter` (`split.py`) — produces the expanding window boundaries and
  the train/validation/test partitions.
- `SearchSpace` (`tuner/space.py`) — bidirectional mapping between a parameter
  dictionary and a unit-box vector.
- `Surrogate` (`tuner/surrogate.py`) — wraps a Gaussian process, standardises
  scores, and refits on each update.
- `Acquisition` (`tuner/acquisition.py`) — Expected Improvement over surrogate
  mean and standard deviation.
- `TrialHistory` (`tuner/history.py`) — persistence and retrieval of trials,
  supporting both exact resumption and warm-start loading.
- `BOTuner` (`tuner/bo_tuner.py`) — the search loop, coordinating the four
  classes above.
- `DriftDetector` (`drift.py`) — per-feature KS tests with Holm correction, plus
  the error-rise ratio.
- `Registry` (`registry.py`) — writes `model.pkl`, `model_card.json` and the
  serving rows per crop, and maintains `models/index.json`.
- `PriceService` (`app/service.py`) — loads models and data, caches on a file
  signature so that changes on disk are picked up automatically.

**Sequence diagram (description).** The drift experiment proceeds as follows.

1. The experiment runner loads the processed weekly table and the window
   definitions.
2. For window 1, the tuner runs a cold search and the best configuration is
   recorded, along with the full trial history.
3. At the boundary into window 2, the drift detector compares feature
   distributions and the deployed model's error, and logs the outcome.
4. Five strategies are then run on window 2 with the same seeds and the same
   budget: no retraining, refit of the old best configuration, a cold search, a
   warm search loaded with all past trials, and a warm search seeded with the
   best past configurations.
5. Each strategy's validation and test error, trial count and wall-clock time
   are appended to `results/drift_experiment.csv` with its configuration.
6. Steps 3 to 5 repeat for windows 3, 4 and 5.
7. `summarize.py` pairs runs on the (crop, seed, window) triple, computes
   per-crop tables before any pooled figure, and applies Wilcoxon signed-rank
   tests with Holm correction.

For a prediction request the sequence is shorter: the browser calls
`/api/predict`, the service checks whether the model or data files have changed
and reloads if so, the per-crop model predicts a delta, the delta is added to
the most recent observed price to produce a price level, and the JSON response
carries that price together with the model card.

### 3.4 User Interface Design

The interface is a single page served by FastAPI from the same process and the
same port as the API, which removes any need for CORS configuration, a Node
build step or a second server. It contains:

- **Crop and state selectors** — driven by `/api/meta`, which reports what the
  registry actually holds rather than a hard-coded list.
- **Next-week forecast** — the predicted price in rupees per quintal, with the
  most recent observed price beside it for context.
- **Historical series chart** — hand-written inline SVG, no charting library and
  no CDN dependency.
- **Backtest panel** — `/api/backtest` re-runs the registered model across
  recent weeks and scores it against the naive forecast on exactly the same
  rows, so the page reports honestly whether the model is currently beating the
  baseline for the selection on screen. For onion and potato it sometimes is
  not, and the page says so.
- **Model card view** — version, training window, metrics and git commit for the
  model actually serving the request.

Nothing displayed is pre-computed or stored as text. Every figure is calculated
per request from the dataset on disk. The whole application starts with a single
command, `uv run web-start`, which runs preflight checks and prints the exact
remedy for anything missing.

### 3.5 Summary

The design's organising principle is that each layer is independently testable
and that no layer trusts another's output silently. Configuration lives in YAML,
time ordering is enforced by tests rather than by convention, trial history is
persisted so that a search can be audited after the fact, and the serving layer
recomputes rather than caches conclusions. The deployment layer is separated
into its own repository so that a change to application code cannot alter
infrastructure.

---

## CHAPTER 4 — IMPLEMENTATION

The system is implemented in Python 3.11, with `uv` managing the interpreter,
the virtual environment and dependencies; `pyproject.toml`, `uv.lock` and
`.python-version` are all committed, so a fresh clone resolves to byte-identical
dependencies. Dependencies are separated into three groups: a runtime group that
ships in the container, a `research` group holding TensorFlow, AutoKeras,
KerasTuner, matplotlib and Jupyter, and a `dev` group with pytest and ruff. The
container installs with `uv sync --frozen --no-default-groups`, which excludes
both of the latter — important because the research group alone is roughly two
gigabytes.

**Data acquisition** is handled by `src/automl/data/fetch.py`, which is
resumable: the CEDA endpoint intermittently returns HTTP 500 under concurrency,
and cabbage required two additional passes to complete. The original intention
was to work at the level of individual mandis, but CEDA's district endpoint
returns HTTP 500 or 504 and the `agmarknet.gov.in` v2 interface requires a
CAPTCHA, which was not bypassed. The only reliable programmatic access is at
state level, so each series is a state-wide aggregate of mandi modal prices.
This is the project's principal data limitation and is stated wherever results
are reported.

**Market selection** picks the six best-covered states per crop from eight
candidates. An early version dropped Maharashtra for onion over two missing
weeks out of 409, so a `coverage_tie_tolerance` was added: near-ties fall back
to the ordering in the configuration file rather than being decided by noise.
Bihar was excluded for every crop, missing 46.70% of weeks for onion, 45.23% for
potato and 79.95% for cabbage.

**The tuner** is the core of the implementation and was built in the order the
textbook prescribes. `space.py` converts between parameter dictionaries and
vectors in the unit hypercube, applying logarithmic scaling to learning rate and
regularisation and rounding integer parameters on the way back.
`surrogate.py` wraps scikit-learn's `GaussianProcessRegressor` with a Matern
kernel at ν = 2.5 and standardises the observed scores before fitting; the
kernel's `length_scale_bounds` were widened to (1e-2, 1e4) after convergence
warnings showed the default range was too narrow for this space.
`acquisition.py` implements Expected Improvement directly from the surrogate's
mean and standard deviation. `bo_tuner.py` runs five random initial trials, then
repeatedly fits the surrogate and maximises Expected Improvement over random
candidates until the fifteen-trial budget is exhausted. `history.py` serialises
every trial, and its round-trip is covered by a test asserting that a resumed
search produces an identical history.

**Baselines** span the course's modules. Two naive forecasts — last week's price
and a four-week mean — provide the floor. LightGBM with default settings
provides an untuned learned model. AutoKeras's structured-data regressor
provides the end-to-end AutoML comparison, and a KerasTuner MLP with a custom
search space and one custom block provides the custom-search-space comparison;
that block chooses which lag features to retain, and selected `lag_1` for every
crop. AutoKeras 3.0.0 required a compatibility shim: it rejects pandas
DataFrames, so inputs are jointly encoded to float32 arrays, and it hands Keras
3 a `numpy.int64` where `Dense` requires a Python `int`, which is worked around
by an explicitly constructed regression head. The textbook targets AutoKeras
1.0.x, so this divergence was expected and is documented in the code.

**Parallel tuning** (`experiments/run_parallel.py`) distributes the trials of a
single search across worker processes with joblib and times 1, 2, 4 and 8
workers on a 24-trial budget.

**Serving.** `registry.py` writes, for each crop, a pickled model, a model card
carrying version, training window, metrics, date and git commit, and the feature
rows needed for serving; `models/index.json` is the registry the API reads. One
subtle failure was caught here: pandas 3 produces PyArrow-backed columns, and a
PyArrow-backed column *Index* was being pickled. The pickle loaded correctly in
development and failed inside the container with
`ModuleNotFoundError: No module named 'pyarrow'`, because PyArrow is deliberately
excluded from the runtime image. Both the data columns and `DataFrame.columns`
must be converted to object dtype before saving. For the same reason, the
processed price table ships into the image as `models/weekly_snapshot.csv`
rather than as a Parquet file; carrying PyArrow would add roughly 170 MB.

**The web application** is FastAPI. `app/main.py` holds the routes — `/`,
`/api/health`, `/api/meta`, `/api/predict`, `/api/history`, `/api/backtest`,
`/api/model/{crop}` and the automatically generated `/docs`. `app/service.py`
holds data and model access with signature-based cache invalidation.
`app/cli.py` provides the `web-start` entry point. Startup uses an
`asynccontextmanager` lifespan handler rather than the deprecated `on_event`
hook.

Two serving bugs are worth recording because both produced plausible-looking
wrong answers rather than errors. The first: the API initially returned the raw
predicted delta, around ₹41, where the user expected a price, around ₹1304 — the
reconstruction step was missing from the serving path. The second: the model was
predicting a week that had already been observed and labelling it "next week".
That is fixed by `build_forecast_rows()`, which appends a genuine future week
per series, and because such a row has no target columns, a `target_mode` column
is stamped onto every frame so the mode can still be determined.

**Containerisation.** A multi-stage Dockerfile builds on
`python:3.11-slim-bookworm`, copies the `uv` binary from its official image,
installs `libgomp1` for LightGBM's OpenMP runtime, runs as a non-root user with
UID 10001, exposes port 8000, declares a health check against `/api/health`, and
starts uvicorn directly, since FastAPI is an ASGI application and needs no
separate process supervisor. The finished image is 754 MB.

**Continuous integration and deployment.** The `Jenkinsfile` in the application
repository declares seven stages: Checkout; Lint and Test (`ruff check`,
`ruff format --check`, `pytest`); Docker Build tagged with the short commit
hash; Smoke Test Image, which runs the container and asserts that
`/api/health` returns `{"status":"ok"}`; Push to GHCR; Deploy; and Verify
Deployment. Jenkins runs locally, so GitHub cannot reach it with a webhook and
`pollSCM('H/2 * * * *')` is used instead. Every credential is referenced only by
its Jenkins credential ID — `ghcr-credentials`, `azure-vm-ssh` and
`azure-vm-ip` — and no secret appears in either repository.

**Infrastructure** lives in a separate repository, `Devops_IA3`, and is
provisioned by Terraform: a Resource Group, Virtual Network, Subnet, Network
Security Group and its subnet association, a Static Public IP, a Network
Interface, and a `Standard_B1ms` Linux virtual machine running Ubuntu 22.04 —
eight resources in total. A cloud-init file installs Docker on first boot.
Ansible then deploys by pulling the built image from GHCR onto the virtual
machine and restarting the container with host port 80 mapped to container port
8000.

One design decision there is worth making explicit, because it is the opposite
of what is often assumed. **The virtual machine never receives source code and
never builds anything.** It has 1898 MB of RAM; building a 754 MB image with
SciPy, scikit-learn and pandas layers on it would exhaust memory. Jenkins builds
on the developer's machine, which has sixteen cores, and the virtual machine only
pulls a finished image.

---

## CHAPTER 5 — TESTING AND VALIDATION

Testing is organised around a single principle: every bug that was found once
has a test that would catch it again. The suite comprises thirteen modules and
**188 tests, all passing locally**. In a clean continuous-integration checkout
187 pass and one is skipped — that one compares the generated technical report
against its committed copy, which requires a deliberately git-ignored dataset
file, so it skips rather than failing every build for a reason that says nothing
about the code.

| Module | What it establishes |
|---|---|
| `test_features.py` | No future information leaks into any feature. Lags and rolling means are verified by perturbing a future value and asserting that no earlier feature changes. |
| `test_split.py` | Windows are strictly time-ordered, never overlap, and respect the minimum training length. Pins the `label="left", closed="left"` weekly resampling. |
| `test_tuner.py` | 33 tests. Vector round-trip is exact; Expected Improvement is non-negative and correctly shaped; a resumed search reproduces its history identically; both warm-start variants load as specified. |
| `test_drift.py` | KS tests behave correctly on known-drifted and known-stable inputs; Holm correction is applied. |
| `test_evaluate.py` | MAE, RMSE and MAPE match hand-computed values; delta predictions are reconstructed to the price scale before scoring. |
| `test_data.py` | Deduplication, outlier screening and weekly aggregation. |
| `test_configs.py` | Every YAML file parses and contains the keys the code requires. |
| `test_registry.py` | Saved artefacts reload in an environment without PyArrow — the regression test for the container failure. |
| `test_target_modes.py` | Level and delta modes both round-trip correctly. |
| `test_app.py` | Application starts, `/api/health` returns 200, `/api/predict` returns a numeric price on the correct scale. |
| `test_summarize.py` | Pairing on (crop, seed, window) is correct; asserts that 40 crop-seed-window rows collapse to 20 pairs, which is the regression test for a pairing bug described below. |
| `test_make_report.py` | The generated report builds with no results present, leaves no unresolved placeholder, and never emits an empty table. |

### Validation of the experiment itself

Three safeguards protect the comparison rather than the code.

**Equal budget.** Every search strategy receives the same fifteen trials, the
same five random initial points, the same data windows and the same five seeds.
Trial counts are logged per run so that this can be verified after the fact
rather than assumed.

**Correct pairing.** An early version paired runs on (seed, window) alone. Since
four crops with price levels spanning an order of magnitude were being compared,
that would have averaged four different price regimes into a single comparison —
silently, and in a way that would have looked like a legitimate result. Pairing
is now on the (crop, seed, window) triple, and a test asserts the pair count.

**Multiple-comparison correction.** Seven paired comparisons are made, so raw
p-values are corrected by Holm-Bonferroni. The corrected column is the one
reported as decisive, and it is reported even though it is unfavourable.

### Bugs found and fixed, with the misleading symptom each produced

Several defects produced plausible wrong answers rather than crashes, which is
the category worth recording:

- Weekly resampling labelled each week by the day it *ended* on, shifting every
  observation forward by one week with no error raised.
- Naive baselines were scored against the delta target rather than the price,
  giving a MAE of 1193 and a MAPE of 3191% — absurd enough to notice, but only
  because the unit was interpretable.
- The API returned a raw delta where a price was expected.
- The serving path predicted an already-observed week and labelled it "next
  week".
- A PyArrow-backed column index passed into a pickle, working in development and
  failing only inside the container.
- A `pgrep` pattern matched its own shell, causing a finished background process
  to be reported as still running.

### Deployment validation

The container was verified to build and serve: `/api/health` returns 200 and
`/api/predict` returns a live price from inside the container. The continuous
integration test stage was run against an extraction of only git-tracked files,
simulating a clean checkout, which is how two build-breaking defects were found
before the first Jenkins run — a test that required a git-ignored dataset, and a
Docker Build precondition that required a file excluded by `.dockerignore`.

---

## CHAPTER 6 — RESULTS AND DISCUSSION

Facts are presented first and interpretation is kept separate and labelled.

### 6.1 Presentation of the Final System

The delivered system forecasts next week's wholesale price for four crops across
six Indian states each. It comprises a tested Python package, a FastAPI
application serving both a JSON API and an HTML page from one process, four
registered per-crop models with model cards, a 754 MB container image, a
seven-stage Jenkins pipeline, and Terraform and Ansible definitions that
provision and deploy to an Azure virtual machine. The application starts locally
with `uv run web-start`, and a commit to the main branch reaches the deployed
virtual machine without manual intervention.

**Registered model performance on the held-out test block:**

| Crop | Test MAE (₹/quintal) | Test RMSE | Test MAPE |
|---|---|---|---|
| Potato | 42.55 | 63.50 | 2.82% |
| Onion | 73.93 | 96.065 | 5.92% |
| Cabbage | 75.59 | 105.95 | 5.53% |
| Tomato | 201.10 | 258.68 | 9.16% |

### 6.2 Facts: baselines on window 1

156 test rows per crop. MAE in rupees per quintal.

| Crop | Naive (last week) | Naive (4-week mean) | LightGBM default | AutoKeras | KerasTuner MLP |
|---|---|---|---|---|---|
| Onion | 60.63 | 105.31 | 66.56 | **56.88** | 59.36 |
| Potato | 30.11 | 43.63 | 32.78 | **28.95** | 29.70 |
| Tomato | 598.26 | 1368.44 | 483.80 | **457.02** | 461.25 |
| Cabbage | 99.96 | 164.09 | 98.70 | **93.53** | 94.69 |

Two facts stand out. AutoKeras has the lowest error on all four crops. And
*untuned* LightGBM is worse than simply repeating last week's price on onion,
potato and cabbage — the naive baseline is a genuinely strong competitor here,
not a formality.

### 6.3 Facts: the drift experiment

420 runs: four crops × five strategies × five windows × five seeds. Eighty
matched runs per strategy on the post-drift windows 2–5.

| Strategy | Mean test MAE | SD | Mean trials | Mean seconds |
|---|---|---|---|---|
| no_retrain | 129.05 | 70.41 | 0 | 0.00 |
| refit | 129.52 | 74.44 | 0 | 0.10 |
| cold_bo | 129.29 | 71.29 | 15 | 2.37 |
| **warm_bo_all** | **127.50** | 71.27 | 15 | 4.69 |
| warm_bo_topk | 127.76 | 70.79 | 15 | 2.49 |

**The per-crop split, which comes before any pooled reading because pooling
hides the disagreement:**

| Crop | no_retrain | refit | cold_bo | warm_bo_all | warm_bo_topk | Best |
|---|---|---|---|---|---|---|
| Cabbage | 92.8 | **92.7** | 94.5 | 94.4 | 93.3 | refit |
| Onion | 144.5 | 137.9 | 135.6 | **131.0** | 132.4 | warm_bo_all |
| Potato | **69.8** | 70.0 | 71.2 | 71.3 | 71.4 | no_retrain |
| Tomato | **209.2** | 217.6 | 215.9 | 213.2 | 213.9 | no_retrain |

**Paired Wilcoxon signed-rank tests**, n = 80 matched (crop, seed, window)
triples, Holm-Bonferroni correction across seven comparisons:

| Comparison | Wins | Mean difference | Raw p | Survives Holm |
|---|---|---|---|---|
| cold_bo vs no_retrain | 28/80 | +0.24 | 0.0146 | **No** |
| warm_bo_all vs cold_bo | 48/80 | −1.79 | 0.0392 | **No** |
| warm_bo_topk vs cold_bo | 46/80 | −1.53 | 0.0436 | **No** |
| warm_bo_all vs refit | 45/80 | −2.02 | 0.1515 | **No** |
| warm_bo_all vs no_retrain | 33/80 | −1.55 | 0.4176 | **No** |
| cold_bo vs refit | 35/80 | −0.22 | 0.5000 | **No** |
| refit vs no_retrain | 39/80 | +0.47 | 0.5019 | **No** |

**Trials needed to reach the cold search's own final validation score:**

| Strategy | Reached target | Median trials when reached |
|---|---|---|
| cold_bo | 80/80 | 11.5 |
| warm_bo_all | 26/80 | 6 |
| warm_bo_topk | 40/80 | 7 |

The cold row reaches the target by definition, since the target is its own final
score; only the warm rows carry information.

**Drift detection at each window boundary** — nine features tested, KS tests
with Holm correction:

| Window | Drifted features | Input drift | Error ratio |
|---|---|---|---|
| 2 | 9 of 9 | True | 1.646 |
| 3 | 9 of 9 | True | 1.289 |
| 4 | 9 of 9 | True | 0.898 |
| 5 | 9 of 9 | True | 0.740 |

### 6.4 Facts: the custom tuner against random search

On window 1, with an equal budget of fifteen trials across twenty crop-seed
pairs, the custom Bayesian optimiser achieved the lower validation MAE in **10
of 20 pairs** — a coin flip — and the two methods were tied on test error, at
roughly 1.5× the wall-clock cost. An earlier onion-only run had suggested the
tuner was clearly ahead. It does not replicate.

### 6.5 Facts: parallel tuning

24-trial budget, twelve runs per worker count.

| Workers | Mean speed-up | Max speed-up |
|---|---|---|
| 1 | 1.000 | 1.000 |
| 2 | 0.990 | 1.468 |
| 4 | **1.474** | 1.869 |
| 8 | 1.307 | 1.525 |

### 6.6 Facts: generality check

20 Newsgroups, TF-IDF with an SGD classifier, budget of twelve trials, three
seeds.

| Method | Best CV error | Test accuracy | Seconds |
|---|---|---|---|
| custom_bo | **0.1109** | 0.8621 | 13.44 |
| random_search | 0.1164 | **0.8623** | 11.26 |

### 6.7 Interpretation

*The preceding sections are measurements. This section is reading, and should be
weighed accordingly.*

**On the research question.** Taking its three claims separately: against a cold
start, warm starting is **modestly better** — `warm_bo_all` won 48 of 80 matched
runs at a raw p of 0.039, and had the lower mean in three of four crops. Against
a plain refit, **no** — 45 of 80, p = 0.15. Against no retraining at all,
**no** — 33 of 80, p = 0.42. Doing nothing is hard to beat.

**The strongest negative finding is not about warm starting.** A cold search
*lost* to doing nothing in 52 of 80 runs. On this dataset, at this budget,
hyperparameter search is a poor use of effort regardless of how it is
initialised. That is worth more than the headline comparison.

**Nothing survives correction.** Seven comparisons were made; the corrected
column is negative throughout. Pooled strategy means span under two rupees per
quintal, against standard deviations around seventy. The honest summary is that
this experiment cannot distinguish these strategies.

**Where warm starting does show something.** When `warm_bo_all` reached the cold
search's final validation score, it took a median of 6 trials against the 11.5
the cold search needed. The efficiency claim has support; the accuracy claim does
not. But it reached that bar in only 26 of 80 runs, so the efficiency applies to
a minority of cases.

**The drift detector fires every time, which limits its usefulness.** All nine
features registered drift at all four boundaries. With 26-week blocks of a
seasonal commodity series, a KS test will almost always find a distributional
difference, because the seasons differ. The error ratio is more discriminating —
1.646 and 1.289 at windows 2 and 3 indicate genuine degradation, while 0.898 and
0.740 show the model actually improving — and would make the better trigger.

**Parallelism saturates at four workers.** Each trial takes roughly 0.1–0.2
seconds, so beyond four workers process start-up and data pickling cost more than
the parallelism saves. On a tabular task of this size, data and model parallelism
of the kind used for deep learning on GPUs do not apply: the model is small, fits
in cache, and the bottleneck is coordination rather than computation.

**The tuner generalises as a component.** On 20 Newsgroups it found a better
cross-validation score than random search and arrived at the same test accuracy,
which suggests the implementation is sound even though its advantage on the
mandi task is not demonstrable.

### 6.8 Evaluation of Project Objectives

| Objective | Outcome |
|---|---|
| Build a reproducible weekly price forecaster | Met. Four crops, six states each, `uv sync && uv run pytest` passes on a fresh clone. |
| Implement Bayesian optimisation from first principles | Met. Vectorisation, GP surrogate, hand-written Expected Improvement, persistence, resumption and two warm-start variants, covered by 33 tests. |
| Detect drift at window boundaries | Met, with a caveat: the input-drift test fires at every boundary and so carries little information; the error-rise ratio is the more useful signal. |
| Compare adaptation strategies under an equal budget | Met. 420 runs, equal budget verified from logs, paired tests with correction. |
| Answer the research question | Answered, in the negative for two of its three claims and weakly for the third. No effect survives correction. |
| Report AutoKeras and KerasTuner baselines | Met. Both run on all four crops; AutoKeras is the strongest single model on window 1. |
| Measure parallel speed-up | Met. Peaks at 1.474× mean on four workers; eight is slower. |
| Package and deploy | Met. 754 MB image, seven-stage Jenkins pipeline, Terraform-provisioned Azure VM, Ansible deployment, automatic redeploy on commit. |
| Report limitations honestly | Met. Thirteen limitations are listed in the README, and the three conflicting runs appear before any favourable figure. |

### 6.9 Challenges Faced

**The data source was more restricted than planned.** CEDA's district endpoint
returns HTTP 500 or 504 and the Agmarknet v2 interface requires a CAPTCHA. The
project works at state level as a result, which weakens every claim about mandi
prices specifically.

**The naive baseline was unexpectedly strong.** Untuned LightGBM lost to
repeating last week's price. Diagnosing this led to the finding that trees cannot
extrapolate beyond their training range, and to reformulating the target as a
week-on-week delta — which is one of the more instructive outcomes of the work.

**Several bugs produced plausible wrong answers rather than errors.** The weekly
resampling offset, the baselines scored on the wrong scale, the API returning a
delta as a price, and the serving path predicting an already-observed week all
ran without complaint. Each now has a regression test.

**Library versions diverged from the textbook.** AutoKeras 3.0.0 rejects pandas
DataFrames and passes Keras 3 a NumPy integer where a Python integer is required.
The textbook targets 1.0.x. Both issues required documented workarounds.

**The result weakened every time the scope widened.** This was the hardest thing
to handle well, because the temptation to report the six-state onion result — a
clean p = 0.027 — was real. All three runs are reported.

**Deployment surfaced environment-specific obstacles.** The Azure for Students
subscription enforces an allowed-regions policy that rejected `centralindia`
with a 403 after the resource group had already been created; the permitted set
had to be queried from the policy assignment directly. Jenkins had rotated its
apt signing key while its own documentation still referenced the superseded one.
Neither is a programming problem, and both consumed real time.

### 6.10 Summary

The system works and is deployed. The experiment is fair, adequately powered for
a paired test, and returns a largely negative answer. Warm starting is modestly
better than a cold search and no better than a refit or than doing nothing, and
no comparison survives multiple-comparison correction. The most useful finding is
that on this data, at this budget, hyperparameter search of any kind is hard to
justify — and that a result which looked solid on a narrow slice did not survive
a wider one.

---

## CONCLUSION

This project set out to answer a single measurable question about maintaining
deployed machine learning models, and to build a system capable of answering it
credibly. Both were delivered.

The system is an end-to-end AutoML pipeline for weekly crop price forecasting:
data acquisition from a public commodity portal, leakage-free feature
construction, time-ordered window splitting, drift detection by corrected
Kolmogorov-Smirnov tests and error-rise ratio, a Bayesian optimiser written from
first principles with persistent trial history and two warm-start variants, a
per-crop model registry, a FastAPI application that computes every figure it
displays per request, a 754 MB container image, a seven-stage Jenkins pipeline,
and Terraform-provisioned Azure infrastructure deployed by Ansible. One hundred
and eighty-eight tests pass, and a commit to the main branch reaches the running
virtual machine without manual intervention.

The experimental answer is largely negative, and is reported as such. Across 420
runs, warm-started Bayesian search achieved the lowest pooled mean test error at
127.50 rupees per quintal, against 129.52 for a refit and 129.05 for taking no
action. It won 48 of 80 matched runs against a cold search at a raw p of 0.039,
but only 45 of 80 against a refit and 33 of 80 against doing nothing. **No
comparison survives Holm-Bonferroni correction across the seven tests performed.**
The four crops disagree about which strategy is best. A cold search lost to doing
nothing in 52 of 80 runs, which is the more consequential finding: on this
dataset and at this budget, hyperparameter search is a poor use of effort however
it is initialised. Where warm starting does show an advantage is efficiency — a
median of 6 trials to reach the cold search's final score against 11.5 — but it
reached that score in only 26 of 80 runs.

This is an empirical evaluation together with a working system, **not a new
algorithm**. The mechanism it tests was described by Feurer and colleagues in a
preprint, and the question of adaptation strategy under drift is treated in
peer-reviewed work by Celik and Vanschoren and by Wu and colleagues. The
contribution is a measurement under austere, realistic conditions.

The project's most transferable lesson is methodological. The same question was
measured three times as scope widened, and the answer changed each time: refit
won on one crop and three states; warm starting won at p = 0.027 on one crop and
six states; warm starting won by under two rupees at p = 0.15 across four crops.
The custom tuner looked clearly better than random search on onion alone, and was
a coin flip across four crops. Every encouraging narrow result weakened under
widening. Reporting that movement, rather than the most flattering of the three
runs, is the part of this work most worth carrying forward.

Future work would widen the data rather than refine the method: individual mandi
series if a non-CAPTCHA source becomes available, exogenous features such as
rainfall, festival calendars, fuel prices and import policy, more windows and
seeds for genuine statistical power, and an error-rise trigger in place of an
input-drift test that fires unconditionally on seasonal data.

---

## REFERENCES

1. Z. Song, H. Jin and X. Hu, *Automated Machine Learning in Action*. Manning
   Publications, 2022. — **Course textbook.**
2. B. Celik and J. Vanschoren, "Adaptation Strategies for Automated Machine
   Learning on Evolving Data," *IEEE Transactions on Pattern Analysis and
   Machine Intelligence*, vol. 43, no. 9, 2021. — **Peer-reviewed journal.**
3. B. Celik, P. Singh and J. Vanschoren, "Online AutoML: an adaptive AutoML
   framework for online learning," *Machine Learning*, vol. 112, 2023. —
   **Peer-reviewed journal.**
4. Q. Wu, C. Wang, J. Langford, P. Mineiro and M. Rossi, "ChaCha for Online
   AutoML," *Proceedings of the 38th International Conference on Machine
   Learning (ICML)*, 2021. — **Peer-reviewed conference.**
5. M. Feurer, B. Letham, F. Hutter and E. Bakshy, "Practical Transfer Learning
   for Bayesian Optimization," arXiv:1802.02219. — **Preprint only.** Verified
   across four arXiv versions on 2026-10-20: no journal reference and no
   publisher DOI.
6. Centre for Economic Data and Analysis (CEDA), Ashoka University — Agmarknet
   data portal. <https://ceda.ashoka.edu.in>. Underlying data: Agmarknet,
   Ministry of Agriculture and Farmers Welfare, Government of India.
7. G. Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision
   Tree," *Advances in Neural Information Processing Systems 30*, 2017. —
   **Peer-reviewed conference.**
8. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of
   Machine Learning Research*, vol. 12, 2011. — **Peer-reviewed journal.**
9. S. Holm, "A Simple Sequentially Rejective Multiple Test Procedure,"
   *Scandinavian Journal of Statistics*, vol. 6, no. 2, 1979. —
   **Peer-reviewed journal.**
10. FastAPI documentation. <https://fastapi.tiangolo.com>
11. Terraform `azurerm` provider documentation.
    <https://registry.terraform.io/providers/hashicorp/azurerm>
12. Jenkins Pipeline documentation. <https://www.jenkins.io/doc/book/pipeline>

---

## APPENDICES

**Appendix A — Repositories.**
Application, `Dockerfile` and `Jenkinsfile`:
<https://github.com/dharshan-kumarj/AutoML_IA3>
Terraform and Ansible: <https://github.com/dharshan-kumarj/Devops_IA3>

**Appendix B — Commands to reproduce every figure in this report.**

```bash
uv sync --group research                                  # full environment
uv run python -m automl.data.fetch                        # download raw data
uv run python -m automl.data.aggregate                    # → weekly.parquet
uv run python -m automl.experiments.run_baselines         # → baselines.csv
uv run python -m automl.experiments.run_tuner_check       # → tuner_vs_random.csv
uv run python -m automl.experiments.run_drift             # → drift_experiment.csv
uv run python -m automl.experiments.run_parallel          # → parallel.csv
uv run python -m automl.experiments.run_generality        # → generality.csv
uv run python -m automl.experiments.summarize             # → summary.md, plots
uv run python -m automl.experiments.make_report           # → docs/report.md
uv run pytest                                             # 188 tests
uv run web-start                                          # the application
```

**Appendix C — Result files.**
`results/drift_experiment.csv` (420 rows), `results/drift_traces.csv`,
`results/baselines.csv`, `results/tuner_vs_random.csv`, `results/parallel.csv`,
`results/generality.csv`, `results/summary.md`, `results/data_quality.md`,
`results/by_crop/` (20 per-crop files), `results/error_vs_trial.png`,
`results/strategy_comparison.png`, `results/weekly_series.png`.

**Appendix D — The generated technical report.** `docs/report.md` is produced by
`make_report` and interpolates every figure directly from `results/`, so no
number in it can drift from its source CSV unnoticed. This report is the
hand-written course document; that one is the technical reference.

**Appendix E — Screenshots to attach.**
1. `terraform plan` output
2. `terraform apply` completing, with outputs visible
3. Azure Portal resource group listing all eight resources
4. Jenkins stage view, all seven stages green
5. Jenkins console log at Verify Deployment, showing `deployment healthy`
6. The running application with the public URL in the address bar
7. A build with "Started by an SCM change", proving continuous deployment
8. `uv run pytest` showing 188 passing tests

**Appendix F — Infrastructure provisioned.** Resource Group, Virtual Network
(10.20.0.0/16), Subnet (10.20.1.0/24), Network Security Group with SSH and HTTP
rules, NSG-subnet association, Static Public IP, Network Interface, and a
`Standard_B1ms` Ubuntu 22.04 virtual machine in `indiasouthcentral`.

---

## EVALUATION SHEET

**Reg. No:** <FILL: URK__AI____>
**Name:** <FILL: YOUR NAME>
**Course code:** <FILL: 24DC2005>
**Course Name:** AutoML

| S.No | Rubrics | Maximum marks | Marks Obtained |
|---|---|---|---|
| 1 | Problem Framing and Data Pipeline | 10 | |
| 2 | Model Training and Experiment Tracking | 7 | |
| 3 | Model Packaging and Deployment | 5 | |
| 4 | CI/CD and Automation | 5 | |
| 5 | Monitoring and Documentation | 3 | |
| 6 | Presentation and Viva | 5 | |
| 7 | Report | 5 | |
| | **Total** | **40** | |

<br><br>

**Signature of the Faculty-in-charge**
