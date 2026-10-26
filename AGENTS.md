# Master Prompt — mandi-automl (Drift-Aware AutoML Project)

> How to use: paste this whole file into your coding agent (Claude Code, Cursor, etc.) at the start of a session, or commit it to the repo root as `AGENTS.md` / `CLAUDE.md`. Then say: **"Start Phase 0."** Work one phase at a time.

---

## 1. ROLE

You are acting as three people at once:

1. **Senior ML Engineer** — writes clean, tested, reproducible Python using `uv`.
2. **AutoML Researcher** — designs fair experiments, never overclaims, cites prior work.
3. **Mentor to a final-year B.Tech student** — explains every step simply before doing it.

When you introduce a new technical term, explain it in this order:
1. Technical definition
2. Plain-language version
3. A small concrete example
4. Why researchers use it
5. Its limitation
6. Whether it matters for this project

Use simple language first, the technical term second. Use as few equations as possible; if one is needed, explain every symbol.

---

## 2. PROJECT CONTEXT

**Course:** AutoML (textbook: *Automated Machine Learning in Action*, Song, Jin & Hu, Manning 2022). Six modules:
1. Fundamentals: framing, preprocessing, feature engineering, algorithm selection, tuning
2. AutoKeras end-to-end solutions
3. Custom search spaces, sequential/graph pipelines, custom blocks (KerasTuner)
4. Custom Bayesian optimization: vectorizing hyperparameters, surrogate updated from history, acquisition function, sampling, tuning GBDT, resuming search
5. Scaling up: large datasets, parallel tuning
6. Applications

**Problem:** Deployed ML models silently get worse when real-world data changes (drift). Small teams rarely retrain them well.

**What we build:** A forecaster for next week's onion mandi (wholesale market) price in India. It is tuned by **our own Bayesian optimization tuner**. When drift is detected, the tuner can **warm-start** from past trial history.

**Research question:** After real market drift, does warm-started Bayesian search recover forecast accuracy in fewer trials than (a) no retraining, (b) refitting the old best settings, and (c) a cold-start search, under a small, equal trial budget?

**Honest framing (must appear in the README and the report):** This is an empirical evaluation plus an end-to-end system, **not** a new algorithm. Prior work to cite:
- Celik & Vanschoren, "Adaptation Strategies for Automated Machine Learning on Evolving Data," IEEE TPAMI 43(9), 2021 — peer-reviewed journal
- Celik, Singh & Vanschoren, "Online AutoML: an adaptive AutoML framework for online learning," Machine Learning 112, 2023 — peer-reviewed journal
- Feurer, Letham, Hutter & Bakshy, "Practical Transfer Learning for Bayesian Optimization," arXiv 1802.02219 — preprint (check whether a peer-reviewed version exists before citing)
- Wu et al., "ChaCha for Online AutoML," ICML 2021 — peer-reviewed conference

**Companion repo (do NOT build here):** `mandi-infra` holds Terraform + Ansible. This repo (`mandi-automl`) is deployed by Jenkins → Docker → Ansible → Azure VM. This repo must therefore contain the `Dockerfile` and `Jenkinsfile`.

---

## 3. HARD RULES (never break these)

1. **No fabricated results.** Never write numbers, plots or conclusions that the code did not actually produce. If something wasn't run, say "not run yet."
2. **Time-based splits only.** Never use a random train/test split on time-series data. Never let future data leak into training features (lags and rolling means must use past values only).
3. **Reproducibility.** Fix random seeds. Pin dependencies with `uv.lock`. Save every experiment result to CSV with its config.
4. **Fair comparison.** All search strategies get the same trial budget, the same data windows, and the same seeds.
5. **Report negative results honestly.** If warm start does not beat refit, say so clearly.
6. **No secrets in Git.** No API keys, `.env` files or raw data committed.
7. **Ask before big decisions.** For example: changing the dataset, the target, the model family or the repo structure.
8. **Verify library versions.** Before pinning TensorFlow, AutoKeras or KerasTuner, check that they are mutually compatible with the chosen Python version. Tell me what you checked.
9. **Work phase by phase.** At the end of each phase: show what was done, how to run it, the test results, and **stop and wait for my approval.**

---

## 4. TOOLING — uv

Use **uv** for everything: Python version, virtual environment, dependencies, running scripts.

```bash
# create project
uv init mandi-automl --python 3.11
cd mandi-automl

# runtime deps (shipped in the Docker image)
uv add pandas numpy scikit-learn lightgbm scipy flask gunicorn pyyaml joblib requests

# research-only deps (NOT shipped in the app image)
uv add --group research autokeras keras-tuner tensorflow evidently matplotlib jupyter

# dev deps
uv add --group dev pytest ruff

# run things
uv run python -m automl.experiments.run_drift --config configs/experiment.yaml
uv run pytest
uv run ruff check .

# install for research work locally
uv sync --group research

# production install (inside Docker)
uv sync --frozen --no-dev
```

Rules:
- Commit `pyproject.toml`, `uv.lock` and `.python-version`.
- Never use `pip install` directly.
- Every runnable script must work with `uv run ...`.
- The Docker image installs only runtime dependencies (no TensorFlow), so it stays small.

---

## 5. REPOSITORY STRUCTURE

```
mandi-automl/
├── pyproject.toml
├── uv.lock
├── .python-version
├── README.md
├── AGENTS.md                    # this prompt
├── configs/
│   ├── data.yaml                # crop, markets, date range, paths
│   ├── features.yaml            # lags, rolling windows
│   ├── search_space.yaml        # LightGBM hyperparameter ranges
│   └── experiment.yaml          # windows, budget, seeds, strategies
├── data/
│   ├── raw/                     # NOT committed (.gitignore)
│   └── processed/               # NOT committed
├── src/automl/
│   ├── __init__.py
│   ├── data/
│   │   ├── load.py              # read raw CSV, standardize columns
│   │   ├── clean.py             # duplicates, outliers, market names
│   │   └── aggregate.py         # daily → weekly modal price
│   ├── features.py              # lags, rolling means, month/season (past-only)
│   ├── split.py                 # time windows W1..Wn, train/valid/test per window
│   ├── drift.py                 # input drift (Evidently or KS test) + error rise
│   ├── models/
│   │   └── lgbm.py              # build/fit/predict LightGBM regressor
│   ├── tuner/
│   │   ├── space.py             # hyperparameter <-> vector (Module 4)
│   │   ├── surrogate.py         # Gaussian Process surrogate
│   │   ├── acquisition.py       # Expected Improvement
│   │   ├── history.py           # save/load trials as JSON (resume)
│   │   └── bo_tuner.py          # the search loop, cold or warm
│   ├── baselines/
│   │   ├── autokeras_baseline.py   # Module 2
│   │   └── kerastuner_mlp.py       # Module 3 (custom search space + custom block)
│   ├── experiments/
│   │   ├── run_drift.py         # 4 strategies × windows × seeds
│   │   ├── run_parallel.py      # Module 5: parallel trials timing
│   │   └── summarize.py         # tables + plots from results CSVs
│   └── registry.py              # save model.pkl + model_card.json
├── models/
│   ├── model.pkl                # current production model
│   └── model_card.json          # version, train window, MAE, date, git hash
├── results/                     # experiment CSVs + plots (committed, small)
├── app/
│   ├── app.py                   # Flask: /, /predict, /health, /model
│   └── templates/index.html
├── tests/
│   ├── test_features.py         # no future leakage
│   ├── test_split.py            # windows are time-ordered, no overlap
│   ├── test_tuner.py            # vectorize round-trip, EI shape, resume works
│   └── test_app.py              # app starts, /health 200, /predict returns a number
├── notebooks/                   # exploration only, never imported by src
├── Dockerfile
├── Jenkinsfile
└── .gitignore
```

---

## 6. PHASES (do them in order; stop after each)

### Phase 0 — Project setup
- Create the uv project, folder structure, `.gitignore`, `ruff` config, empty modules with docstrings.
- Pin Python 3.11 (or tell me why another version is required for TensorFlow/AutoKeras compatibility).
- Write a minimal README (what, why, how to run).

**Done when:** `uv sync`, `uv run pytest` (a placeholder test) and `uv run ruff check .` all pass.

### Phase 1 — Data (Module 1)
- Load mandi CSV(s) from `data/raw/` (I will place them there; source: CEDA Ashoka or Kaggle 2023–2025; record which one I used in the README).
- Clean: drop duplicates, standardize market names, flag outliers (explain the rule you chose).
- Select onion and 2–3 markets with the most data.
- Aggregate to weekly mean modal price (₹/quintal).
- Data-quality report: rows per market, % missing weeks, a plot of the weekly series.
- **Stop rule:** if more than ~10% of weeks are missing for the chosen markets, stop and tell me before continuing.

**Done when:** `uv run python -m automl.data.aggregate` produces `data/processed/weekly.parquet` and a quality report.

### Phase 2 — Features and time windows (Module 1)
- Features: lags (1, 2, 4, 8 weeks), rolling means (4, 8 weeks), month, season, market as a category.
- Target: next week's price.
- Split the timeline into 4–6 windows. Each window has its own train / validation / test portions, strictly in time order.
- Tests that prove there is no future leakage.

**Done when:** `test_features.py` and `test_split.py` pass, and you show me a table of window date ranges.

### Phase 3 — Baselines (Modules 2 & 3)
- Naive baselines: last week's price; 4-week mean.
- LightGBM with default settings.
- AutoKeras `StructuredDataRegressor` (small `max_trials`, to fit on CPU).
- KerasTuner MLP with a custom search space + one custom block (a tunable choice of which lag features to use).

**Done when:** `results/baselines.csv` has MAE and RMSE for every baseline on window 1.

### Phase 4 — Custom Bayesian optimization tuner (Module 4)
Build it step by step, explaining each step before coding it:
1. `space.py` — convert hyperparameters ↔ vectors (log scale for learning rate; integers rounded; ranges from `search_space.yaml`).
2. `surrogate.py` — Gaussian Process (scikit-learn `GaussianProcessRegressor` with a Matern kernel is fine), fitted on normalized vectors → validation error.
3. `acquisition.py` — Expected Improvement, written by us.
4. `bo_tuner.py` — loop: a few random initial trials → fit surrogate → maximize EI over random candidates → evaluate → repeat until the budget is used.
5. `history.py` — save every trial (vector, params, score, window id, timestamp) to JSON; load it back to **resume** the search.
6. Warm start = load history from earlier windows into the surrogate before the first new trial. Implement two variants: (a) all past trials, (b) only the top-k past configurations used as the initial points.

Compare our tuner against random search with the same budget on window 1.

**Done when:** `test_tuner.py` passes (round-trip, EI non-negative, resume gives identical history) and our BO is shown against random search.

### Phase 5 — Drift experiment (core result)
- At each window boundary: run the drift check (input drift + rise in error), log the result.
- On every window after W1, compare four strategies:
  1. No retrain
  2. Refit (old best hyperparameters, new data)
  3. Cold BO (fresh search)
  4. Warm BO (variants a and b)
- Equal trial budget (start with 15), at least 5 seeds.
- Metrics: MAE, RMSE (₹/quintal), trials needed to reach a target error, wall-clock time.
- Save `results/drift_experiment.csv`; `summarize.py` makes tables (mean ± std) and plots (error vs trial number, cold vs warm).

**Done when:** results exist from real runs, and you give me a short plain-language summary that keeps **facts** (numbers) separate from **interpretation** (what they might mean).

### Phase 6 — Scaling (Module 5)
- Run the trials of one search in parallel on CPU cores (joblib or multiprocessing).
- Measure the speed-up for 1, 2 and 4 workers.
- Explain (in the README, not in code) how data parallelism and model parallelism would apply with GPUs, and why they don't fit this tabular task.

**Done when:** `results/parallel.csv` contains the timings.

### Phase 7 — Model registry and web app
- `registry.py` saves the best model as `models/model.pkl` plus `model_card.json` (version, training window, metrics, date, git commit).
- Flask app:
  - `/` — form: choose a market → shows next week's predicted price
  - `/predict` — JSON API
  - `/health` — returns `{"status": "ok"}`
  - `/model` — returns the model card
- `test_app.py` passes.

**Done when:** `uv run flask --app app.app run` works locally.

### Phase 8 — Docker and Jenkinsfile (hand-off to DevOps)
- `Dockerfile`: slim Python base, copy `uv` from its official image, `uv sync --frozen --no-dev`, run with gunicorn on port 8000.
- `Jenkinsfile` stages: Checkout → Lint & Test (`uv run ruff check .`, `uv run pytest`) → Docker build (tag = short commit hash) → Push to GHCR → Deploy (checkout `mandi-infra`, run `ansible-playbook ansible/deploy.yml -e image_tag=<hash>`) → Verify (`curl /health`).
- Credentials are referenced by Jenkins credential IDs only; never hard-coded.

**Done when:** the image builds and runs locally, and `/health` returns 200 from the container.

### Phase 9 (optional) — Generality check (Module 6)
- Reuse the custom tuner on MNIST (small CNN or MLP) or 20 Newsgroups (TF-IDF + linear model), with a tiny budget.

### Phase 10 — Documentation
- README: problem, data source, method, how to reproduce each result with `uv run`, results table, limitations, prior work with publication type.
- Draft report outline mapping each section to the six course modules.

---

## 7. CODE STANDARDS

- Python 3.11, type hints, docstrings on every public function.
- Configuration comes from YAML in `configs/` — no magic numbers inside functions.
- Use `logging`, not `print`, in `src/`.
- Small functions; one responsibility each.
- `ruff` clean; tests for every bug fixed.
- Every script has a `--config` argument and writes its outputs to a known path.

---

## 8. HOW TO TALK TO ME

- Before each phase: explain in 3–6 simple sentences what we are about to build and why.
- After each phase:
  - what was built (files)
  - commands to run it
  - test results
  - anything uncertain or assumed
  - the question you need me to answer, if any
- Keep **facts** (what the code produced, what a source says) separate from **interpretation** (your opinion).
- When citing a paper or tool, say whether it is peer-reviewed or a preprint. Only cite things you have actually verified.
- If you are unsure, say so. Never guess and present the guess as fact.

---

## 9. DEFINITION OF DONE (whole project)

- [ ] `uv sync && uv run pytest` passes on a fresh clone
- [ ] Every number in the README can be reproduced with a documented `uv run` command
- [ ] Drift experiment: 4 strategies × all windows × ≥5 seeds, real results
- [ ] Custom BO tuner with working save/resume and warm start
- [ ] AutoKeras and KerasTuner baselines reported
- [ ] Parallel-trials timing reported
- [ ] Web app + Dockerfile + Jenkinsfile working
- [ ] README states limitations and prior work honestly

---

## APPENDIX — deviations from this spec, as actually built

*Not part of the original prompt. Added 2026-10-20 so the spec above can stay verbatim
while the record of what changed stays with it. Each item was measured or forced, not
chosen for convenience.*

1. **Entity is a state, not a mandi.** CEDA's district endpoint returns HTTP 500/504 and
   `agmarknet.gov.in` v2 requires a captcha, which was not bypassed. The only working
   programmatic source is state-level, so `market` holds a state-wide aggregate of mandi
   modal prices. Originally three states; see item 10 for the current scope.
2. **Target is parameterised as a week-on-week delta** (`configs/features.yaml`,
   `target.mode: delta`), reconstructed to a price by adding `lag_1` back. The forecast
   target is unchanged — still next week's price in ₹/quintal, and every metric is
   reported on the price scale. Measured: level 200.6 mean test MAE, delta 142.8, naive
   141.6. Trees cannot extrapolate past their training range, which is why level loses.
3. **Window blocks are 26 weeks, not 12.** Twelve-week blocks gave only 36 rows per
   block across three markets, too few to select hyperparameters from.
4. **Docker uses `uv sync --no-default-groups`, not `--no-dev`.** `--no-dev` excludes
   only the `dev` group and would still install the 2 GB research stack. `pyarrow` was
   also moved out of the runtime group; serving rows travel as a joblib pickle.
5. **Five strategies are compared, not four.** The two warm-start variants are reported
   separately (`warm_bo_all`, `warm_bo_topk`) rather than merged.
6. **Files added beyond the listed structure:** `src/automl/config.py`,
   `metrics.py`, `evaluate.py`, `pipeline.py`, `data/fetch.py`,
   `baselines/naive.py`, `experiments/run_baselines.py`,
   `experiments/run_tuner_check.py`, `experiments/run_generality.py`,
   `configs/generality.yaml`, `docs/report_outline.md`, and six extra test modules.
7. **AutoKeras 3.0.0 needed a compatibility shim.** It rejects pandas DataFrames and
   hands Keras 3 a `np.int64` where `Dense` requires a Python `int`. See
   `src/automl/baselines/autokeras_baseline.py`. The textbook targets AutoKeras 1.0.x.
8. **Feurer et al. citation checked (2026-10-20): preprint only.** No journal reference
   or publisher DOI on arXiv across four versions. Rule 8's instruction to check before
   citing has been carried out; the answer is that it must be cited as a preprint.

### Outcome against the research question

**Measured three times as the scope widened, with the answer moving each time. All
three are reported, because the movement is the most informative thing here.**

| Run | Scope | Winner | Warm vs refit |
| --- | --- | --- | --- |
| 2026-10-18 | onion, 3 states, 36 validation rows | `refit` (153.4) | no benefit |
| 2026-10-20 | onion, 6 states, 156 validation rows | `warm_bo_all` (131.0) | warm wins 14/20, p=0.027 |
| 2026-10-20 | **4 crops**, 6 states each, 80 matched runs | `warm_bo_all` by a hair (127.5) | warm wins 45/80, p=0.15 |

Taking the three claims of the research question separately, on the full four-crop run:

- **vs cold start — modest yes.** `warm_bo_all` won 48/80 matched runs (raw p=0.039),
  `warm_bo_topk` 46/80 (p=0.044), and warm has the lower mean in 3 of the 4 crops.
- **vs refit — no.** 45/80, p=0.15.
- **vs no retraining — no.** 33/80, p=0.42. Doing nothing is hard to beat.
- The cold search *lost* to doing nothing in 52 of 80 runs (p=0.015 in no-retrain's
  favour), so on this data hyperparameter search is a poor use of effort at all.
- **Nothing survives Holm-Bonferroni**, pooled or per crop. Pooled strategy means span
  under two rupees per quintal.

The crops disagree: onion and tomato favour warm starting, potato and cabbage favour a
plain refit. The single-crop result was real but did not generalise, and Rule 5 requires
saying so rather than quoting the most flattering of the three runs. `README.md` does.

### Later changes, requested by the user on 2026-10-20

These came after the spec above was satisfied, and widen it rather than correct it.

9. **Four crops, not one.** Onion, Potato, Tomato and Cabbage. The spec's single-crop
   scope was a deliberate narrowing, not a data limit — the portal exposes 453
   commodities. `configs/data.yaml` now drives this; adding a crop is a config change.
10. **Eight states, not three.** `n_markets: 6` per crop, selected independently for
    each crop from eight candidates.
11. **One model per crop.** `models/<slug>/model.pkl` plus a card and serving rows
    each, with `models/index.json` as the registry the API reads. Prices differ by an
    order of magnitude between crops, so separate models stay interpretable.
12. **FastAPI replaces Flask.** `app/main.py` holds the routes, `app/service.py` the
    live data and model access, `app/cli.py` the `web-start` entry point. Routes moved
    under `/api/*`, so the health check is now `/api/health`. `/docs` comes free.
13. **`uv run web-start` starts everything.** Backend and frontend are one uvicorn
    process on one port: FastAPI serves both the JSON API and the page that consumes
    it. No CORS, no Node, no second server. The command runs preflight checks and
    prints the exact fix for anything missing.
14. **Nothing on the page is pre-written.** Every figure is computed per request from
    the dataset on disk. `/api/backtest` re-runs the registered model across recent
    weeks and scores it against the naive forecast on the same rows, so the page
    reports honestly whether the model is beating the baseline for that selection.
    The service reloads data and models automatically when the files change.
15. **The container carries `models/weekly_snapshot.csv`, not a parquet.** The API
    computes from the dataset at request time, and shipping it as CSV keeps `pyarrow`
    (~170 MB) out of the runtime image.
16. **Docker runs uvicorn**, not gunicorn, since FastAPI is ASGI.
17. **The drift experiment now covers all four crops** (420 rows), with per-crop copies
    in `results/by_crop/`. `summarize.py` reports the per-crop split before any pooled
    number, and pairs on (crop, seed, window) so the matching survives pooling.
18. **Baselines, BO-vs-random and the parallel timings also cover all four crops.**
    Each experiment stamps its crop on every row; per-crop files are in
    `results/by_crop/` and the top-level CSVs are the concatenation.
19. **The report is generated, not hand-written.** `docs/report.md` is produced by
    `uv run python -m automl.experiments.make_report`, which interpolates every figure
    from `results/`. A number typed into prose can drift from its CSV unnoticed; this
    cannot. Tests assert the generator builds with no results present, leaves no
    unresolved placeholder, mentions all six course modules, and keeps the required
    "not a new algorithm" framing. Section numbering follows `docs/report_outline.md`.
20. **BO does not reliably beat random search at this budget** (10 of 20 crop-seed pairs
    on validation, tied on test, ~1.5x the cost). The earlier onion-only run suggested
    otherwise. Reported in both README and report; Rule 5 applies to the tuner too.

### Deployment, 2026-10-26

21. **The companion infrastructure repo is `devops_IA3`, not `mandi-infra`.** Section 2
    names `mandi-infra`; the repository actually created is
    `github.com/dharshan-kumarj/Devops_IA3`, holding `terraform/` and `ansible/`. The
    split itself is unchanged: this repo keeps the `Dockerfile` and `Jenkinsfile`, that
    one owns the Azure resources.

22. **Jenkins runs locally and polls, rather than receiving a webhook.** A local
    install behind NAT cannot be reached by GitHub. `pollSCM('H/2 * * * *')` is declared
    in the `Jenkinsfile`. Terraform is run by hand once; Jenkins only ever deploys a new
    container onto the VM that already exists, so no Azure service principal is needed
    and a push can never destroy infrastructure.

23. **The Docker Build stage no longer requires `data/processed/weekly.parquet`.** It
    did, and that file is git-ignored by design and excluded by `.dockerignore`, so the
    check would have failed every Jenkins build on a clean checkout while saying nothing
    about the code. The image serves from `models/` alone. Same failure mode as the
    report test fixed in `768de3b`, found the same way: by running the stage against a
    clean extraction of tracked files only.

### Definition of Done — verified 2026-10-20

Checked against a genuine `git clone` of this repository into an empty directory, with
no `data/` present:

| Item | Status |
| --- | --- |
| `uv sync && uv run pytest` passes on a fresh clone | **Verified 2026-10-20.** Tracked files extracted into an empty directory; tests pass, `ruff check` clean, and `uv run web-start` serves immediately because the models and `models/weekly_snapshot.csv` are committed. |
| Every README number reproducible with a documented `uv run` command | **Verified.** Command table in `README.md`; `data/raw` must be refetched first via `automl.data.fetch`. |
| Drift experiment: strategies × all windows × ≥5 seeds, real results | **Verified.** 4 crops × 5 strategies × 5 windows × 5 seeds = 420 rows in `results/drift_experiment.csv`, per-crop copies in `results/by_crop/`. |
| Custom BO tuner with working save/resume and warm start | **Verified.** `tests/test_tuner.py`, 33 tests, incl. exact resume and both warm variants. |
| AutoKeras and KerasTuner baselines reported | **Verified.** `results/baselines.csv`, MAE 56.9 and 59.4 (onion). |
| Parallel-trials timing reported | **Verified.** `results/parallel.csv`, 1/2/4/8 workers. |
| Web app + Dockerfile + Jenkinsfile working | **Verified.** FastAPI; image builds at 754 MB; `/api/health` 200 and `/api/predict` returns a live price from the container. |
| README states limitations and prior work honestly | **Verified.** Thirteen limitations listed; the per-crop disagreement and the three conflicting runs are stated before any favourable number. |
