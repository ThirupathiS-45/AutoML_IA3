# mandi-automl

Drift-aware AutoML for weekly onion **mandi** (wholesale market) price forecasting in India.

> **Status: Phase 0 (project setup).** No data has been loaded and no experiment has been
> run yet. Every results table below is empty on purpose — numbers appear here only after
> the code has actually produced them.

## What this is

A forecaster for next week's onion modal price (₹/quintal) in a handful of Indian mandis,
tuned by a **Bayesian optimizer written from scratch** in this repo. When the market drifts,
the tuner can **warm-start** from the trial history of earlier time windows instead of
searching from nothing.

## Why

Deployed models quietly get worse when the world changes, and small teams rarely retrain
them well. The question is whether reusing past tuning effort buys back accuracy faster
than the obvious alternatives.

## Research question

After real market drift, does warm-started Bayesian search recover forecast accuracy in
fewer trials than (a) no retraining, (b) refitting the previous best settings, and
(c) a cold-start search — under a small, equal trial budget?

## Honest framing

This is an **empirical evaluation plus an end-to-end system, not a new algorithm.** The
idea of adapting AutoML to evolving data, and of transferring tuning history between
related tasks, is established prior work:

| Work | Type |
| --- | --- |
| Celik & Vanschoren, *Adaptation Strategies for Automated Machine Learning on Evolving Data*, IEEE TPAMI 43(9), 2021 | Peer-reviewed journal |
| Celik, Singh & Vanschoren, *Online AutoML: an adaptive AutoML framework for online learning*, Machine Learning 112, 2023 | Peer-reviewed journal |
| Wu et al., *ChaCha for Online AutoML*, ICML 2021 | Peer-reviewed conference |
| Feurer, Letham, Hutter & Bakshy, *Practical Transfer Learning for Bayesian Optimization*, arXiv:1802.02219 | Preprint — a peer-reviewed version has **not yet been verified**; to be checked before the report cites it |

## Data source

**Not yet selected.** Raw CSVs go in `data/raw/` (git-ignored). The exact source
(CEDA Ashoka or Kaggle, 2023–2025) and the download date will be recorded here in Phase 1.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.11 is pinned in `.python-version`.

```bash
uv sync                    # runtime + dev dependencies
uv sync --group research   # adds TensorFlow / AutoKeras / KerasTuner / Evidently
```

## Running things

```bash
uv run pytest              # tests
uv run ruff check .        # lint
```

Phase-specific commands are added to this section as each phase lands.

## Why Python 3.11

Checked against PyPI metadata on 2026-10-18: `tensorflow` 2.21.0 requires `>=3.10`,
while `autokeras` 3.0.0 and `keras-tuner` 1.4.8 publish Python classifiers only through
**3.11**. Python 3.11 is the one version all three support, so `requires-python` is
pinned to `>=3.11,<3.12`.

Note that AutoKeras 3.0.0 is a different generation from the 1.0.x line used in the
course textbook (*Automated Machine Learning in Action*, Manning 2022): it targets
Keras 3 and has dropped `TimeseriesForecaster`. `StructuredDataRegressor`, which this
project needs, is still present. Expect API differences from the book's listings.

## Dependency split

Runtime dependencies (`pandas`, `lightgbm`, `flask`, …) ship in the Docker image.
Research-only dependencies (`tensorflow`, `autokeras`, `keras-tuner`, `evidently`,
`matplotlib`, `jupyter`) live in the `research` group and are **excluded** from the
image, which keeps it small — the served model is LightGBM and needs no deep learning
stack.

## Repository layout

```
configs/       YAML configuration — no magic numbers live in code
src/automl/    the library: data, features, split, drift, tuner, baselines, experiments
app/           Flask app serving the current model
results/       experiment CSVs and plots (committed, small)
models/        model.pkl (git-ignored) + model_card.json (committed)
tests/         leakage, split-ordering, tuner and app tests
notebooks/     exploration only; never imported by src/
```

Infrastructure (Terraform + Ansible) lives in a separate `mandi-infra` repo. This repo
contains only the `Dockerfile` and `Jenkinsfile` it is deployed with.

## Results

Not run yet.

## Limitations

To be written once there are results to be honest about. The framing commitment stands:
if warm starting does not beat a plain refit, that is what this README will say.
