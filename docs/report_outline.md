# Report outline — mandi-automl

Each section names the course module it demonstrates and the artefact in this repo
that backs it. Nothing here should be written up before the matching command has
actually been run.

---

## 1. Introduction and problem framing — Module 1

- The operational problem: deployed price models decay silently when the market moves,
  and small teams rarely have the budget to retrain them properly.
- The forecasting task: next week's onion modal price, rupees per quintal.
- The research question, stated as a comparison under equal budget.
- Honest positioning: an empirical evaluation plus a working system, not a new
  algorithm.

*Backed by:* `README.md`

## 2. Prior work

- Celik & Vanschoren 2021 (IEEE TPAMI, peer-reviewed) — adaptation strategies for
  AutoML on evolving data. Closest prior art; establishes that the question matters.
- Celik, Singh & Vanschoren 2023 (Machine Learning, peer-reviewed) — online AutoML.
- Wu et al. 2021 (ICML, peer-reviewed) — ChaCha for online AutoML.
- Feurer, Letham, Hutter & Bakshy, arXiv:1802.02219 (preprint; check for a published
  version before citing) — transfer learning for Bayesian optimization, which is the
  mechanism warm starting relies on.
- What this project adds: an end-to-end deployment of those ideas on Indian mandi
  price data, with the tuner written from scratch rather than imported.

## 3. Data and preprocessing — Module 1

- Source, licence and download date; the state-level caveat stated plainly.
- Cleaning rules and why each was chosen, including what was *not* removed.
- Weekly aggregation and the missing-week stop rule.
- Coverage table and the plotted series.

*Backed by:* `uv run python -m automl.data.fetch`,
`uv run python -m automl.data.aggregate`, `results/data_quality.md`

## 4. Feature engineering and evaluation design — Module 1

- The target-week indexing convention and why it makes leakage testable.
- Lags, rolling statistics, calendar features, market as a category.
- Window layout: train / validation / test, strictly forward in time.
- The leakage tests, including the perturbation test, as evidence rather than
  assertion.

*Backed by:* `src/automl/features.py`, `src/automl/split.py`,
`uv run pytest tests/test_features.py tests/test_split.py`

## 5. Baselines — Modules 2 and 3

- Naive forecasters, and why a model that cannot beat "same as last week" is useless.
- Untuned LightGBM.
- AutoKeras `StructuredDataRegressor` (Module 2): what it automates, what it costs.
- KerasTuner MLP with a custom search space and a custom block that treats lag
  selection as a hyperparameter (Module 3).
- Note the AutoKeras 3.0 / Keras 3 divergence from the textbook's 1.0.x API.

*Backed by:* `uv run python -m automl.experiments.run_baselines --with-deep`,
`results/baselines.csv`

## 6. The custom Bayesian optimizer — Module 4

- Vectorising hyperparameters: unit-box normalisation, log axes, integer rounding.
- The Gaussian Process surrogate and why the Matern kernel and score standardisation
  matter.
- Expected Improvement, derived in words, with each symbol explained.
- The search loop and the random-candidate maximisation of the acquisition function.
- History persistence, resume, and the two warm-start variants.
- Evidence it works: our BO against random search under identical budget.

*Backed by:* `src/automl/tuner/`, `uv run pytest tests/test_tuner.py`

## 7. The drift experiment — core result

- What a window boundary represents operationally.
- The two drift signals: KS tests on inputs with Holm-Bonferroni correction, and the
  rise in deployed-model error.
- Five strategies, equal budget, five seeds.
- **Facts**: mean and standard deviation of test MAE and RMSE per strategy and window;
  trials needed to match the cold search; wall-clock time.
- **Interpretation**, kept in its own subsection and clearly labelled as such.
- Negative results reported as found.

*Backed by:* `uv run python -m automl.experiments.run_drift`,
`uv run python -m automl.experiments.summarize`, `results/summary.md`

## 8. Scaling — Module 5

- Why Bayesian optimization resists parallelism, and what batch BO gives up to get it.
- Measured wall-clock and speed-up for 1, 2 and 4 workers.
- Discussion of data and model parallelism on GPUs, and why neither helps a few
  hundred rows of tabular data.

*Backed by:* `uv run python -m automl.experiments.run_parallel`, `results/parallel.csv`

## 9. Deployment — Module 6

- Model registry and the model card as the unit of accountability.
- Flask service and its four routes.
- Container design: why the research stack is excluded from the image.
- The CI/CD path: Jenkins → GHCR → Ansible → Azure VM, with `mandi-infra` owning the
  infrastructure.

*Backed by:* `src/automl/registry.py`, `app/`, `Dockerfile`, `Jenkinsfile`

## 10. Generality check — Module 6

- The same tuner, unmodified, on 20 Newsgroups text classification.
- Against random search at equal budget.

*Backed by:* `uv run python -m automl.experiments.run_generality`,
`results/generality.csv`

## 11. Limitations

- Entity level is a state aggregate, not an individual mandi.
- Small number of windows and seeds; wide confidence intervals.
- "Drift" is observed, not induced, so its size and timing were not controlled.
- A single commodity and a single country.
- The trial budget of 15 is small by design and the conclusions do not extend to
  larger budgets.

## 12. Conclusion and future work
