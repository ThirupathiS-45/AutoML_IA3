# Prompt for generating the skill-based evaluation report

Paste everything inside the fenced block below into an LLM (or hand it to a
teammate) to produce the course report in the institute's format. The facts
section is filled in from this repository's actual outputs, so the model has no
reason to invent numbers — and anything it adds that is not in that section
should be treated as suspect.

A filled-in version already exists at `docs/skill_evaluation_report.md`. Use
this prompt to regenerate it, to produce a teammate's copy, or to retarget the
same content at a different rubric.

**Before using it:** replace every `<FILL: ...>` marker with your own details.
Those are the only things the repository cannot tell you.

---

````text
You are helping a final-year B.Tech student write a skill-based evaluation
report for an AutoML course. Produce a complete report in Markdown, ready to
paste into Google Docs or Word and export as PDF.

## Required structure (follow this order exactly)

1. Title page block — institute, division, title, student name(s), register
   number(s), course code, course name, month and year
2. ABSTRACT — one page, no headings inside, no bullet points
3. CHAPTER 1 — INTRODUCTION
4. CHAPTER 2 — LITERATURE REVIEW
5. CHAPTER 3 — SYSTEM DESIGN
   5.1 System Architecture
   5.2 High-Level Design
   5.3 UML Diagrams (class diagram description, sequence diagram description)
   5.4 User Interface Design
   5.5 Summary
6. CHAPTER 4 — IMPLEMENTATION
7. CHAPTER 5 — TESTING AND VALIDATION
8. CHAPTER 6 — RESULTS AND DISCUSSION
   8.1 Presentation of the Final System
   8.2 Evaluation of Project Objectives
   8.3 Challenges Faced
   8.4 Summary
9. CONCLUSION
10. REFERENCES
11. APPENDICES
12. EVALUATION SHEET (reproduce the rubric table with marks left blank)

## Absolute rules

- **Use only the numbers in the FACTS section below.** Do not round them
  differently, do not average them into new figures, and do not invent any
  metric that is not listed. If a number is needed that is not there, write
  "not measured" instead.
- **Report the negative results.** This project's honest finding is that the
  headline effect is weak and does not survive multiple-comparison correction.
  Say so plainly, in the Abstract and in Chapter 6. Do not quote the most
  flattering of the three runs as if it were the result.
- **State the framing**: this is an empirical evaluation plus an end-to-end
  system, not a new algorithm. This must appear in the Abstract and the
  Conclusion.
- Label every citation as peer-reviewed or preprint.
- Keep facts (what the code produced) separate from interpretation (what it
  might mean). Chapter 6 should present the table first and the reading second.
- Academic register, third person, no contractions. No marketing language — no
  "revolutionary", "cutting-edge", "seamless", "powerful".
- Each chapter should be substantial prose, not bullet lists, except where a
  table is genuinely the clearest form.

## PROJECT IDENTITY

- Title: Drift-Aware AutoML for Weekly Crop Price Forecasting with a
  Hand-Written Bayesian Optimisation Tuner
- Student: <FILL: your name> (<FILL: your register number>)
- Team: <FILL: teammate names and register numbers, or delete if solo>
- Course code: <FILL: e.g. 24DC2005>   Course name: AutoML
- Institute: <FILL: e.g. Karunya Institute of Technology and Sciences>
- Division: <FILL: e.g. Division of Data Science and Cyber Security>
- Month/Year: <FILL: e.g. October 2026>
- Repositories: github.com/dharshan-kumarj/AutoML_IA3 (application),
  github.com/dharshan-kumarj/Devops_IA3 (Terraform + Ansible)

## RESEARCH QUESTION

After real market drift, does a warm-started Bayesian search recover forecast
accuracy in fewer trials than (a) no retraining, (b) refitting the previous best
hyperparameters on new data, and (c) a cold-start search, under a small and
equal trial budget?

## FACTS — the only numbers you may use

### Data
- Source: CEDA Agmarknet portal, Ashoka University. Underlying data: Agmarknet,
  Ministry of Agriculture and Farmers Welfare, Government of India.
- Crops: Onion, Potato, Tomato, Cabbage (4).
- Entity level: STATE — a state-wide aggregate of mandi modal prices, not a
  single mandi. CEDA's district endpoint returns HTTP 500/504 and the
  agmarknet.gov.in v2 interface requires a CAPTCHA, which was not bypassed.
  This is the project's principal data limitation and must be stated.
- 8 candidate states, 6 selected per crop by coverage.
- Range 2018-01-01 to 2025-10-30; weekly frequency W-MON; unit rupees/quintal.
- 409 weeks per series. Most selected series miss 5 of 409 weeks (1.22%).
- Bihar was excluded for every crop: 46.70% missing (Onion), 45.23% (Potato),
  79.95% (Cabbage).
- Target: next week's price, parameterised internally as a week-on-week delta
  and reconstructed by adding lag_1 back. Every metric is reported on the price
  scale. Measured mean test MAE: level 200.6, delta 142.8, naive 141.6 — trees
  cannot extrapolate beyond their training range, which is why level loses.
- Features: lags 1, 2, 4, 8 weeks; rolling means over 4 and 8 weeks; month;
  season; market as a category. All past-only.
- 5 expanding time-ordered windows; 26-week validation and test blocks;
  minimum 104 training weeks. 156 test rows per crop on window 1.

### Baselines, window 1, 156 test rows, MAE in rupees/quintal
| Crop | naive last week | naive 4-week mean | LightGBM default | AutoKeras | KerasTuner MLP |
|---|---|---|---|---|---|
| Onion | 60.63 | 105.31 | 66.56 | 56.88 | 59.36 |
| Potato | 30.11 | 43.63 | 32.78 | 28.95 | 29.70 |
| Tomato | 598.26 | 1368.44 | 483.80 | 457.02 | 461.25 |
| Cabbage | 99.96 | 164.09 | 98.70 | 93.53 | 94.69 |

AutoKeras used max_trials 3, 30 epochs. KerasTuner used max_trials 8, 40 epochs
and a custom block that selects which lag features to keep; it chose lag_1 for
every crop. AutoKeras beats the naive baseline on all four crops on window 1.

### Drift experiment
4 crops x 5 strategies x 5 windows x 5 seeds = 420 rows. Trial budget 15 per
search, 5 random initial points. 80 matched runs per strategy on windows 2-5.

Mean test MAE (SD) across post-drift windows:
| Strategy | MAE | SD | mean trials | mean seconds |
|---|---|---|---|---|
| no_retrain | 129.05 | 70.41 | 0 | 0.00 |
| refit | 129.52 | 74.44 | 0 | 0.10 |
| cold_bo | 129.29 | 71.29 | 15 | 2.37 |
| warm_bo_all | 127.50 | 71.27 | 15 | 4.69 |
| warm_bo_topk | 127.76 | 70.79 | 15 | 2.49 |

Per crop — THE CROPS DISAGREE, and this must be reported before any pooled number:
| Crop | no_retrain | refit | cold_bo | warm_bo_all | warm_bo_topk | best |
|---|---|---|---|---|---|---|
| Cabbage | 92.8 | 92.7 | 94.5 | 94.4 | 93.3 | refit |
| Onion | 144.5 | 137.9 | 135.6 | 131.0 | 132.4 | warm_bo_all |
| Potato | 69.8 | 70.0 | 71.2 | 71.3 | 71.4 | no_retrain |
| Tomato | 209.2 | 217.6 | 215.9 | 213.2 | 213.9 | no_retrain |

Paired Wilcoxon signed-rank on matched (crop, seed, window) triples, n=80,
Holm-Bonferroni correction over 7 comparisons:
| Comparison | wins | mean diff | p | significant after Holm |
|---|---|---|---|---|
| cold_bo vs no_retrain | 28/80 | +0.24 | 0.0146 | No |
| warm_bo_all vs cold_bo | 48/80 | -1.79 | 0.0392 | No |
| warm_bo_topk vs cold_bo | 46/80 | -1.53 | 0.0436 | No |
| warm_bo_all vs refit | 45/80 | -2.02 | 0.1515 | No |
| warm_bo_all vs no_retrain | 33/80 | -1.55 | 0.4176 | No |
| cold_bo vs refit | 35/80 | -0.22 | 0.5000 | No |
| refit vs no_retrain | 39/80 | +0.47 | 0.5019 | No |

NOTHING survives Holm-Bonferroni correction, pooled or per crop. Pooled
strategy means span under two rupees per quintal.

Trials needed to match the cold search's own final validation score:
| Strategy | reached | median trials when reached |
|---|---|---|
| cold_bo | 80/80 | 11.5 |
| warm_bo_all | 26/80 | 6 |
| warm_bo_topk | 40/80 | 7 |
(cold_bo reaches it by definition, since the target is its own score. Only the
warm rows carry information.)

Drift detection at each window boundary — two-sample Kolmogorov-Smirnov tests
per feature with Holm-Bonferroni correction, plus an error-rise ratio:
| Window | drifted features | tested | input drift | error ratio |
|---|---|---|---|---|
| 2 | 9 | 9 | True | 1.646 |
| 3 | 9 | 9 | True | 1.289 |
| 4 | 9 | 9 | True | 0.898 |
| 5 | 9 | 9 | True | 0.740 |

### Custom tuner against random search
Window 1, equal budget of 15 trials, 20 crop-seed pairs. The custom Bayesian
optimiser had the lower validation MAE in 10 of 20 pairs — a coin flip — and the
two were tied on test error, at roughly 1.5x the wall-clock cost. An earlier
onion-only run had suggested the tuner was clearly ahead; it does not replicate.

### Parallel tuning, budget 24 trials, 12 runs per worker count
| Workers | mean speed-up | max speed-up |
|---|---|---|
| 1 | 1.000 | 1.000 |
| 2 | 0.990 | 1.468 |
| 4 | 1.474 | 1.869 |
| 8 | 1.307 | 1.525 |
Eight workers are slower than four. Each trial takes about 0.1-0.2 s, so
process start-up and data pickling dominate beyond four workers.

### Generality check — 20 Newsgroups, TF-IDF + SGD, budget 12, 3 seeds
| Method | best CV error | test accuracy | seconds |
|---|---|---|---|
| custom_bo | 0.1109 | 0.8621 | 13.44 |
| random_search | 0.1164 | 0.8623 | 11.26 |
The tuner transfers to a different task and data type, finds a slightly better
cross-validation score, and ends up with the same test accuracy.

### Registered production models
| Crop | test MAE | test RMSE | test MAPE |
|---|---|---|---|
| Potato | 42.55 | 63.50 | 2.82% |
| Onion | 73.93 | 96.07 | 5.92% |
| Cabbage | 75.59 | 105.95 | 5.53% |
| Tomato | 201.10 | 258.68 | 9.16% |

### System and deployment
- Python 3.11, uv for dependency management, uv.lock committed.
- LightGBM for the model; scikit-learn GaussianProcessRegressor with a Matern
  (nu=2.5) kernel as the surrogate; Expected Improvement written by hand.
- FastAPI serves the JSON API and the HTML page from one uvicorn process on one
  port. Routes: /, /api/health, /api/meta, /api/predict, /api/history,
  /api/backtest, /api/model/{crop}, /docs. Started with `uv run web-start`.
- Nothing on the page is pre-computed; every figure is calculated per request
  from the dataset on disk, and the service reloads when files change.
- One model per crop at models/<slug>/, with models/index.json as the registry.
- 188 tests pass locally; 187 pass and 1 is skipped in a clean CI checkout
  (the skipped one needs a git-ignored dataset file).
- Docker: multi-stage, python:3.11-slim-bookworm, `uv sync --frozen
  --no-default-groups` so TensorFlow never enters the image, non-root user
  10001, image size 754 MB, health check on /api/health.
- Jenkins runs locally with a declarative pipeline of 7 stages: Checkout,
  Lint & Test, Docker Build, Smoke Test Image, Push to GHCR, Deploy, Verify
  Deployment. Triggered by Poll SCM every two minutes, because a local Jenkins
  behind NAT cannot receive a GitHub webhook.
- Terraform provisions 8 Azure resources: Resource Group, Virtual Network,
  Subnet, Network Security Group, NSG-subnet association, Public IP (Static),
  Network Interface, and a Standard_B1ms Linux VM running Ubuntu 22.04 in the
  indiasouthcentral region. cloud-init installs Docker on first boot.
- Ansible deploys by pulling the built image from GHCR onto the VM and
  restarting the container with host port 80 mapped to container port 8000. The
  VM never receives source code and never builds anything; it has 1898 MB of
  RAM, and building a 754 MB image there would exhaust it.
- Credentials are referenced only by Jenkins credential ID: ghcr-credentials,
  azure-vm-ssh, azure-vm-ip. No secret appears in either repository.

### Prior work to cite (label each correctly)
- Celik and Vanschoren, "Adaptation Strategies for Automated Machine Learning
  on Evolving Data", IEEE TPAMI 43(9), 2021 — PEER-REVIEWED JOURNAL
- Celik, Singh and Vanschoren, "Online AutoML: an adaptive AutoML framework for
  online learning", Machine Learning 112, 2023 — PEER-REVIEWED JOURNAL
- Wu et al., "ChaCha for Online AutoML", ICML 2021 — PEER-REVIEWED CONFERENCE
- Feurer, Letham, Hutter and Bakshy, "Practical Transfer Learning for Bayesian
  Optimization", arXiv:1802.02219 — PREPRINT ONLY. Checked across four arXiv
  versions on 2026-10-20: no journal reference and no publisher DOI. Must be
  cited as a preprint.
- Song, Jin and Hu, "Automated Machine Learning in Action", Manning, 2022 —
  the course textbook.

### Limitations that must appear in the report
1. State-level aggregates, not individual mandis.
2. One commodity portal; no weather, festival, fuel or import/export features.
3. Five windows and five seeds — enough for a paired test, not enough to
   settle the question.
4. No effect survives multiple-comparison correction.
5. The result moved three times as scope widened (onion/3 states: refit wins;
   onion/6 states: warm wins p=0.027; 4 crops: warm wins by under 2 rupees,
   p=0.15). The movement is the most informative outcome of the project.
6. The custom tuner does not reliably beat random search at this budget.
7. Doing nothing at all beat a cold search in 52 of 80 runs, so on this data
   hyperparameter search is a poor use of effort regardless of warm starting.

## EVALUATION SHEET to reproduce
Reg. No, Name, Course code, Course Name, then this table with marks blank:
| S.No | Rubrics | Maximum marks | Marks Obtained |
| 1 | Problem Framing and Data Pipeline | 10 | |
| 2 | Model Training and Experiment Tracking | 7 | |
| 3 | Model Packaging and Deployment | 5 | |
| 4 | CI/CD and Automation | 5 | |
| 5 | Monitoring and Documentation | 3 | |
| 6 | Presentation and Viva | 5 | |
| 7 | Report | 5 | |
| | Total | 40 | |
Then a line for "Signature of the Faculty-in-charge".

Write the full report now.
````

---

## Reusing this for the DevOps assignment

The same facts cover the separate DevOps report (*Automated Deployment of a Web
Application using CI/CD, Terraform, Azure and Jenkins*). Swap the structure
section for that assignment's deliverables — title, objective, tools used,
architecture diagram, Terraform snippets, Jenkinsfile, screenshots of
`terraform apply`, a successful Jenkins build and the running app, observations,
conclusion — and keep the FACTS section unchanged.

Two things worth putting in that report's *Observations* section, because they
were real friction rather than textbook steps:

- The Azure for Students subscription carries an **"Allowed resource deployment
  regions"** policy. `centralindia` is refused with a 403
  `RequestDisallowedByAzure` *after* the resource group has already been
  created. The allowed set had to be read from
  `az policy assignment list --query "[].parameters" -o json`.
- Jenkins rotates its apt signing key and its own documentation still points at
  `jenkins.io-2023.key`, which no longer matches the repository. The current key
  is `jenkins.io-2026.key`; `-2024` and `-2025` do not exist.
