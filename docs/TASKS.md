# Task Plan: who does what, in what order

> Companion to [BLUEPRINT.md](../BLUEPRINT.md) (the *what*) and [PROGRESS.md](../PROGRESS.md) (the *status*).
> This file is the *order of work*. Tick boxes here as tasks merge.
> Sprint dates get filled in once Dr. Dacon posts deadlines.

---

## The big picture: what the final demo shows

```
 ① Model serves predictions  →  ② Batches of live data flow in  →  ③ Monitor checks each batch
        (Marissa)                      (Oba: simulator)                (Oba: drift + anomaly)
                                                                              │
 ⑥ Live URL + demo.gif  ←  ⑤ Dashboard shows it all  ←  ④ Alert saved + LLM explains it
     (everyone)                 (Aubrenesia lead)            (Aubrenesia: DB + LLM)
```

Every task below builds one box of that chain. **Presentation story:** "Here's a healthy model → we inject drift → watch DeployMate catch it and explain what happened → here are our measured results."

### The contracts (agree on these early so nobody waits on anybody)
| Contract | Owner | Used by |
|---|---|---|
| `deploymate.data.load_dataset()` → `(X_train, X_test, y_train, y_test)` as DataFrames | Marissa | Oba (simulator), Aubrenesia (API) |
| `models/model.joblib`: a fitted sklearn `Pipeline` (preprocessing + classifier) with `.predict` / `.predict_proba` | Marissa | Oba (batch metrics), Aubrenesia (`/predict`) |
| `deploymate.monitor.drift.detect_drift(ref_df, cur_df)` → list of `FeatureDrift` | Oba ✅ | Aubrenesia (alerts, dashboard) |
| `deploymate.monitor.run_batch(batch)` → `BatchReport` (metrics + drift + anomaly flag) | Oba | Aubrenesia (API, DB, dashboard) |
| `deploymate.llm.explain_alert(report)` → `{summary, likely_cause, suggested_action}` | Aubrenesia | Dashboard |

**Until a contract is ready, use fake data with the same shape.** For example, Oba's simulator can run on synthetic DataFrames before Marissa's loader exists.

---

## Sprint 1: Foundations (all three work in parallel)

**Marissa: Data + baseline model**
- [ ] `scripts/download_data.py`: download UCI Adult into `data/raw/` (PR #1 created the file; fill it in)
  - Zip: https://archive.ics.uci.edu/static/public/2/adult.zip (page: https://archive.ics.uci.edu/dataset/2/adult)
  - Files: `adult.data` (train), `adult.test` (test), `adult.names` (column docs)
  - Gotchas: no header row; `adult.test` has a junk first line and labels like `>50K.`; missing values are `?`; values have leading spaces
  - `data/raw/` is gitignored, so the **script** is how teammates and the live server get the data
- [ ] `deploymate/data/load.py`: `load_dataset()`: clean (strip spaces, `?` → missing), fixed train/test split with `random_state=42`
- [ ] `notebooks/01_eda.ipynb`: target balance, feature distributions, missing values, correlations
- [ ] Baseline Logistic Regression and its first test metrics → README table

**Oba: Monitoring core**
- [x] `deploymate/monitor/drift.py`: PSI / KS / chi² / Cramér's V + tests (PR #2)
- [ ] `deploymate/simulate/scenarios.py`: split a dataset into batches and inject labeled problems: `healthy`, `gradual_shift`, `sudden_shift`, `category_shift`, `noise`, `missing_values`
- [ ] Tests for the simulator

**Aubrenesia: Backend skeleton**
- [ ] `deploymate/db/models.py`: SQLAlchemy tables: `deployments`, `batches`, `metrics`, `alerts` (SQLite)
- [ ] `api/main.py`: FastAPI app with `GET /health` + a test
- [ ] `deploymate/llm/client.py`: the provider interface + **offline fallback** (template text when there's no key)

**Sprint 1 done when:** the dataset loads, the baseline is scored, the simulator makes labeled batches, and the API answers `/health`.

---

## Sprint 2: Real model + monitor wired together

**Marissa: Final model**
- [ ] `scripts/train.py`: compare LogReg vs Random Forest vs Gradient Boosting with cross-validation
- [ ] Save the best one as `models/model.joblib` (a Pipeline with preprocessing included)
- [ ] `scripts/evaluate.py`: train/test accuracy, F1, ROC-AUC; confusion matrix + ROC curve PNGs in `assets/`

**Oba: Anomaly detection + pipeline**
- [ ] `deploymate/monitor/metrics.py`: per-batch metrics from model predictions (accuracy, positive rate, mean confidence, missing-value rate)
- [ ] `deploymate/monitor/anomaly.py`: Isolation Forest trained on healthy batch metrics + adaptive thresholds (rolling mean ± k·std)
- [ ] `deploymate/monitor/pipeline.py`: `run_batch()` combining drift + metrics + anomaly into one `BatchReport`

**Aubrenesia: API endpoints**
- [ ] `POST /predict`: loads `model.joblib` and returns predictions
- [ ] `POST /batches`: runs `run_batch()` and stores the report + any alerts in the DB
- [ ] `GET /alerts`, `GET /deployments/{id}/metrics`

**Sprint 2 done when:** you can POST a drifted batch and get an alert stored in the DB.

---

## Sprint 3: Evaluation + LLM

**Marissa: Model results**
- [ ] Hyperparameter tuning on the winning model; final numbers in the README
- [ ] Short "why this model" writeup for the README's *Model Architecture & Training* section

**Oba: Detector evaluation (the headline result)**
- [ ] `scripts/evaluate_monitor.py`: run every scenario many times; compute precision, recall, F1, and mean detection delay
- [ ] Results table + plots in the README and `assets/`
- [ ] Tune the thresholds (PSI, Cramér's V, Isolation Forest contamination) on a validation set of scenarios, **not** the test set

**Aubrenesia: LLM assistant**
- [ ] `explain_alert(report)`: structured JSON output, tested with and without an API key
- [ ] `generate_config(form)`: Dockerfile + GitHub Actions + `.env.example`
- [ ] `POST /assistant/explain`, `POST /assistant/config`

**Sprint 3 done when:** both README results tables are filled with real numbers and every alert has an explanation.

---

## Sprint 4: Dashboard (everyone builds one page)

Aubrenesia sets up the Streamlit app shell (`app/Home.py`, navigation, shared styling); then each person builds a page:
- [ ] **Aubrenesia**: Overview + Alerts page (alert list, LLM explanation panel) + Config Generator page
- [ ] **Oba**: Live Monitor page (pick a scenario → stream batches → PSI/metric charts update → alert fires)
- [ ] **Marissa**: Model page (metrics, confusion matrix, ROC curve, feature importance)

**Sprint 4 done when:** the full demo story runs locally, start to finish.

---

## Sprint 5: Ship + present (everyone)

- [ ] Deploy to Streamlit Community Cloud or HF Spaces (lead: Aubrenesia); API key in host secrets; test the URL in an incognito window
- [ ] `demo.gif` recorded and embedded in the README (lead: Marissa)
- [ ] README final pass: every template section filled in (lead: Oba)
- [ ] Presentation: each person presents their own part
- [ ] Rehearse the live demo on the deployed URL, with offline mode as a backup

---

## Rules that keep the grade safe
1. **Each person commits their own tasks.** Helping = reviewing PRs, pairing, unblocking. Don't write someone else's task under your account.
2. **Every PR gets a review from a teammate.** Reviews are visible collaboration.
3. **Merge small PRs often.** One task ≈ one PR.
