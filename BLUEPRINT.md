# ML DeployMate — Project Blueprint

> Team **SavageAI** · Dr. Dacon's Machine Learning course · Fall 2026
> Status: **Phase 0: scaffolding** (repo created 2026-10-05). Day-to-day progress lives in [PROGRESS.md](PROGRESS.md).

This is the plan we agree on **before** writing code. If we change the plan, we change this file first, in a PR.

---

## 1. What we're building (one paragraph)

ML DeployMate watches a deployed machine-learning model and tells the engineer when something is going wrong, in plain English. It has three parts:

1. **A real model being monitored.** We train a classifier on a public dataset and serve it through FastAPI.
2. **An ML monitoring engine.** Statistical drift tests plus an Isolation Forest anomaly detector run on every batch of incoming predictions and metrics.
3. **An LLM assistant (OpenAI API).** It turns raw alerts and logs into explanations and suggested fixes, and it generates deployment configs (Dockerfile, CI workflow, env template) from a short form.

A Streamlit dashboard ties it together. You can pick a deployment, watch live metric charts, inject a drift scenario, see the alert fire, and read the LLM's explanation.

## 2. Why this shape (read this before arguing with it)

The concept doc describes an **LLM tool**. The course grades an **ML project**: the README template asks for a dataset, model architecture, training process, and a train/test metrics table. A pure OpenAI wrapper has no dataset and no metrics to report. So:

- **The ML is the core:** the monitored classifier and the drift/anomaly detector. Both are trained, evaluated, and reported with real numbers.
- **The LLM is the assistant layer on top.** It's what makes the project feel like "DeployMate," but the grade-bearing ML doesn't depend on it.
- **Offline mode:** if there's no API key or OpenAI is down, the LLM layer falls back to template-based text. **The live demo must never break during the presentation.**

### Deliberate deviations from the concept doc

| Concept doc says | We do | Why |
|---|---|---|
| React + D3.js dashboard | Streamlit + Plotly | Must be live on a free host (Streamlit Cloud / HF Spaces). A Python-only stack also means all 3 of us can work on the UI. |
| AWS ECS / GCP App Engine | Streamlit Community Cloud or Hugging Face Spaces (Docker) | Free, and the course explicitly names these. ECS costs money and adds risk. |
| TensorFlow | scikit-learn | Tabular data, so tree models win. TF adds ~500MB to the deploy for no gain. We can revisit if we add a neural model. |
| "Real-time streaming pipelines" | Simulated stream: replay the test set in batches with controllable injected drift | Same ML problem, demo-able, and we can label it, which lets us measure detector precision/recall. |
| Resume numbers ("40% fewer errors", "50% faster") | **Only claim numbers we actually measure** | Those figures are made up. The README reports our real evaluation results. |

## 3. Architecture

```
                ┌──────────────────────── Streamlit dashboard (app/) ────────────────────────┐
                │  Deployments · Live metrics charts · Drift injector · Alerts · Config gen    │
                └───────────────┬───────────────────────────────────────────────┬─────────────┘
                                │ imports directly (for a reliable live demo)    │ or HTTP
                                ▼                                                ▼
┌──────────────── deploymate/ (core Python package) ────────────────┐   ┌── api/ (FastAPI) ──┐
│ data/       load + split dataset                                   │◄──│ /predict           │
│ model/      train + evaluate the monitored classifier              │   │ /metrics/ingest    │
│ simulate/   replay batches, inject drift scenarios (labeled)       │   │ /alerts            │
│ monitor/    drift tests (PSI, KS, chi²) + IsolationForest          │   │ /assistant/explain │
│ llm/        OpenAI client, prompts, offline fallback               │   │ /assistant/config  │
│ db/         SQLAlchemy models: deployments, batches, metrics,      │   └────────────────────┘
│             alerts (SQLite locally)                                │
└────────────────────────────────────────────────────────────────────┘
```

**Rule:** all logic lives in `deploymate/`. `api/` and `app/` are thin layers that call it. That way tests cover the real logic, and the Streamlit demo still works if the API host sleeps.

## 4. The ML, concretely

### 4a. Monitored model
- **Dataset (proposed):** UCI Adult / Census Income (~48k rows, binary target `income >50K`). It's public, tabular, and has numeric + categorical features, which gives drift something interesting to work with. Backup: Kaggle "Default of Credit Card Clients."
- **Model:** baseline Logistic Regression, then Random Forest / Gradient Boosting. We pick the best with cross-validation.
- **Metrics:** accuracy, precision, recall, F1, ROC-AUC on train and held-out test. These fill the README table.

### 4b. Monitoring engine (the "DeployMate" ML)
- **Data drift:** PSI and Kolmogorov–Smirnov per numeric feature, chi-squared per categorical feature, comparing each live batch to the training reference.
- **Performance anomaly:** Isolation Forest over the batch-level metric vector (accuracy, mean confidence, positive rate, latency, error rate).
- **Adaptive thresholds:** rolling mean ± k·std per metric, learned from the "healthy" period. This covers the concept doc's "dynamically adjusting thresholds."
- **Evaluation (this is what makes it real ML):** we generate labeled scenarios: healthy, gradual covariate drift, sudden shift, label drift, and noise/corruption. We measure the detector's **precision, recall, F1, and detection delay** (batches until the alert fires).

### 4c. LLM layer (provider: pending, Claude recommended)
- **Provider-agnostic:** all LLM calls go through one small interface in `deploymate/llm/` (`explain_alert`, `generate_config`). The rest of the code never imports a vendor SDK directly, so switching OpenAI ↔ Claude is a one-file change.
- **Recommendation: Claude (Anthropic API), on Oba's account.** Oba has already shipped Claude API code (Morgan IT Portal), so there's less to learn. Structured JSON outputs are supported. Cost per alert explanation is about 1–2¢ on Claude Opus 5.5, or well under 1¢ on Claude Haiku 4.5 if we want cheaper. Either way the whole project should cost a few dollars at most.
- **Tradeoff:** the concept doc says "OpenAI API." If we use Claude, we note the deviation in the README. The design and graded ML are the same either way.
- `explain_alert(alert, recent_metrics)` → what happened, likely cause, suggested action.
- `generate_config(form)` → Dockerfile + GitHub Actions workflow + `.env.example` for the user's model.
- Structured outputs (JSON schema) so the UI can render the results cleanly.
- The key comes from env / Streamlit secrets only. **It is never committed and never shown in the UI.**

## 5. Repo layout

```
ml-deploymate/
├── README.md            # course template, filled in as we go
├── BLUEPRINT.md         # this file
├── requirements.txt
├── .env.example
├── deploymate/          # core package (data, model, simulate, monitor, llm, db)
├── api/                 # FastAPI app
├── app/                 # Streamlit dashboard
├── notebooks/           # EDA + experiments (exploration only, not imported by code)
├── scripts/             # download_data.py, train.py, evaluate.py
├── tests/               # pytest
├── models/              # saved model artifacts (gitignored except small final ones)
├── data/                # raw/ and processed/ are gitignored
├── assets/              # demo.gif, confusion matrix, ROC curves, drift plots
└── .github/workflows/   # CI: lint + pytest on every PR
```

## 6. Team & ownership

> Team of 3 confirmed: Marissa Savage (`sadiebird5`), Obaloluwa Wojuade (`obwoj1`), and Aubrenesia Tomilson (`aubrenesia`), invited 2026-10-06.

| Area | Owner | Notes |
|---|---|---|
| Data pipeline, EDA, monitored model training + evaluation | Marissa Savage (`sadiebird5`) | `deploymate/data`, `deploymate/model`, `scripts/`, `notebooks/` |
| Monitoring engine, drift simulation, detector evaluation | Obaloluwa Wojuade (`obwoj1`) | `deploymate/monitor`, `deploymate/simulate` |
| FastAPI + DB, LLM layer, Streamlit dashboard, deployment | Aubrenesia Tomilson (`aubrenesia`) | `api/`, `deploymate/llm`, `deploymate/db`, `app/` (proposed, confirm with her) |
| README, demo.gif, CI | Shared | Everyone commits |

Grading looks at **each person's commit history**, so ownership is per area, but everyone touches everything at least a little (reviews, tests, docs).

## 7. Phases

Deadlines haven't been posted yet. Fill in dates when Dr. Dacon posts the rubric.

| # | Phase | Deliverable | Done when |
|---|---|---|---|
| 0 | Scaffold | Private repo, README skeleton, this blueprint, Dr. Dacon added as collaborator | Both teammates have cloned and pushed one commit |
| 1 | Data + baseline | `download_data.py`, EDA notebook, baseline model, first metrics | Test-set F1 recorded in README |
| 2 | Monitored model | Tuned model, saved artifact, FastAPI `/predict` | `pytest` passes and the API serves predictions |
| 3 | Drift simulation + monitoring | Scenario generator, drift tests, Isolation Forest, detector eval | Detector precision/recall table in README |
| 4 | LLM assistant | Alert explanations + config generator + offline fallback | Works both with a key and with no key |
| 5 | Dashboard | Streamlit pages: overview, live metrics, drift injector, alerts, config gen | The full demo flow runs locally end to end |
| 6 | Deploy + CI | Live URL, GitHub Actions running pytest | URL works in an incognito window |
| 7 | Polish | `demo.gif`, final README, assets, presentation | Rubric checklist all ✅ |

## 8. Git workflow (team rules)

1. `main` is always deployable. **No direct pushes to `main`.** Everything goes through a PR.
2. Branch names: `feature/<thing>`, `fix/<thing>`, `docs/<thing>`, e.g. `feature/drift-psi`.
3. Commit small and often, with descriptive messages ("Add PSI drift test for numeric features", not "update").
4. Every PR needs **one review from a teammate** before merging. Reviewing counts as collaboration.
5. `git pull origin main` before starting work and before opening a PR.
6. Never commit `.env`, API keys, raw data, or large model files.

## 9. Submission checklist (from Dr. Dacon's post)

- [x] Team of 3 (third member `aubrenesia` invited 2026-10-06)
- [x] Dr. Dacon (`daconjam`) invited as repo collaborator (2026-10-05)
- [ ] Every member has meaningful commits
- [ ] README follows the course template
- [ ] `demo.gif` embedded in README
- [ ] Live demo URL, working during the presentation
- [ ] Branches + PRs used throughout
- [ ] Rubric items (to be posted)

## 10. Open questions

1. ~~Third member~~ Aubrenesia Tomilson (`aubrenesia`), invited 2026-10-06.
2. ~~Dr. Dacon's GitHub username~~ `daconjam` (invited)
3. ~~Marissa's GitHub username?~~ `sadiebird5` (verified: Marissa Savage)
4. LLM API key: Oba's account. Provider decision pending (Claude recommended, see §4c).
5. Deadlines and rubric: TBD from Dr. Dacon.

## 11. Status log

| Date | Update |
|---|---|
| 2026-10-05 | Repo created (private) at github.com/obwoj1/ml-deploymate; blueprint, README, folder skeleton, CI + PR template pushed. Marissa (sadiebird5) invited with write access. Still need: Dr. Dacon's GitHub username, third-member answer. Note: branch protection is not available on free private repos, so the "no direct push to main" rule is enforced by the team, not by GitHub. |
