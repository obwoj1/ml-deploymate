# Task Plan: who does what, in what order

> Companion to [BLUEPRINT.md](../BLUEPRINT.md) (the *what*) and [PROGRESS.md](../PROGRESS.md) (the *status*).
> This file is the *order of work*. Tick boxes here as tasks merge.
> Sprint dates get filled in once Dr. Dacon posts deadlines.

---

## Team split (updated 2026-10-08)

| Who | Owns | Folders |
|---|---|---|
| **Oba** (`obwoj1`) | **ML + backend**: data, model, drift simulation, monitoring engine, evaluation, LLM layer, API | `deploymate/`, `api/`, `scripts/`, `notebooks/`, `tests/` (backend) |
| **Marissa** (`sadiebird5`) | **Frontend**: Live Monitor page, Model Performance page, `demo.gif` | `app/pages/` (her pages), `tests/app/` |
| **Aubrenesia** (`aubrenesia`) | **Frontend**: app shell + theme, Overview, Alerts + AI panel, Config Generator, deploying the live site | `app/` (shell + her pages), `.streamlit/`, `tests/app/` |

Oba explains the ML/backend to the team once it's built, so everyone can present their part.

---

## The big picture: what the final demo shows

```
 ┌──────────── BACKEND (Oba) ────────────┐        ┌──────────── FRONTEND (Marissa + Aubrenesia) ────────────┐
 │ data → model → simulated live batches │        │ Overview · Live Monitor · Alerts + AI explanation ·      │
 │ → drift + anomaly detection → alerts  │ ─────► │ Model Performance · Config Generator                     │
 │ → LLM explanation                     │ service│ (Streamlit, deployed to a live URL)                      │
 └───────────────────────────────────────┘  .py   └──────────────────────────────────────────────────────────┘
```

**Presentation story:** "Here's a healthy model → we inject drift → watch DeployMate catch it on the dashboard and explain what happened → here are our measured results."

### The one contract between backend and frontend: `deploymate/service.py`

The frontend **only** imports from `deploymate.service`. Nothing else. Oba ships a **mock version first** (returns realistic fake data), so the frontend team can build every page from day one. Later Oba swaps the internals for the real thing, and the pages don't change.

| Function | Returns | Used on page |
|---|---|---|
| `get_overview()` | deployment name, model, status, # batches, # active alerts | Overview |
| `list_scenarios()` | `["healthy", "gradual_shift", "sudden_shift", ...]` | Live Monitor |
| `run_scenario(name, n_batches)` | iterator of batch reports (metrics, per-feature drift, alert flag) | Live Monitor |
| `list_alerts()` | alerts with time, severity, features, metrics | Alerts |
| `explain_alert(alert_id)` | `{summary, likely_cause, suggested_action}` | Alerts |
| `get_model_report()` | train/test metrics, confusion matrix, ROC points, feature importance | Model Performance |
| `generate_config(form)` | `{dockerfile, github_actions, env_example}` | Config Generator |

---

## Sprint 1: Foundations (in parallel)

**Oba: Data, baseline model, mock service**
- [x] `deploymate/monitor/drift.py`: PSI / KS / chi² / Cramér's V + tests (PR #2)
- [x] **`deploymate/service.py` (mock version)**: do this FIRST, it unblocks the frontend
- [x] `scripts/download_data.py`: download UCI Adult into `data/raw/`
  - Zip: https://archive.ics.uci.edu/static/public/2/adult.zip (page: https://archive.ics.uci.edu/dataset/2/adult)
  - Gotchas: no header row; `adult.test` has a junk first line and labels like `>50K.`; missing values are `?`; leading spaces
- [x] `deploymate/data/load.py`: `load_dataset()`: clean the data, uses the dataset's own train/test split
- [x] `notebooks/01_eda.ipynb` + baseline Logistic Regression → first metrics in README (test acc 0.853, F1 0.657, ROC-AUC 0.905)
- [x] `deploymate/simulate/scenarios.py` + tests: labeled drift scenarios (+ `scripts/simulate.py` to watch them)

**Aubrenesia: App shell**
- [ ] `app/Home.py`: Streamlit entry point, sidebar navigation, page layout
- [ ] `.streamlit/config.toml`: theme (colors, font)
- [ ] Empty stub pages for all 5 pages so the navigation works
- [ ] Sketch wireframes for her pages (paper or Figma, screenshots in `assets/wireframes/`)

**Marissa: Page designs**
- [ ] Sketch wireframes for Live Monitor + Model Performance (`assets/wireframes/`)
- [ ] Learn Streamlit + Plotly basics (charts, `st.metric`, `st.columns`, `st.session_state`)
- [ ] Review Oba's PRs (counts as collaboration)

**Sprint 1 done when:** the app shell runs with 5 navigable pages, the mock service exists, and the dataset loads.

---

## Sprint 2: Build pages on mock data / real model

**Oba: Real model + monitoring engine**
- [x] `scripts/train.py`: LogReg vs Random Forest vs Gradient Boosting, cross-validated; best saved to `models/model.joblib` (Gradient Boosting won)
- [x] Evaluation: accuracy, F1, ROC-AUC, confusion matrix, ROC data, permutation importance → `models/report.json` (built into train.py)
- [x] `deploymate/monitor/metrics.py`, `anomaly.py` (Isolation Forest + adaptive thresholds), `pipeline.py` (`Monitor.check()`)

**Aubrenesia: Overview + Alerts pages** (using the mock service)
- [ ] Overview: status cards (`st.metric`), recent-alerts table
- [ ] Alerts: filterable alert list, detail view, "Explain with AI" button → explanation panel

**Marissa: Live Monitor + Model Performance pages** (using the mock service)
- [ ] Live Monitor: scenario picker → "Start" streams batches → Plotly line charts update live (accuracy, PSI per feature) → red banner when an alert fires
- [ ] Model Performance: metric cards, confusion matrix heatmap, ROC curve, feature importance bar chart

**Sprint 2 done when:** every page works end to end on mock data.

---

## Sprint 3: Real backend + LLM, swap mock → real

**Oba**
- [ ] `scripts/evaluate_monitor.py`: detector precision / recall / F1 / detection delay → README + `assets/`
- [ ] `deploymate/llm/`: `explain_alert`, `generate_config`, with offline fallback
- [ ] `api/`: FastAPI endpoints (`/predict`, `/batches`, `/alerts`, `/assistant/*`)
- [ ] Swap `service.py` internals from mock → real

**Aubrenesia**
- [ ] Config Generator page: form → generated files shown in code blocks + download buttons
- [ ] Test her pages against the real service; fix anything that breaks
- [ ] `tests/app/`: Streamlit `AppTest` smoke tests (each page loads without errors)

**Marissa**
- [ ] Test her pages against the real service; polish charts (labels, colors, empty/loading states)
- [ ] `tests/app/`: `AppTest` smoke tests for her pages

**Sprint 3 done when:** the dashboard runs on real data and both README results tables are filled in.

---

## Sprint 4: Ship + present (everyone)

- [ ] **Aubrenesia**: deploy to Streamlit Community Cloud / HF Spaces; API key in host secrets; works in an incognito window
- [ ] **Marissa**: record `demo.gif` of the full demo story, embed it in the README
- [ ] **Oba**: final README (dataset, model, results); walk the team through the ML/backend
- [ ] **Everyone**: presentation (each person presents their own part); rehearse on the live URL

---

## 🙋 Pick-up list: choose what you want (added 2026-10-08)

Instead of assigning everything, **teammates pick tasks they're interested in**. Oba carries the ML + backend; these are smaller, self-contained tasks anyone can take.

**How to claim one:**
1. Open a **GitHub Issue** (repo → Issues → New issue) with the task name, and assign it to yourself so nobody doubles up.
2. Work on **your own branch** (`frontend-aubrenesia` / `frontend-marissa`; one branch each, reused).
3. Push and open a **PR** that says `Closes #<issue number>`.
4. A teammate reviews and merges. **We all review each other's PRs.**

| Task | Difficulty | Needs |
|---|---|---|
| Sketch wireframes for the 5 dashboard pages (paper or Figma → `assets/wireframes/`) | 🟢 Easy | Nothing |
| Test `docs/LOCAL_SETUP.md` on your laptop; fix any confusing step | 🟢 Easy | Nothing |
| Research 2–3 real tools like ours (Evidently AI, WhyLabs, Arize) for a "related work" slide | 🟢 Easy | Nothing |
| Presentation slide outline (intro, problem, dataset, demo, results, future work) | 🟢 Easy | Nothing |
| Pick the dashboard theme: colors + font in `.streamlit/config.toml` | 🟢 Easy | Nothing |
| Streamlit app shell: `app/Home.py` + sidebar navigation + 5 empty pages | 🟡 Medium | Streamlit basics |
| Overview page: status cards with `st.metric` from `service.get_overview()` | 🟡 Medium | App shell |
| Alerts page: table from `service.list_alerts()` + "Explain" button | 🟡 Medium | App shell |
| Model Performance page: Plotly confusion matrix + ROC curve from `service.get_model_report()` | 🟡 Medium | App shell |
| Live Monitor page: scenario dropdown + live-updating line chart from `service.run_scenario()` | 🟠 Harder | App shell |
| Config Generator page: form → `service.generate_config()` → code blocks + download buttons | 🟡 Medium | App shell |
| Record `demo.gif` + write the demo script | 🟢 Easy | Working dashboard (later) |
| README "Responsible use / bias" section (see docs/DATASET.md §6) | 🟢 Easy | Nothing |
| Review any open PR (read it, leave a comment, approve) | 🟢 Easy | Nothing |

Stuck? Comment on your issue or ping the group chat. Asking early is faster than guessing.

## Rules that keep the grade safe
1. **Each person commits their own tasks.** Oba doesn't build the pages; Marissa and Aubrenesia don't build the ML.
2. **Every PR gets a review from a teammate.** Frontend folks review backend PRs too (that's how they learn it for the presentation).
3. **Issue → your branch → PR → teammate review → merge.** Claim work with an issue; one branch per person, PR per finished chunk. Branches: `backend` (Oba), `frontend-aubrenesia`, `frontend-marissa`. Open a PR into `main` whenever a chunk works; after merging, keep using the same branch.
4. Frontend imports **only** `deploymate.service`, so backend changes never break the pages.
