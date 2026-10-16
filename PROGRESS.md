# ML DeployMate: Progress Log

> Running log of what's been done, what we're working on, and what's next.
> **Update this at the end of every work session.** Newest entries go at the top of the log.
> For the full plan, see [BLUEPRINT.md](BLUEPRINT.md).

---

## 📍 Where we are right now

**Sprint 1: Foundations.** Setup is done; drift tests are built (PR #2); roles were updated 2026-10-08.

**Goal for the project:** build a tool that watches a deployed ML model, detects when it starts going wrong (data drift, performance drops), and uses an LLM to explain each alert in plain English and suggest fixes. The graded ML pieces are:
1. A classifier trained on a public dataset (the model being monitored).
2. A drift / anomaly detector that we evaluate on labeled, injected-drift scenarios.

The LLM is the assistant layer on top.

**Goal for this sprint:** Oba ships the mock `service.py` + dataset loader + baseline model; Aubrenesia ships the app shell; Marissa designs her pages. See [docs/TASKS.md](docs/TASKS.md).

---

## ✅ Done

| Date | What | Who |
|---|---|---|
| 2026-10-05 | Created private GitHub repo `obwoj1/ml-deploymate` | Oba |
| 2026-10-05 | Wrote BLUEPRINT.md: scope, architecture, ML plan, phases, team rules | Oba |
| 2026-10-05 | README.md filled in from Dr. Dacon's template (results tables blank until we have numbers) | Oba |
| 2026-10-05 | Folder skeleton: `deploymate/` core package, `api/`, `app/`, `scripts/`, `notebooks/`, `tests/`, `models/`, `data/`, `assets/` | Oba |
| 2026-10-05 | `.gitignore`, `.env.example`, `requirements.txt` | Oba |
| 2026-10-05 | GitHub Actions CI (pytest on every PR) + PR template | Oba |
| 2026-10-05 | Invited Marissa Savage (`sadiebird5`) as collaborator | Oba |
| 2026-10-05 | Invited Dr. Jamell Dacon (`daconjam`) as collaborator | Oba |
| 2026-10-06 | Third member confirmed: invited `aubrenesia` as collaborator; added to README + BLUEPRINT | Oba |

## 🔄 In progress

- [x] Marissa accepted the repo invite (confirmed 2026-10-07)
- [x] Marissa's first commit: created `scripts/download_data.py` (empty placeholder) via PR #1, merged 2026-10-08
- [ ] Dr. Dacon accepts the repo invite (still pending as of 2026-10-07)
- [x] Aubrenesia accepted the repo invite (confirmed 2026-10-07)
- [ ] Aubrenesia pushes her first commit
- [x] LLM provider decided: OpenAI `gpt-4o-mini` (2026-10-13)

## ⏭️ Up next: Sprint 1 (full list in [docs/TASKS.md](docs/TASKS.md))

**Oba (ML + backend)**
- [x] Drift tests: PSI/KS + chi²/Cramér's V, 7 tests passing
- [x] Mock `deploymate/service.py` **first**, so the frontend can start building
- [x] `scripts/download_data.py` + `deploymate/data/load.py` (UCI Adult), 29 tests passing
- [x] EDA notebook + baseline model (test accuracy 0.853, F1 0.657, ROC-AUC 0.905)
- [x] Drift simulator (`deploymate/simulate/`), 52 tests passing

**Aubrenesia (frontend)**
- [ ] Streamlit app shell (`app/Home.py`), theme, 5 stub pages, wireframes

**Marissa (frontend)**
- [ ] Wireframes for Live Monitor + Model Performance; Streamlit/Plotly basics; review Oba's PRs

---

## 🧭 Decisions

| Date | Decision | Why |
|---|---|---|
| 2026-10-05 | ML core = classifier + drift/anomaly detector; LLM sits on top | Grading needs a dataset + real metrics; a pure LLM wrapper has neither |
| 2026-10-05 | Streamlit + Plotly instead of React + D3 | Free live hosting; everyone can work on it in Python |
| 2026-10-05 | scikit-learn instead of TensorFlow | Tabular data; TF bloats the deploy |
| 2026-10-05 | LLM has an offline fallback mode | The live demo must never break during the presentation |
| 2026-10-05 | Report only measured numbers, not the made-up resume figures | Honesty, and graders will ask |
| 2026-10-08 | **Team split:** Oba = all ML + backend; Marissa + Aubrenesia = frontend dashboard | Oba wants to own the core ML; frontend gives the others a clear, separate area with their own commits |
| 2026-10-08 | Dataset confirmed: UCI Adult. Keep rows with `?` as an "Unknown" category instead of dropping them | Live data has gaps too; dropping would lose ~7% of rows |
| 2026-10-08 | Report F1 + ROC-AUC, not just accuracy | 76% of rows are `<=50K`, so always guessing that already scores 76% |
| 2026-10-08 | Drop `fnlwgt` and `education` columns | `fnlwgt` is a sampling weight, not a person attribute; `education` duplicates `education_num` |
| 2026-10-08 | Teammates **pick** tasks from a pick-up list (docs/TASKS.md) and claim them with GitHub Issues instead of fixed assignments; Oba keeps the ML + backend bulk | Lets them choose what they're comfortable with; issues + PRs make everyone's contributions visible |
| 2026-10-12 | Monitored model = Gradient Boosting; pick by CV ROC-AUC | Best on every CV metric; ROC-AUC isn't fooled by class imbalance |
| 2026-10-12 | Don't commit `model.joblib`; regenerate it with `scripts/train.py` | Pickled sklearn models break across library versions; training takes ~15s |
| 2026-10-12 | Pool categories <1% of the reference into "Other" for drift tests | Rare categories made healthy batches look drifted by chance (native_country false alarms) |
| 2026-10-12 | Anomaly detector uses only label-free metrics | In production, true labels arrive late; the monitor must work without them |
| 2026-10-12 | Isolation Forest alarm line = strangest of 200 healthy calibration batches (not `contamination`) | `contamination` guarantees ~2% false alarms on healthy data |
| 2026-10-13 | **LLM provider: OpenAI (`gpt-4o-mini`)**, key in Oba's local `.env` (verified working) | Oba has an OpenAI key; matches the concept doc. Interface stays provider-agnostic |

## ❓ Open questions

1. ~~Third team member~~ Aubrenesia Tomilson (`aubrenesia`).
2. ~~LLM provider~~ OpenAI, key set up in Oba's `.env`.
3. **Deadlines + rubric:** not posted yet.

---

## 📓 Session log

### 2026-10-15
- **Monitor evaluation done** (`deploymate/monitor/evaluate.py`, `scripts/evaluate_monitor.py`): 10 runs per scenario, scored at batch level. Overall combined: precision 0.992, recall 0.932, F1 0.961, false-alarm rate 0.6%, mean detection delay 1.5 batches, 50/50 broken runs detected. Results in README + `assets/monitor_results.json`.
- **Evaluation bug found + fixed:** a shared seed made every scenario replay identical healthy batches, so one false alarm was counted up to 6 times (showed 1.4%). Each scenario now gets its own seed; calibration seed moved to 100000 so it never overlaps.
- **Honest finding (ablation):** drift tests alone score F1 0.965 with 0% false alarms; the anomaly detector alone has recall 0.287 and adds 0.6% false alarms for +0.1% recall. On THESE scenarios it's mostly redundant (except it independently catches 100% of missing_values). It's a label-free safety net for problems that don't show up as column drift. Open decision for Oba: keep it in the alert, or show it as an "info" signal only.

### 2026-10-13
- OpenAI key added to Oba's `.env` and verified; provider = OpenAI `gpt-4o-mini`.
- **Created GitHub issues #9–#19** for Marissa and Bree, each with step-by-step instructions and tested Streamlit starter code (every snippet run through Streamlit's AppTest first): #9 setup, #10 wireframes, #11 theme, #12 app shell (do first), #13 Overview, #14 Live Monitor, #15 Model Performance, #16 Alerts, #17 Config Generator, #18 presentation outline, #19 related-work research. Unassigned; they pick.

### 2026-10-12
- **Monitor complete (both halves):** `metrics.py` (label-free: positive rate, mean confidence, low-confidence share, unknown rate; accuracy/F1 only when labels exist), `anomaly.py` (per-metric z-score > 4 vs healthy + Isolation Forest), `pipeline.py` (`Monitor.check()` → metrics, drift, anomaly, alert, severity, same shape as service.py). 58 tests.
- **Isolation Forest tuning:** sklearn's `contamination` setting flags a fixed share of healthy batches by design (~2% false alarms, 6–7 per 3-seed sweep). Switched to calibrating on 200 healthy batches and setting the alarm line at the strangest healthy batch → 0 false alarms, still catches problems on its own.
- **Results (full monitor, 1 run):** 0 false alarms in every scenario; caught 20/20 on all abrupt scenarios, 21/28 gradual. Anomaly detector alone catches 20/20 `missing_values` (via unknown_rate) and some of the others.
- **False alarms fixed:** categories under 1% of the reference are pooled into "Other" before the chi² test. Scenario summary now: 0 false alarms in every scenario (was 4–6), still 20/20 caught on abrupt scenarios, 21/28 on gradual.
- `report.json` no longer stores training times, so it's identical between runs.
- **Drift simulator built:** `simulate()` replays the test set as 40 batches × 400 rows; 6 scenarios (healthy, gradual_shift, sudden_shift, category_shift, missing_values, noise), each healthy for the first part of the run, then broken, with ground-truth labels. `scripts/simulate.py` shows it live.
- **First end-to-end run (drift tests only, no tuning yet):** every abrupt scenario caught 20/20; gradual_shift 21/28 (misses the earliest tiny shifts, as expected).
  - **Finding 1, false alarms:** 4–6 per run, almost all on `native_country`. 41 categories, most very rare, so a 400-row batch looks "different" by chance and Cramér's V gets inflated. Fix next: group rare categories into "Other" before the chi² test.
  - **Finding 2, accuracy barely drops** (0.871 → ~0.865) under these shifts. The model is robust to these feature shifts. Presentation point: drift ≠ broken model, and in production the labels arrive late, so drift monitoring is the early warning. Consider adding a label/concept-shift scenario that really hurts accuracy, for a more dramatic demo.
- **Final model trained:** `scripts/train.py` compares 3 models with 5-fold CV. Gradient Boosting won (CV ROC-AUC 0.928 vs RF 0.919 vs LogReg 0.907). Test: accuracy 0.873, F1 0.707, ROC-AUC 0.928. Top features: marital_status, capital_gain, age, education_num. Saves `models/model.joblib` (gitignored; regenerate in ~15s) + `models/report.json` (committed; the dashboard's Model Performance data).
- PR #5 merged to main. Teammates haven't started yet, so Oba is pushing ahead on the backend.
- Built `deploymate/model/pipeline.py` (Pipeline: StandardScaler + OneHotEncoder(handle_unknown="ignore") + classifier; `score()` returns accuracy/precision/recall/F1/ROC-AUC) + tests.
- `notebooks/01_eda.ipynb`: target balance, numeric/categorical exploration, missing values, baseline LogReg, confusion matrix, top coefficients. Committed without outputs.
- **Baseline (test):** accuracy 0.853, precision 0.728, recall 0.599, F1 0.657, ROC-AUC 0.905 vs 0.764 for always guessing <=50K.

### 2026-10-08
- Data pipeline: `download_data.py` (uses certifi; python.org macOS Python failed SSL verification without it) and `load_dataset()` (fixes all quirks, drops `fnlwgt` + `education`, missing → "Unknown", y = 1 for >50K). Tests use fake quirk-matching files so CI passes without the real data.
- Built mock `deploymate/service.py` (the frontend contract) + `tests/test_service.py`; 16 tests passing.
- **Branch cleanup:** Oba wants fewer branches. All of Oba's work now lives on ONE branch, `backend` (drift tests + mock service + dataset docs). Old `feature/drift-tests`, `feature/mock-service`, `docs/task-plan` deleted.
- Oba: set up local venv; built `deploymate/monitor/drift.py` (PSI + KS for numeric, chi² + Cramér's V for categorical; drift decision uses effect size, not p-values, because big batches make p-values flag tiny shifts) + `tests/test_drift.py`. 
- Wrote `docs/DATASET.md` (what the data is, columns, quirks, why we chose it, its 3 jobs in the project, bias note, 30-sec presentation summary); README Dataset section filled in.
- Dataset downloaded by hand into `data/raw/` on Oba's Mac (32,561 train rows, 16,281 test rows; gitignored, so not on GitHub). `download_data.py` still needed for teammates + the live server.
- **Roles changed:** Oba takes the data, model, monitoring, LLM and API work; Marissa + Aubrenesia take the Streamlit dashboard. BLUEPRINT §6–7, README, and docs/TASKS.md rewritten to match.
- Marissa opened PR #1 (`feature/download-data`) creating an empty `scripts/download_data.py`; Oba merged it. First PR-based contribution ✅
- Project folder moved on Oba's Mac from `~/Developer/ml-deploymate` to `~/Desktop/ml-deploymate`.

### 2026-10-06: Third member
- Third teammate's GitHub is `aubrenesia`; invited with write access. Proposed role: API + DB + LLM layer + dashboard + deployment.
- 2026-10-07: name confirmed as Aubrenesia Tomilson; added to README.
- **Next:** run a team kickoff where everyone confirms their area and clones the repo.

### 2026-10-05: Kickoff / scaffolding
- Read Dr. Dacon's project post + the ML DeployMate concept doc.
- Realized the concept is an LLM tool but the course grades an ML project, so we redesigned around a monitored classifier + drift detector, with the LLM as the explanation layer.
- Set up the repo, blueprint, README, skeleton, and CI; invited Marissa + Dr. Dacon.
- **Next session:** Oba starts drift tests on `feature/drift-tests`. (Original plan had Marissa on the data pipeline; superseded 2026-10-08.)
