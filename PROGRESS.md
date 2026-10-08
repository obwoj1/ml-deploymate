# ML DeployMate: Progress Log

> Running log of what's been done, what we're working on, and what's next.
> **Update this at the end of every work session.** Newest entries go at the top of the log.
> For the full plan, see [BLUEPRINT.md](BLUEPRINT.md).

---

## 📍 Where we are right now

**Phase 0: Scaffold** (almost done)

**Goal for the project:** build a tool that watches a deployed ML model, detects when it starts going wrong (data drift, performance drops), and uses an LLM to explain each alert in plain English and suggest fixes. The graded ML pieces are:
1. A classifier trained on a public dataset (the model being monitored).
2. A drift / anomaly detector that we evaluate on labeled, injected-drift scenarios.

The LLM is the assistant layer on top.

**Goal for this phase:** get the repo, plan, and team access set up so all three of us can start committing on feature branches.

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
- [ ] Aubrenesia confirms she's OK owning API/LLM/dashboard and pushes her first commit
- [ ] Decide LLM provider: OpenAI vs Claude (see Decisions below)

## ⏭️ Up next (Phase 1: Data + baseline, owner: Marissa)

- [ ] `scripts/download_data.py`: fetch the UCI Adult dataset into `data/raw/`
- [ ] `notebooks/01_eda.ipynb`: explore features, target balance, missing values
- [ ] Train/test split + baseline Logistic Regression
- [ ] Record first test-set metrics in the README table

Phase 4 prep (Aubrenesia, can start in parallel):
- [ ] FastAPI skeleton in `api/` with a `/health` endpoint + test
- [ ] SQLAlchemy models in `deploymate/db/` (deployments, batches, metrics, alerts)

Phase 3 prep (Oba, can start in parallel):
- [ ] Sketch the drift scenario generator (`deploymate/simulate/`)
- [ ] Implement PSI + KS drift tests in `deploymate/monitor/` with unit tests

---

## 🧭 Decisions

| Date | Decision | Why |
|---|---|---|
| 2026-10-05 | ML core = classifier + drift/anomaly detector; LLM sits on top | Grading needs a dataset + real metrics; a pure LLM wrapper has neither |
| 2026-10-05 | Streamlit + Plotly instead of React + D3 | Free live hosting; everyone can work on it in Python |
| 2026-10-05 | scikit-learn instead of TensorFlow | Tabular data; TF bloats the deploy |
| 2026-10-05 | LLM has an offline fallback mode | The live demo must never break during the presentation |
| 2026-10-05 | Report only measured numbers, not the made-up resume figures | Honesty, and graders will ask |
| *pending* | **LLM provider:** recommendation is Claude, with the provider behind one small interface in `deploymate/llm/` so it can be swapped | See BLUEPRINT §4c |

## ❓ Open questions

1. ~~Third team member~~ Aubrenesia Tomilson (`aubrenesia`).
2. **LLM provider + API key:** Oba's account. Provider decision pending.
3. **Deadlines + rubric:** not posted yet.

---

## 📓 Session log

### 2026-10-08
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
- **Next session:** Marissa starts Phase 1 on `feature/data-pipeline`; Oba starts drift tests on `feature/drift-tests`.
