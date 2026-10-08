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
- [ ] Decide LLM provider: OpenAI vs Claude (see Decisions below)

## ⏭️ Up next: Sprint 1 (full list in [docs/TASKS.md](docs/TASKS.md))

**Oba (ML + backend)**
- [ ] Mock `deploymate/service.py` **first**, so the frontend can start building
- [ ] `scripts/download_data.py` + `deploymate/data/load.py` (UCI Adult)
- [ ] EDA notebook + baseline model
- [ ] Drift simulator (`deploymate/simulate/`)

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
| *pending* | **LLM provider:** recommendation is Claude, with the provider behind one small interface in `deploymate/llm/` so it can be swapped | See BLUEPRINT §4c |

## ❓ Open questions

1. ~~Third team member~~ Aubrenesia Tomilson (`aubrenesia`).
2. **LLM provider + API key:** Oba's account. Provider decision pending.
3. **Deadlines + rubric:** not posted yet.

---

## 📓 Session log

### 2026-10-08
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
