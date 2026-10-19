# How ML DeployMate Works: The Complete Explanation

> Everything about the project in one place: what each part does, **how** it works, and **why** we built it that way.
> Written so all three of us can explain any part in the presentation or when Dr. Dacon asks.
> Every number here comes from the actual code and results files (as of 2026-10-19).

**Contents**
1. [The big picture](#1-the-big-picture)
2. [How the pieces connect](#2-how-the-pieces-connect)
3. [The dataset](#3-the-dataset)
4. [Cleaning the data](#4-cleaning-the-data)
5. [The model](#5-the-model)
6. [Drift detection: the core of DeployMate](#6-drift-detection-the-core-of-deploymate)
7. [The simulator: fake live traffic](#7-the-simulator-fake-live-traffic)
8. [The anomaly detector](#8-the-anomaly-detector)
9. [Evaluating the monitor: our headline result](#9-evaluating-the-monitor-our-headline-result)
10. [AI explanations](#10-ai-explanations)
11. [The service layer](#11-the-service-layer)
12. [The API](#12-the-api)
13. [The dashboard](#13-the-dashboard)
14. [Testing](#14-testing)
15. [Deployment](#15-deployment)
16. [Limitations and future work](#16-limitations-and-future-work)
17. [Questions we might get, with answers](#17-questions-we-might-get-with-answers)
18. [Glossary](#18-glossary)
19. [File map](#19-file-map)

---

## 1. The big picture

### The problem
When a machine-learning model is deployed, it was trained on data from the past. But the real world keeps changing: new kinds of users show up, a form changes, a data pipeline breaks. When the incoming data stops looking like the training data, the model's predictions quietly get worse. This is called **data drift**.

The dangerous part: **in real life you usually don't find out the correct answers until much later** (weeks or months), so you can't just watch the model's accuracy. By the time accuracy visibly drops, the damage is done.

### Our solution
**ML DeployMate watches a deployed model and catches drift early**, by checking every incoming batch of data against what "normal" looked like. When something changes, it raises an alert, and an AI explains the alert in plain English: what happened, the likely cause, and what to do.

### In one sentence
> We trained a model on census data, built a monitor that compares each new batch of data to the training data using statistical tests, proved it works by breaking data on purpose and measuring how much it catches, and wrapped it in an AI explainer, a dashboard and an API.

### The headline results
| What | Result |
|---|---|
| Model accuracy (test set) | **87.3%** (ROC-AUC 0.928) |
| Monitor precision (when it alerts, is it right?) | **100%** |
| Monitor recall (how many broken batches does it catch?) | **93.2%** |
| False alarms on healthy data | **0%** |
| Broken runs detected | **50 / 50** |
| Time to catch a sudden problem | **The very first broken batch** |

---

## 2. How the pieces connect

```
 ┌───────────────────┐   ┌──────────────────┐   ┌───────────────────────┐
 │ 1. DATA           │ → │ 2. MODEL         │ → │ 3. SIMULATOR          │
 │ download + clean  │   │ Gradient Boosting│   │ replays test data as  │
 │ UCI Adult census  │   │ 87.3% accuracy   │   │ live batches, breaks  │
 └───────────────────┘   └──────────────────┘   │ some on purpose       │
                                                └──────────┬────────────┘
                                                           ↓
 ┌───────────────────┐   ┌──────────────────┐   ┌───────────────────────┐
 │ 6. SERVICE        │ ← │ 5. AI EXPLAINER  │ ← │ 4. MONITOR            │
 │ one module that   │   │ alert → plain    │   │ drift tests → ALERT   │
 │ the dashboard and │   │ English (Claude  │   │ anomaly detector →    │
 │ API both use      │   │ or template)     │   │ info signal           │
 └────────┬──────────┘   └──────────────────┘   └───────────────────────┘
          ↓
   ┌──────────────┐   ┌──────────────┐
   │ 7. DASHBOARD │   │ 8. API       │
   │ Streamlit    │   │ FastAPI      │
   │ (for people) │   │ (for programs)│
   └──────────────┘   └──────────────┘
```

|
---

## 3. The dataset

Full guide: [DATASET.md](DATASET.md). Short version:

- **UCI Adult / Census Income**, from the 1994 U.S. Census. Free to use (CC BY 4.0).
- **48,842 people**: 32,561 for training and 16,281 for testing (the dataset comes pre-split).
- **The question:** does this person earn more than $50K a year? (yes/no = *binary classification*)
- **Lopsided:** 76% earn ≤$50K, 24% earn more.

### Why this dataset
| Reason | Why it matters for us |
|---|---|
| Real, messy table data | That's what most deployed models in industry actually use |
| Mix of number and category columns | Lets us demonstrate **both** kinds of drift test |
| 16k test rows | Enough to replay as hundreds of "live" batches |
| A well-known benchmark | Published models get ~85–87%, so we know our model is reasonable |
| Small (6 MB), trains in ~1 second | The live site can download and train it by itself |

---

## 4. Cleaning the data

**File:** `deploymate/data/load.py` · **Function:** `load_dataset()` → returns `X_train, X_test, y_train, y_test`

The raw files have quirks. Each one is fixed in code:

| Quirk | Fix | Why |
|---|---|---|
| No header row | We supply the 15 column names | Otherwise pandas uses the first person as headers |
| Leading spaces (`" State-gov"`) | `skipinitialspace=True` | Otherwise `"State-gov"` and `" State-gov"` look like different values |
| Missing values written as `?` | Read as missing, then filled with `"Unknown"` | See below |
| Test file has a junk first line | Skip it | It's not data |
| Test labels end in a period (`>50K.`) | Strip the `.` | Otherwise test labels never match training labels |

### Decision: keep missing values as "Unknown" instead of deleting rows
About 7% of rows have a missing job, employer or country. **We keep them, and "Unknown" becomes its own category.**
- **Why:** real live data has gaps too. If we deleted those rows, the model would never learn how to handle them, and it would break in production.
- **Bonus:** it lets the monitor notice when blanks *suddenly increase* (our `missing_values` scenario).

### Decision: drop two columns
| Column | Why it's dropped |
|---|---|
| `fnlwgt` | A census *sampling weight* (how many people this row represents in the population). It describes the survey, not the person, so it shouldn't drive predictions. |
| `education` | The same information as `education_num`, just as text. Keeping both double-counts it. |

That leaves **12 features**: 5 numbers (age, years of education, capital gain, capital loss, hours per week) and 7 categories (employer type, marital status, job, household role, race, sex, country).

---

## 5. The model

**Files:** `deploymate/model/pipeline.py`, `deploymate/model/train.py`, `scripts/train.py` · **Exploration:** `notebooks/01_eda.ipynb`

### 5.1 The pipeline: preprocessing + model in one object
Every model is a scikit-learn **Pipeline**:

```
raw row → [scale the number columns] + [one-hot encode the category columns] → classifier → prediction
```

- **Scaling (`StandardScaler`):** puts every number column on the same scale (mean 0, spread 1). Without it, `capital_gain` (up to 99,999) would dominate `age` (17–90) in models like Logistic Regression.
- **One-hot encoding (`OneHotEncoder`):** models need numbers, not words. Each category becomes its own yes/no column (e.g. `occupation_Tech-support = 1`).
- **`handle_unknown="ignore"`:** if live data contains a category the model has never seen (a brand-new job title), it's encoded as all zeros **instead of crashing**. That's essential for a monitoring tool, where the whole point is that live data changes.

**Why a Pipeline:** the saved model takes **raw rows** and does its own preprocessing. The API, the monitor and the dashboard never have to repeat the preprocessing steps, so they can't get them subtly wrong.

### 5.2 Step 1: a baseline
We first trained the simplest reasonable model, **Logistic Regression**: **85.3% test accuracy**.
**Why start with a baseline:** it's the bar every fancier model has to beat. If a complex model can't beat it, the complexity isn't worth it.

The "always guess ≤$50K" shortcut scores **76.4%**. That's why **accuracy alone is misleading** on this data, and why we also report F1 and ROC-AUC.

### 5.3 Step 2: compare three models with cross-validation
| Model | How it works (simply) | CV accuracy | CV F1 | **CV ROC-AUC** |
|---|---|---|---|---|
| **Gradient Boosting** ✅ | Builds many small decision trees one after another; each new tree fixes the previous ones' mistakes | 0.873 | 0.713 | **0.928** |
| Random Forest | Builds 300 decision trees independently and lets them vote | 0.866 | 0.690 | 0.919 |
| Logistic Regression | Draws one weighted line between the two classes | 0.852 | 0.662 | 0.907 |

**Cross-validation (5-fold):** we split the **training data** into 5 parts. Each model trains on 4 parts and is tested on the 5th, five times over, so every part gets a turn being the test. We average the 5 scores.
- **Why:** one train/test split can be lucky or unlucky. Averaging 5 is much more reliable (the spread across folds was only ±0.001–0.002 ROC-AUC).
- **The real test set is never touched during this.** It's used **exactly once**, at the very end, on the winner. That makes the final test score an honest estimate of real-world performance.

**Why the winner is picked by ROC-AUC:** ROC-AUC measures how well the model *ranks* people (higher-earners above lower-earners) across every possible cut-off. Unlike accuracy, it can't be gamed by always guessing the majority class.

**Why tree models beat Logistic Regression here:** income depends on *combinations* (e.g. married **and** highly educated **and** working long hours). Trees capture combinations naturally; a single weighted line can't.

### 5.4 Final model: Gradient Boosting
| Metric | Training set | **Test set** |
|---|---|---|
| Accuracy | 0.884 | **0.873** |
| Precision | 0.805 | **0.773** |
| Recall | 0.684 | **0.652** |
| F1 | 0.740 | **0.707** |
| ROC-AUC | 0.943 | **0.928** |

- **Train and test are close**, so the model **isn't overfitting** (memorizing training data).
- **87.3% matches published results** for this dataset (~85–87%).
- **Confusion matrix (test, 16,281 people):**

  | | Predicted ≤50K | Predicted >50K |
  |---|---|---|
  | **Actually ≤50K** | 11,697 ✅ | 738 ❌ |
  | **Actually >50K** | 1,338 ❌ | 2,508 ✅ |

  The model misses some high earners (recall 0.65). That's normal for an imbalanced dataset, and fine for our purpose: **the model is the thing being monitored, not the product.**

### 5.5 Which features matter most
**Permutation importance:** shuffle one column at a time and measure how much the test ROC-AUC drops. A big drop means the model relies on that column.

| Feature | Importance |
|---|---|
| marital_status | 0.084 |
| capital_gain | 0.057 |
| age | 0.052 |
| education_num | 0.029 |
| capital_loss | 0.015 |
| occupation | 0.014 |
| hours_per_week | 0.013 |
| relationship, workclass, sex | < 0.004 |
| native_country, race | ≈ 0 |

**Why permutation importance:** it works the same way for any model type, and it measures the *original* columns (not the dozens of one-hot columns).

### 5.6 Why the trained model file isn't on GitHub
`models/model.joblib` is gitignored. Saved scikit-learn models can break when library versions change, and retraining takes about 1 second. Anyone (including the live site) recreates it with `python scripts/train.py`. The **results** (`models/report.json`) *are* committed, because the dashboard displays them.

---

## 6. Drift detection: the core of DeployMate

**File:** `deploymate/monitor/drift.py` · **Function:** `detect_drift(reference, current)`

**The idea:** keep a *reference* sample of the training data (5,000 rows), which is what "normal" looks like. For every new batch, test each column: **does this column still look like the reference?**

Number columns and category columns need different tests.

### 6.1 Number columns: PSI (decides) + KS test (reported)

**PSI, the Population Stability Index**, is the industry standard for drift in credit scoring and finance.
1. Split the reference values into **10 bins**, each holding 10% of the data (bin edges come from the reference's quantiles).
2. Count what share of the new batch falls into each bin.
3. Compare: `PSI = Σ (new% − ref%) × ln(new% / ref%)`

| PSI | Meaning |
|---|---|
| < 0.1 | Stable |
| 0.1 – 0.25 | Moderate shift |
| > 0.25 | Major shift |

**Our threshold: 0.2** (drifted), and **≥ 0.5 counts as severe.**

**KS test (Kolmogorov–Smirnov):** measures the biggest gap between the two distributions and gives a p-value. **We report it, but don't use it to decide.** (Why: see 6.3.)

### 6.2 Category columns: chi-squared test + Cramér's V (decides)
- **Chi-squared test:** builds a table of how often each category appears in the reference vs. the batch, and checks whether the difference is bigger than chance.
- **Cramér's V:** turns that into an **effect size** from 0 (identical) to 1 (completely different).

**Our threshold: Cramér's V ≥ 0.1** (drifted), and **≥ 0.25 counts as severe.**

### 6.3 Why we decide with effect sizes, not p-values
A p-value answers "is there *any* difference at all?" With big batches, the answer is almost always yes: even a meaningless 0.5-year change in average age comes out "statistically significant." Deciding on p-values would mean **constant false alarms**.

Effect sizes (PSI, Cramér's V) answer the question that matters: **"is the difference *big*?"**

### 6.4 The rare-category fix (a problem we found and solved)
**The problem:** our first test run had **4–6 false alarms per run**, almost all on `native_country`. That column has 41 countries, and most are tiny (0.1% of people). In a 400-person batch, a rare country appearing 2 times instead of 0 looks like a huge relative change, purely by chance.

**The fix:** before testing, **categories under 1% of the reference are grouped into "Other"**. Brand-new categories go into "Other" too, so a sudden flood of a new value still gets caught.

**Result: false alarms went from 4–6 per run to 0**, with no loss in catching real problems.

### 6.5 What an alert contains
For each drifted column we also record **how** it shifted, used by the AI explanations:
- Number column: reference average vs. batch average (e.g. hours per week 40.3 → 55.1)
- Category column: which category changed most, and its share before vs. after (e.g. "Unknown" job 6% → 42%)

**Severity:** `high` if any shift is severe, or 2+ columns drifted; otherwise `medium`.

---

## 7. The simulator: fake live traffic

**File:** `deploymate/simulate/scenarios.py` · **Watch it:** `python scripts/simulate.py sudden_shift`

**The problem:** we can't wait months for real drift to happen, and we'd have no way to know which batches were "really" broken.

**The solution:** take the 16,281 test people, deal them out as **40 batches of 400** (like traffic arriving over time), and **break some batches on purpose**:

| Scenario | What we change | When it starts | Real-world story |
|---|---|---|---|
| `healthy` | Nothing | Never | A normal day |
| `gradual_shift` | Ages creep up, reaching +15 years by the end | 30% of the way in, ramping up | Customer base slowly aging |
| `sudden_shift` | Everyone works ~15 more hours/week | Halfway | A policy change or new data source |
| `category_shift` | Half the people switch to 3 occupations | Halfway | Launching in a new market |
| `missing_values` | 40% of employer/job/country fields go blank | Halfway | A broken upstream data pipeline |
| `noise` | Random jitter added to age and hours | Halfway | Data-entry or sensor errors |

**The key idea: because we did the breaking, every batch carries the true answer** ("was this batch really broken?"). That's what lets us grade the monitor with real precision and recall numbers.

**Why the first part of each run is healthy:** to measure false alarms, there must be healthy batches in every run.

---

## 8. The anomaly detector

**Files:** `deploymate/monitor/metrics.py`, `deploymate/monitor/anomaly.py`

The drift tests look at the **input columns**. The anomaly detector looks at **how the model is behaving**.

### 8.1 Label-free metrics
For each batch we compute 4 numbers **that don't need the correct answers**:
| Metric | What it is |
|---|---|
| `positive_rate` | Share of people predicted >$50K |
| `mean_confidence` | How sure the model is, on average |
| `low_confidence_share` | Share of predictions where it's less than 60% sure |
| `unknown_rate` | Share of blank ("Unknown") fields |

**Why label-free:** in production you don't know the correct answers yet. Accuracy is shown in the simulation (where we do know the answers), but the monitor never relies on it.

### 8.2 Two checks
1. **Adaptive thresholds (z-scores):** learn each metric's normal average and spread from **200 known-healthy batches**. Flag a batch if any metric is more than **4 standard deviations** from normal. The thresholds come from data, not guesswork.
2. **Isolation Forest:** an ML algorithm that finds outliers by looking at **all 4 metrics together**. It can flag a strange *combination* even when no single number is extreme. (It works by randomly splitting the data; unusual points get isolated in fewer splits.)

### 8.3 A tuning problem we found
scikit-learn's Isolation Forest has a `contamination` setting that **by design flags a fixed share (about 2%) of normal data as anomalies.** We measured it: 6–7 false alarms per test sweep. **Fix:** we set the alarm line at the **strangest healthy batch seen during calibration**, so a batch only fires if it's weirder than anything normal traffic produced.

### 8.4 Why it's an info signal, not an alert trigger
The evaluation (section 9) showed that letting the anomaly detector trigger alerts **added 0.6% false alarms for only +0.1% recall**. The drift tests already catch almost everything. So:
- **Alerts come from the drift tests only.**
- The anomaly detector is **still computed and shown on the dashboard** as extra information. It's a label-free safety net for problems that don't show up as column drift (and it independently catches 100% of `missing_values` batches).

This decision is based on our own measurements, which is a good point to make in the presentation.

---

## 9. Evaluating the monitor: our headline result

**Files:** `deploymate/monitor/evaluate.py`, `scripts/evaluate_monitor.py` · **Results:** `assets/monitor_results.json`

### 9.1 How we scored it
Every batch decision falls into one of four boxes:
| | Monitor alerted | Monitor stayed quiet |
|---|---|---|
| **Batch really broken** | ✅ Caught (true positive) | ❌ Missed (false negative) |
| **Batch really healthy** | ❌ False alarm (false positive) | ✅ Quiet (true negative) |

| Metric | Question it answers | Formula |
|---|---|---|
| **Precision** | When it alerts, is it right? | caught ÷ (caught + false alarms) |
| **Recall** | What share of broken batches does it catch? | caught ÷ (caught + missed) |
| **F1** | Balance of the two | 2 × P × R ÷ (P + R) |
| **False-alarm rate** | How often does it cry wolf on healthy data? | false alarms ÷ healthy batches |
| **Detection delay** | How many broken batches go by before the first alert? | (0 = caught immediately) |

### 9.2 The setup
- **Every scenario run 10 times** with different random batches: 6 scenarios × 10 runs × 40 batches = **2,400 batches** (1,080 broken, 1,320 healthy).
- **Each scenario gets its own random seed** (see 9.4).
- The monitor is calibrated on a **separate** set of 200 healthy batches, never the same ones it's scored on.

### 9.3 Results
| Scenario | Precision | Recall | F1 | False alarms | Delay |
|---|---|---|---|---|---|
| healthy | – | – | – | 0.0% | – |
| gradual_shift | 1.000 | 0.736 | 0.848 | 0.0% | 7.4 batches |
| sudden_shift | 1.000 | 1.000 | 1.000 | 0.0% | 0 |
| category_shift | 1.000 | 1.000 | 1.000 | 0.0% | 0 |
| missing_values | 1.000 | 1.000 | 1.000 | 0.0% | 0 |
| noise | 1.000 | 1.000 | 1.000 | 0.0% | 0 |
| **Overall** | **1.000** | **0.932** | **0.965** | **0.0%** | **1.5** |

**Reading it:**
- **Sudden problems:** caught on the very first broken batch, every time.
- **Gradual drift:** recall 74% and about 7 batches of delay, because the first ramped batches are shifted by only +1–2 years of age, which is practically undetectable *and arguably harmless*. Once the shift becomes meaningful, it's caught. **Every gradual run was still detected (10/10).**
- **Zero false alarms** across 1,320 healthy batches.

### 9.4 A bug we found in our own evaluation
Our first evaluation showed 1.4% false alarms. Investigating, we found that **every scenario was replaying the same healthy batches** (they shared a random seed), so **one false alarm was being counted up to 6 times**. After giving each scenario its own seed, the true rate was much lower. Testing the evaluation itself is a good presentation point.

### 9.5 The ablation: what each part contributes
| Detector | Precision | Recall | F1 | False alarms |
|---|---|---|---|---|
| **Drift tests (what DeployMate alerts on)** | **1.000** | **0.931** | **0.965** | **0.0%** |
| Anomaly detector alone | 0.975 | 0.287 | 0.444 | 0.6% |
| Alert if either fires | 0.992 | 0.932 | 0.961 | 0.6% |

This table is the evidence behind the decision in section 8.4.

---

## 10. AI explanations

**File:** `deploymate/llm/explain.py` · **Try it:** `python scripts/simulate.py category_shift --explain`

An alert like `occupation drifted, Cramér's V = 0.31` means nothing to most people. The AI turns it into:
- **What happened**
- **Likely cause**
- **Suggested action**

### A real example (Claude, `sudden_shift` scenario)
> **What happened:** Batch 20 triggered a high-severity alert because hours worked per week shifted sharply, with an average of 55.09 hours versus 40.31 in the training data (PSI 2.718, a very large shift). The model's outputs show no unusual behavior flags.
>
> **Likely cause:** This could be an upstream data issue, such as a changed unit, a scaling or default-value bug, or a field mapping error. It could also be a real change in who is in this batch.
>
> **Suggested action:** Inspect the raw hours-per-week values at the upstream source and compare them against recent pipeline or schema changes. If the shift is real and persists, collect fresh labeled data and retrain.

### How it works
1. **Only the alert's facts are sent**: which columns changed (in plain words, like "hours worked per week"), before/after numbers, severity, model-behavior numbers. **Never the raw data rows.** (Why: privacy, cost, and focus.)
2. **The instructions (system prompt) say "use ONLY the numbers given"**, and say causes must be worded as possibilities. (Why: so the AI explains what was *measured* instead of making things up.)
3. **Structured output:** the AI is forced to answer in an exact format (`summary`, `likely_cause`, `suggested_action`), so the dashboard can always display it.
4. **Model:** Anthropic **Claude** (`claude-opus-5-5`) at low effort, since it's a short writing task. Cost: well under a cent per explanation. If Claude's safety filter ever declines, a server-side fallback retries on a backup model in the same call.
5. **Provider order:** Claude if `ANTHROPIC_API_KEY` is set, else OpenAI (`gpt-4o-mini`) if `OPENAI_API_KEY` is set, else the template. **We switched from OpenAI to Claude** because the OpenAI account had no API credits; the code supports both (`deploymate/llm/claude.py`).

### The fallback: why it never breaks
If there's **no API key, no credits, the internet is down, or the AI refuses**, it automatically uses a **template explanation** built from the same numbers, e.g.:
> *"Batch 40 doesn't look like the training data: hours worked per week averaged 55.09 (normally 40.31)."*

The dashboard always works: for teammates without a key, in automated tests, and during the live demo.

**About the cost:** API usage is billed separately from ChatGPT/Claude subscriptions (both APIs are prepaid). Without a key or credits, the template is used.

---

## 11. The service layer

**Files:** `deploymate/service.py` (real), `deploymate/mock_service.py` (demo data)

**The rule:** the dashboard and the API only ever talk to **one module**, `deploymate.service`. It's the "menu" between the backend and the frontend:

| Function | Used by |
|---|---|
| `get_overview()` | Overview page, Home page |
| `list_scenarios()`, `run_scenario()` | Live Monitor page |
| `list_alerts()`, `explain_alert()` | Alerts page |
| `get_model_report()` | Model Performance page |
| `generate_config()` | Config Generator page |
| `predict()`, `check_batch()` | API only |

**Why this design:**
1. **The frontend team could start on day one.** We shipped a fake version first (realistic made-up data), and the pages were built against it. Later the real backend was plugged in behind the same functions, and **no page had to change**.
2. **Contract tests** run every function in both modes and check the shapes match. They caught one real mismatch (`missing_rate` vs `unknown_rate`) before it broke anyone's page.
3. **Demo-data fallback:** if the real backend can't start, everything quietly switches to demo data instead of crashing, and `get_overview()["mock"]` says which mode it's in (the Home page shows a banner).
4. **Self-setup:** on a brand-new server it downloads the dataset and trains the model by itself (~10–15 seconds the first time, then cached).

---

## 12. The API

**File:** `api/main.py` · **Run:** `uvicorn api.main:app --reload` → open http://localhost:8000/docs

**Why an API:** the dashboard is for **people**. An API is for **other programs**: a company's real system would send its live data to DeployMate automatically and get alerts back. The concept doc asks for a FastAPI backend.

| Endpoint | What it does |
|---|---|
| `GET /health` | Is it up? Running on real data or demo data? |
| `POST /predict` | Send people's details → get predictions |
| `POST /monitor/check` | Send **any batch** → get a drift report and alert |
| `POST /monitor/scenarios/{name}/run` | Replay a simulated scenario |
| `GET /alerts`, `GET /alerts/{id}/explain` | Alerts and their explanations |
| `GET /model/report` | Model metrics |
| `POST /config` | Generate deployment files |

- **Every endpoint is a thin wrapper around `service.py`**, so the API and dashboard always agree.
- **Input validation:** e.g. age must be 17–90; invalid requests get a clear error instead of a crash.
- **`/docs`** is an automatic interactive page where you can try every endpoint in the browser, which is good for the presentation.

---

## 13. The dashboard

**Folder:** `app/` · **Run:** `streamlit run app/Home.py` · **Built by:** Marissa and Bree

| Page | Shows |
|---|---|
| Home | What DeployMate is, status, navigation |
| Overview | Model health at a glance: status, accuracy, batches, alerts |
| Live Monitor | Pick a scenario → watch batches stream in → alert fires |
| Alerts | Every alert, with an "Explain with AI" button |
| Model Performance | Scores, confusion matrix, ROC curve, feature importance |
| Config Generator | A form → Dockerfile, CI workflow, .env file |

**Why Streamlit:** pure Python (no separate JavaScript frontend), free hosting on Streamlit Community Cloud, and charts via Plotly. The course lists Streamlit as an accepted demo platform.

---

## 14. Testing

**97 automated tests** in `tests/`, run with `pytest`, and **on every pull request automatically** (GitHub Actions CI).

| Test file | Checks |
|---|---|
| `test_load.py` | Every data quirk is fixed (uses tiny fake files with the same quirks) |
| `test_model.py` | The model learns, handles unseen categories, comparison and report |
| `test_drift.py` | Drift tests catch shifts, ignore healthy data, rare-category fix |
| `test_simulate.py` | Scenarios break the right things at the right time |
| `test_monitor.py` | Metrics, anomaly detector, alerts, severity, shift details |
| `test_evaluate.py` | The scoring math, using fake monitors with known answers |
| `test_explain.py` | AI explanations, the fallback, and that the real numbers reach the prompt (uses a **fake** AI client, so it never costs money) |
| `test_service.py` | Every dashboard function returns the right shape, in **both** real and demo mode |
| `test_api.py` | Every endpoint, input validation, error codes |

**Why tests matter here:** they caught real bugs (the metric-name mismatch), and they let us change things confidently without breaking teammates' pages.

---

## 15. Deployment

Full guide: [DEPLOY.md](DEPLOY.md).

- **Streamlit Community Cloud** (free), deploying from the `main` branch, entry point `app/Home.py`, Python 3.13.
- **Tested from a fresh copy** of the repo with no data, no model and no key: it installs in ~20 seconds, sets itself up, and every page loads on real data.
- **The API key** goes in Streamlit's **Secrets** settings, never in the code.
- ⚠️ **Free apps sleep after a few days with no visitors. Open the site 10 minutes before presenting.**

---

## 16. Limitations and future work

Being upfront about these shows we understand the project:

| Limitation | Future work |
|---|---|
| Drift is **simulated**, not from a real production system | Connect to a real data stream |
| We only detect **data drift** (inputs changing). **Concept drift** (the relationship between inputs and income changing) needs labels | Add delayed-label monitoring once true answers arrive |
| Gradual drift takes ~7 batches to catch | Add a cumulative test (e.g. CUSUM) that adds up small shifts over time |
| One model on one dataset | Support uploading any model + dataset |
| Alerts live in memory (they reset when the server restarts) | Store them in a database |
| **Bias:** the dataset includes race and sex and reflects 1994 inequalities | Add **fairness monitoring** (compare error rates across groups) |
| AI explanations need API credits | Already handled by the template fallback |

---

## 17. Questions we might get, with answers

**"Why not just monitor accuracy?"**
In production you don't know the correct answers until much later. Accuracy also barely moved in our tests (87.1% → 86.5%) even when the data clearly changed. Drift detection is the **early warning**.

**"Why Gradient Boosting?"**
It won all three cross-validated metrics (ROC-AUC 0.928 vs 0.919 vs 0.907). Income depends on combinations of features, which tree models capture.

**"How do you know the monitor works?"**
We broke data on purpose in 5 realistic ways, ran each scenario 10 times (2,400 batches), and measured: 100% precision, 93% recall, 0 false alarms, every broken run detected.

**"Why is recall only 93%?"**
All of the misses come from the earliest batches of *gradual* drift, where the shift is only +1–2 years of age. Every sudden problem was caught immediately, and every gradual run was eventually detected.

**"Why PSI and Cramér's V instead of p-values?"**
With big batches, p-values flag meaningless tiny differences. Effect sizes measure whether the change is *big*.

**"Isn't 100% precision suspicious?"**
It's measured on 1,320 healthy batches across independent runs. We found and fixed two sources of false alarms along the way (rare countries, and the Isolation Forest default), and we documented both.

**"Did you look at the test set while choosing the model?"**
No. Model selection used cross-validation on training data only. The test set was used once, at the end.

**"Why Claude and not OpenAI?"**
The concept doc said OpenAI, but the code is provider-agnostic: Claude is used when its key is set, OpenAI otherwise. We switched because our OpenAI account had no API credits. The explanations are grounded in the same measured numbers either way.

**"Can the AI make things up?"**
It only receives the measured numbers and is instructed to use only those. If it fails or isn't available, a template built from the same numbers is used.

**"What would you do with more time?"**
Real data streams, concept-drift detection with delayed labels, cumulative tests for gradual drift, and fairness monitoring.

---

## 18. Glossary

| Term | Plain meaning |
|---|---|
| **Data drift** | Incoming data stops looking like the training data |
| **Batch** | A group of incoming rows checked together (we use 400) |
| **Reference** | A sample of training data that defines "normal" |
| **Features / columns** | The details about each person (age, job, …) |
| **Label / target** | The answer being predicted (>$50K or not) |
| **Baseline** | The simple model every other model must beat |
| **Cross-validation** | Testing a model on 5 different splits and averaging |
| **Overfitting** | Memorizing training data instead of learning patterns |
| **Precision / recall / F1** | See section 9.1 |
| **ROC-AUC** | How well the model ranks positives above negatives (0.5 = random, 1 = perfect) |
| **PSI** | Drift score for number columns (> 0.2 = drifted here) |
| **Cramér's V** | Drift score for category columns (≥ 0.1 = drifted here) |
| **p-value** | Probability a difference is due to chance (misleading with big samples) |
| **Effect size** | How *big* a difference is |
| **Isolation Forest** | Algorithm that finds outliers by how easily they're isolated |
| **z-score** | How many standard deviations a value is from normal |
| **Ablation** | Removing parts to measure what each contributes |
| **Pipeline** | Preprocessing + model saved as one object |
| **One-hot encoding** | Turning a category into yes/no columns |
| **API** | A way for programs to talk to DeployMate |
| **Mock** | Fake but realistic data used before the real thing exists |
| **CI** | Automated tests that run on every pull request |

---

## 19. File map

```
ml-deploymate/
├── deploymate/                 the core Python package (Oba)
│   ├── data/load.py            load + clean the dataset
│   ├── data/download.py        download the dataset
│   ├── model/pipeline.py       preprocessing + model, scoring
│   ├── model/train.py          cross-validated model comparison + report
│   ├── simulate/scenarios.py   fake live traffic with labeled breaks
│   ├── monitor/drift.py        PSI, KS, chi², Cramér's V
│   ├── monitor/metrics.py      label-free batch metrics
│   ├── monitor/anomaly.py      z-scores + Isolation Forest
│   ├── monitor/pipeline.py     Monitor.check(): one report per batch
│   ├── monitor/evaluate.py     precision / recall / F1 / delay
│   ├── llm/explain.py          AI explanations + template fallback
│   ├── service.py              the one module the dashboard + API use
│   └── mock_service.py         demo-data fallback
├── api/main.py                 FastAPI app (Oba)
├── app/                        Streamlit dashboard (Marissa + Bree)
├── scripts/
│   ├── download_data.py        python scripts/download_data.py
│   ├── train.py                python scripts/train.py
│   ├── simulate.py             python scripts/simulate.py sudden_shift --explain
│   └── evaluate_monitor.py     python scripts/evaluate_monitor.py
├── notebooks/01_eda.ipynb      data exploration + baseline
├── models/report.json          model results (dashboard data)
├── assets/monitor_results.json monitor evaluation results
├── tests/                      97 automated tests
└── docs/                       DATASET, DEPLOY, TASKS, LOCAL_SETUP, this file
```
