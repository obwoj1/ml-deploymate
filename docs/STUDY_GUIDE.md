# DeployMate Study Guide: Everything, Technically, Explained Simply

> For learning the project end to end, so you can present it and answer any question.
> [HOW_IT_WORKS.md](HOW_IT_WORKS.md) is the reference ("what we built"); this is the **teaching** version ("how and why it works"), including the ML basics underneath.
> Read it in order. Each part builds on the one before. "📂" shows where it lives in the code.

**Parts**
- [Part 1: ML basics you need](#part-1-ml-basics-you-need)
- [Part 2: Our data](#part-2-our-data)
- [Part 3: The models](#part-3-the-models)
- [Part 4: Measuring a model](#part-4-measuring-a-model)
- [Part 5: Why 87%, and can it be improved?](#part-5-why-87-and-can-it-be-improved)
- [Part 6: Drift detection](#part-6-drift-detection-the-heart-of-deploymate)
- [Part 7: The simulator and the anomaly detector](#part-7-the-simulator-and-the-anomaly-detector)
- [Part 8: Proving it works](#part-8-proving-it-works-the-evaluation)
- [Part 9: The AI explanations](#part-9-the-ai-explanations-claude)
- [Part 10: How the software fits together](#part-10-how-the-software-fits-together)
- [Part 11: How to present it](#part-11-how-to-present-it)
- [Part 12: Practice questions](#part-12-practice-questions-with-answers)

---

## Part 1: ML basics you need

### 1.1 What machine learning is
Normal programming: **you** write the rules ("if age > 40 and education > 12 then...").
Machine learning: you give the computer **examples with answers**, and it **finds the rules itself**.

We gave it 32,561 people, each with their details and the answer (earns >$50K or not). It learned patterns like "married + educated + older → more likely >$50K."

### 1.2 Features, labels, rows
| Term | Meaning | Ours |
|---|---|---|
| **Row / sample** | One example | One person |
| **Feature** | An input detail | age, education, job, hours... (12 of them) |
| **Label / target** | The answer to predict | income >$50K? (yes = 1, no = 0) |
| **X** | The table of features | `X_train`, `X_test` |
| **y** | The column of answers | `y_train`, `y_test` |

**Classification** = predicting a category (yes/no). **Regression** = predicting a number. **Ours is binary classification** (2 categories).

### 1.3 Train/test split: the single most important idea
If you study using the exact questions on the exam, a perfect score proves nothing. Same with models.

- **Training set** (32,561 people): the model **learns** from these.
- **Test set** (16,281 people): **hidden** from the model until the very end. We score it on these.

The test score tells you how the model does on **people it has never seen**, which is what matters in the real world.

The Adult dataset comes **pre-split** (two files), so we used that split. That way, our numbers are comparable to everyone else who uses this dataset.

**Golden rule:** never use the test set to make decisions (picking models, tuning settings). If you do, it's no longer a fair exam. We used it **exactly once**, at the very end.

### 1.4 Overfitting and underfitting
- **Overfitting** = memorizing instead of learning. It's great on training data and bad on new data. Like a student who memorized last year's answers.
- **Underfitting** = too simple to learn the pattern. Bad on both.
- **How to spot overfitting:** a big gap between train and test scores.
- **Ours:** train 88.4% vs test 87.3%. That's a tiny gap, so **no overfitting.** ✅

### 1.5 Cross-validation (CV)
**Problem:** to compare models, we need a score, but we can't use the test set (golden rule). And one split of the training data can be lucky or unlucky.

**Solution, 5-fold cross-validation:**
1. Cut the training data into 5 equal parts.
2. Train on parts 1–4, test on part 5. Then train on 1,2,3,5 and test on 4. And so on, 5 times.
3. Average the 5 scores.

Every row gets used for testing exactly once, and the average is **reliable**: our scores varied only ±0.002 across the 5 folds.

"**Stratified**" means each part keeps the same 76/24 mix of the two classes, so no fold is accidentally lopsided.

📂 `deploymate/model/train.py` → `compare_models()`

### 1.6 Preprocessing: getting data model-ready
Models only understand numbers, on comparable scales. Two steps:

**Scaling (StandardScaler)** for number columns:
`scaled = (value − average) / spread`
After this, every number column has average 0 and spread 1. **Why:** capital_gain goes up to 99,999 while age stops at 90. Without scaling, some models would think capital_gain matters 1,000× more just because its numbers are bigger.

**One-hot encoding** for category columns:
`occupation = "Sales"` becomes columns like `occupation_Sales = 1`, `occupation_Tech-support = 0`, ... **Why:** you can't do math on words. And you can't just number the categories (Sales = 1, Tech = 2), because that would wrongly imply Tech is "twice" Sales.

**`handle_unknown="ignore"`:** if live data has a job the model never saw, it becomes all zeros instead of crashing. Essential for a **monitoring** tool, since live data *will* surprise you.

**Pipeline:** preprocessing + model packed into one object. Raw data goes in, a prediction comes out. **Why:** the exact same preprocessing is guaranteed everywhere (training, the API, the dashboard), so nothing can get out of sync.

📂 `deploymate/model/pipeline.py`

---

## Part 2: Our data

📂 `deploymate/data/load.py`, `docs/DATASET.md`

- **UCI Adult**, from the 1994 U.S. Census: 48,842 people, a famous benchmark dataset.
- **Task:** predict income >$50K.
- **Imbalanced:** 76% earn ≤$50K, 24% earn more. This matters a lot (Part 4).

### The 12 features we use
- **5 numbers:** age, education_num (years of school, 1–16), capital_gain, capital_loss, hours_per_week
- **7 categories:** workclass, marital_status, occupation, relationship, race, sex, native_country

### Cleaning decisions and why
| Decision | Why |
|---|---|
| `?` → "Unknown" (kept, not deleted) | 7% of rows have a gap. Real live data has gaps too, and deleting them would teach the model a world with no gaps |
| Dropped `fnlwgt` | It's a census **sampling weight** (how many people a row represents), not a fact about the person |
| Dropped `education` | Same info as `education_num`, so keeping both double-counts it |
| Fixed spaces, the junk line, `>50K.` labels | File quirks; otherwise the same value looks different |

---

## Part 3: The models

We compared **3 models**. Same data and preprocessing; only the "brain" differs.

### 3.1 Logistic Regression (the baseline)
**How it works:** gives every feature a **weight** (how much it pushes toward ">$50K"), adds them up, and squashes the total into a probability between 0 and 1.
`score = w1×age + w2×education + ...` → `probability = 1 / (1 + e^(−score))`

- ✅ Simple, fast, easy to explain (you can read the weights)
- ❌ Each feature acts **on its own**. It can't learn combinations like "married **and** educated **and** works long hours"
- **Result:** 85.2% CV accuracy

**Why start with it:** a **baseline** is the bar to beat. If a fancy model can't beat a simple one, the fancy one isn't worth it.

### 3.2 Decision trees (the building block of the next two)
A tree asks yes/no questions:
```
Married?
├─ yes → Education ≥ 13 years?
│        ├─ yes → likely >$50K
│        └─ no  → Capital gain > 5000? ...
└─ no  → likely ≤$50K
```
Trees naturally capture **combinations**. One tree alone is unstable and overfits, so we combine many.

### 3.3 Random Forest
**How it works:** builds **300 trees**, each on a random sample of the rows and a random subset of the features, then **they vote**.
- ✅ Many slightly different trees average out each other's mistakes ("wisdom of the crowd")
- **Result:** 86.6% CV accuracy

### 3.4 Gradient Boosting ✅ (the winner)
**How it works:** builds trees **one after another**. Each new tree focuses on **the mistakes the previous trees made**.
- Tree 1 makes rough predictions.
- Tree 2 learns to fix tree 1's errors.
- Tree 3 fixes what's still wrong... and so on, about 100 times.

Like a student who reviews only the questions they got wrong, over and over.
- ✅ Usually the strongest model for **table data** like ours
- We used `HistGradientBoostingClassifier`, a fast version that groups numbers into bins
- **Result:** 87.3% CV accuracy, **ROC-AUC 0.928**

### 3.5 The comparison
| Model | CV accuracy | CV F1 | CV ROC-AUC |
|---|---|---|---|
| **Gradient Boosting** | **0.873** | **0.713** | **0.928** |
| Random Forest | 0.866 | 0.690 | 0.919 |
| Logistic Regression | 0.852 | 0.662 | 0.907 |

**Why tree models win:** income depends on **combinations** of features, and trees capture combinations; a weighted sum can't.

**Why we pick by ROC-AUC** (not accuracy): see Part 4.

📂 `deploymate/model/train.py`, `scripts/train.py` (run: `python scripts/train.py`)

---

## Part 4: Measuring a model

### 4.1 The confusion matrix: where every metric comes from
On the 16,281 test people:

| | Predicted ≤$50K | Predicted >$50K |
|---|---|---|
| **Actually ≤$50K** | **11,697** correct ✅ (true negative) | 738 wrong ❌ (false positive) |
| **Actually >$50K** | 1,338 wrong ❌ (false negative) | **2,508** correct ✅ (true positive) |

### 4.2 The metrics
| Metric | Question | Ours | Formula |
|---|---|---|---|
| **Accuracy** | What share of all predictions were right? | 87.3% | (11,697 + 2,508) / 16,281 |
| **Precision** | When it says ">$50K", how often is it right? | 77.3% | 2,508 / (2,508 + 738) |
| **Recall** | Of the people who really earn >$50K, how many did it find? | 65.2% | 2,508 / (2,508 + 1,338) |
| **F1** | Balance of precision and recall | 0.707 | 2 × P × R / (P + R) |
| **ROC-AUC** | How well does it **rank** people (high earners above low)? | 0.928 | Area under the ROC curve |

### 4.3 Why accuracy alone is misleading here
A model that **always says "≤$50K"** gets **76.4% accuracy** while being completely useless; it finds zero high earners. So 87% sounds less impressive once you know 76% is free. That's why we also report F1 and ROC-AUC, which a lazy model can't fake.

### 4.4 ROC-AUC, explained simply
The model actually outputs a **probability** (like "73% likely >$50K"), and we turn it into yes/no with a **threshold** (default 0.5).
- The **ROC curve** shows the trade-off at **every** possible threshold: how many real high earners you catch vs. how many false alarms you get.
- **AUC** = the area under that curve. **0.5 = coin flip, 1.0 = perfect.**
- Plain-English meaning of **0.928**: pick one random high earner and one random low earner, and the model gives the high earner the higher probability **92.8% of the time**.
- **Why we pick by it:** it judges the model's ranking skill **without** depending on one threshold, and the class imbalance can't fool it.

### 4.5 Feature importance
**Permutation importance:** shuffle one column at a time (scrambling it into nonsense) and see how much the score drops. A big drop means the model relied on it.

Top features: **marital_status, capital_gain, age, education_num**. Race and country are near zero.

📂 the Model Performance page shows all of this; data in `models/report.json`

---

## Part 5: Why 87%, and can it be improved?

### 5.1 Why not higher
1. **The data doesn't contain enough information.** Income depends on things that aren't in the dataset: industry, city, years of experience, company size, negotiation... Two people with **identical** rows can earn very different amounts. **No model can predict what the data doesn't contain.** This limit is called the *irreducible error* (or *Bayes error*).
2. **It's a known ceiling.** Published results on this dataset land around **85–88%**. We're right in that range.
3. **The labels are noisy.** It's self-reported 1994 survey data.

### 5.2 We tested improvements for real
(Done on the training data with cross-validation, to keep the test set fair.)

| Attempt | Result |
|---|---|
| **Tuning** Gradient Boosting (slower learning, more trees, early stopping, regularization) | 87.28% → **87.31%**: basically nothing |
| **Lowering the decision threshold** 0.5 → 0.4 (chosen on training data) | Recall **65% → 73%**, F1 **0.707 → 0.719**, accuracy 87.3% → 86.6% |

**What this tells us:**
- Tuning doesn't help, so **the model is already at the data's ceiling.**
- The threshold is a **business choice**, not a fix. If missing a high earner is costly, use 0.4 and catch more of them (with more false positives). We kept 0.5, the standard default.

### 5.3 Other ways it could improve (for "future work")
- **XGBoost / LightGBM / CatBoost:** other boosting libraries, maybe +0.5%
- **Feature engineering:** e.g. "has any capital gain" as a yes/no, or age groups
- **Class weights:** tell the model that missing a high earner costs more
- **More/better data:** the only thing that would *really* move it (industry, location...)

### 5.4 Why it doesn't really matter for our project
**The model isn't the product. The monitor is.** The model's job is to be a realistic "patient" for DeployMate to watch. An 87% model on a standard benchmark is exactly that. Our real results are the **monitor's** (Part 8).

---

## Part 6: Drift detection, the heart of DeployMate

📂 `deploymate/monitor/drift.py`

### 6.1 The idea
Keep a **reference** sample: 5,000 rows of training data = "normal." For each new **batch** (400 rows), ask, **column by column**: *does this still look like normal?*

Numbers and categories need different tests.

### 6.2 Number columns: PSI (decides) + KS test (just reported)
**PSI, the Population Stability Index.** Standard in banking/credit.
1. Split the reference values into **10 buckets**, each holding 10% of the reference data.
2. See what % of the new batch falls into each bucket.
3. Add up how different the percentages are: `PSI = Σ (new% − ref%) × ln(new% / ref%)`

Example: normally 10% of people are in the "45–50 hours" bucket. If suddenly 40% are, PSI shoots up.

| PSI | Meaning |
|---|---|
| < 0.1 | stable |
| 0.1–0.25 | moderate change |
| > 0.25 | big change |
| **Our alarm line: 0.2** · severe: 0.5 | |

**KS test (Kolmogorov–Smirnov):** finds the biggest gap between the two distributions and gives a **p-value**. We show it, but **don't** decide with it (6.4 explains why).

### 6.3 Category columns: chi-squared + Cramér's V (decides)
- **Chi-squared test:** counts each category in the reference vs. the batch, and checks whether the difference is more than chance.
- **Cramér's V:** turns that into an **effect size** between 0 (identical) and 1 (totally different).
- **Our alarm line: 0.1** · severe: 0.25

### 6.4 Why effect sizes, not p-values (a smart choice; mention it)
- A **p-value** answers "is there **any** difference at all?"
- With hundreds of rows, even a meaningless tiny change (say the average age moves 0.3 years) comes out "statistically significant." Deciding by p-value would give **constant false alarms.**
- An **effect size** (PSI, Cramér's V) answers "is the difference **big**?", which is what matters.

### 6.5 The rare-category fix (a bug we found)
- **Problem:** `native_country` has 41 values, most tiny (0.1%). In a 400-person batch, a rare country showing up 2 times instead of 0 *looks* like a huge change, just by chance. That caused **4–6 false alarms per run**.
- **Fix:** before testing, group categories under 1% into **"Other."**
- **Result:** false alarms went to **0**, with no loss in catching real problems.

### 6.6 What an alert contains
For each column that drifted: the score, and **how** it moved (e.g. "hours: 40.3 → 55.1 average", or "'Unknown' job: 6% → 42%").
**Severity:** high if any change is severe or 2+ columns drifted; otherwise medium.

---

## Part 7: The simulator and the anomaly detector

### 7.1 The simulator
📂 `deploymate/simulate/scenarios.py` · run: `python scripts/simulate.py sudden_shift`

**Problem:** we can't wait months for real drift, and with real drift we'd never know for sure which batches were "really" broken.

**Solution:** take the 16,281 test people, deal them out as **40 batches of 400** (like a stream of traffic), and **break some on purpose**:

| Scenario | What we change | Real-world version |
|---|---|---|
| healthy | nothing | a normal day |
| gradual_shift | ages slowly creep up (+15 years by the end) | customers aging |
| sudden_shift | everyone +15 hours/week from halfway | a system/unit change |
| category_shift | half switch to 3 jobs | a new market |
| missing_values | 40% of job fields blank | a broken data pipeline |
| noise | random jitter on age/hours | data-entry errors |

**The key idea:** we did the breaking, so **we know exactly which batches are broken**. That's what lets us score the monitor (Part 8). The first part of each run is healthy, so we can also count false alarms.

### 7.2 The anomaly detector (info signal)
📂 `deploymate/monitor/metrics.py`, `anomaly.py`

The drift tests look at the **input columns**. This looks at the **model's behavior**, using 4 numbers that don't need the true answers:
- share predicted >$50K
- average confidence
- share of unsure predictions (<60% sure)
- share of blank fields

Two checks, both learned from 200 known-healthy batches:
1. **z-score:** how many "spreads" a number is from its normal average. Flag if more than 4.
2. **Isolation Forest:** an ML outlier finder. It randomly splits the data; weird points get isolated in very few splits. It catches **weird combinations**.

**A problem we found:** scikit-learn's default Isolation Forest setting (`contamination`) **by design** flags ~2% of normal data. We changed it to "only flag if weirder than the weirdest healthy batch we've seen."

**Final decision: it's an info signal, not an alarm.** The evaluation showed it added false alarms and almost no extra catches (Part 8.4). It's still shown on the dashboard as extra information.

---

## Part 8: Proving it works (the evaluation)

📂 `deploymate/monitor/evaluate.py` · run: `python scripts/evaluate_monitor.py`

### 8.1 Scoring the monitor (same idea as scoring the model)
| | Monitor alerted | Monitor quiet |
|---|---|---|
| **Batch really broken** | ✅ caught | ❌ missed |
| **Batch really healthy** | ❌ false alarm | ✅ quiet |

- **Precision:** when it alerts, is it right?
- **Recall:** what share of broken batches did it catch?
- **False-alarm rate:** how often does it cry wolf on healthy batches?
- **Detection delay:** how many broken batches go by before the first alert?

### 8.2 Setup
6 scenarios × 10 runs × 40 batches = **2,400 batches** (1,080 broken, 1,320 healthy). Each run uses different random batches.

### 8.3 Results
| | Precision | Recall | False alarms | Delay |
|---|---|---|---|---|
| sudden / category / missing / noise | 100% | 100% | 0% | **0**: caught on the first broken batch |
| gradual_shift | 100% | 74% | 0% | ~7 batches |
| **Overall** | **100%** | **93%** | **0%** | 1.5 |

**Why gradual is 74%:** the first gradual batches are shifted by only +1–2 years of age. That's practically invisible, and harmless. Once it becomes meaningful, it's caught. **All 10 gradual runs were detected.**

### 8.4 The ablation: "what does each part add?"
| Detector | F1 | False alarms |
|---|---|---|
| **Drift tests (what DeployMate alerts on)** | **0.965** | **0%** |
| Anomaly detector alone | 0.444 | 0.6% |
| Alert if either fires | 0.961 | 0.6% |

That's why alerts come only from the drift tests. **Decisions backed by measurements** is a strong point to make.

### 8.5 A bug we found in our own evaluation
The first evaluation showed 1.4% false alarms. Investigating, we found every scenario was **replaying the same healthy batches** (they shared a random seed), so one false alarm got counted up to 6 times. After fixing it, the real number was 0%. **Checking our own testing** is worth mentioning.

---

## Part 9: The AI explanations (Claude)

📂 `deploymate/llm/explain.py`, `deploymate/llm/claude.py`

**Why:** "occupation drifted, Cramér's V = 0.31" means nothing to most people. Claude turns an alert into:
- **What happened** · **Likely cause** · **Suggested action**

**How:**
1. We send **only the alert's facts** (which columns changed and by how much, in plain words), **never the raw data**. That's for privacy, cost and focus.
2. The instructions say "**use ONLY these numbers**," so it explains what was measured instead of making things up.
3. **Structured output:** Claude must answer in the exact 3-field format, so the dashboard can always show it.
4. Model: `claude-opus-5-5`, low effort (it's a short task). Under a cent per explanation.

**Never breaks:** no key, no internet, or an error → it automatically uses a **template** built from the same numbers.

**Why Claude, not OpenAI:** the concept doc said OpenAI, but our OpenAI account had no API credits. The code supports both (Claude first, then OpenAI, then template).

---

## Part 10: How the software fits together

```
data/load.py ──► model (pipeline.py, train.py) ──► monitor (drift, metrics, anomaly, pipeline)
                                                        │
simulate/scenarios.py (fake live traffic) ─────────────┤
                                                        ▼
                                  service.py  ◄── llm/explain.py (Claude)
                                   │      │
                         app/ (dashboard)  api/main.py (API)
```

- **`service.py` is the "menu" between the backend and the screens.** Pages only call its functions (`get_overview`, `run_scenario`, `list_alerts`, `explain_alert`, ...). We shipped a fake version first, so the dashboard could be built before the ML was done, then swapped in the real one without changing the pages.
- **Dashboard (Streamlit):** Python that becomes a website. 5 pages. Live at https://ml-deploymate-2tgkbhmailycfxbjgpvyys.streamlit.app
- **API (FastAPI):** the same features for other programs. `uvicorn api.main:app` → `/docs`
- **102 automated tests** (`pytest`), run on every PR by GitHub Actions
- **Self-setup:** on a fresh server, it downloads the data and trains the model by itself

---

## Part 11: How to present it

**A suggested 8-minute flow:**
1. **The problem (1 min):** models silently get worse when data changes; accuracy can't be watched live because true answers arrive late.
2. **Our solution (30 s):** DeployMate watches the incoming data, alerts on drift, and explains it with AI.
3. **Data + model (1 min):** UCI Adult, 48k people; compared 3 models with cross-validation; Gradient Boosting at 87.3% (and why 87% is the data's ceiling).
4. **How drift detection works (1.5 min):** PSI and Cramér's V; effect sizes, not p-values; the rare-category fix.
5. **Live demo (2 min):** Live Monitor → `healthy` (quiet) → `sudden_shift` (red bars at batch 16) → Alerts → Explain with AI.
6. **Results (1.5 min):** 2,400 batches, 100% precision, 93% recall, 0 false alarms; the ablation; the bug we caught.
7. **Limitations + future work (30 s):** simulated drift, no concept drift, gradual drift is slower, fairness monitoring.

**Demo tips:** open the site 10 minutes early (free apps sleep). Have the local version (`streamlit run app/Home.py`) as a backup.

---

## Part 12: Practice questions (with answers)

**What's the difference between training and test data?**
The model learns from training data; the test data stays hidden until the end, so the score reflects people it has never seen.

**What's overfitting, and do you have it?**
Memorizing instead of learning. No: train 88.4% vs test 87.3%.

**What's cross-validation, and why use it?**
Splitting the training data into 5 parts, testing on each part in turn, and averaging. It gives reliable scores for comparing models without touching the test set.

**Why Gradient Boosting?**
It won all three cross-validated metrics. It builds trees one after another, each fixing the previous one's mistakes, and it captures feature combinations.

**Why is accuracy only 87%?**
The data doesn't contain everything that determines income (industry, location, experience...). That's the ceiling: published results are 85–88%, and our tuning only moved it 0.03%.

**How would you improve it?**
Better data first. Then feature engineering, other boosting libraries, and class weights. Or lower the threshold to catch more high earners (recall 65% → 73%) at a small accuracy cost.

**Why isn't accuracy enough?**
Always guessing "≤$50K" scores 76%. F1 and ROC-AUC can't be faked that way.

**What does ROC-AUC 0.928 mean?**
92.8% of the time, the model ranks a random high earner above a random low earner.

**What's data drift, and why does it matter?**
Live data stops looking like training data, so predictions quietly degrade. It matters because true answers arrive late, so you can't watch accuracy live.

**How does PSI work?**
Bucket the reference into 10 equal groups, see how the new batch spreads across them, and sum the differences. Over 0.2 = drifted.

**Why not use p-values?**
With big batches, tiny meaningless changes look "significant." Effect sizes measure whether the change is big.

**How do you know the monitor works?**
We broke data on purpose in 5 ways, 10 runs each, 2,400 batches: 100% precision, 93% recall, 0 false alarms, every broken run caught.

**Isn't 100% precision suspicious?**
It's over 1,320 healthy batches with independent runs, and we found and fixed two sources of false alarms along the way, which we documented.

**Why is the anomaly detector not used for alerts?**
Measured: it added 0.6% false alarms for +0.1% recall. It's shown as information instead.

**Can the AI make things up?**
It only gets the measured numbers and is told to use only those. If it fails, a template is used.

**What are the limitations?**
Simulated drift; we catch data drift, not concept drift (which needs labels); gradual drift takes ~7 batches; there's bias in the 1994 data.
