# The Dataset: UCI Adult (Census Income)

A plain-English guide to the data behind ML DeployMate: what it is, why we picked it, and what it does in each part of the project.

---

## 1. What it is

The **Adult dataset** (also called "Census Income") comes from the **1994 U.S. Census**. Ronny Kohavi and Barry Becker extracted it in 1996, and it's hosted by the UC Irvine Machine Learning Repository. It's one of the most widely used datasets for teaching and benchmarking ML on tabular data.

- **Source:** https://archive.ics.uci.edu/dataset/2/adult
- **Direct download:** https://archive.ics.uci.edu/static/public/2/adult.zip
- **License:** CC BY 4.0 (free to use with credit)

**Each row is one person. The question the model answers is: does this person earn more than $50,000 a year?**

## 2. What's in it

| | Training file (`adult.data`) | Test file (`adult.test`) |
|---|---|---|
| Rows (people) | 32,561 | 16,281 |
| Earn `<=50K` | 75.9% | 76.4% |
| Earn `>50K` | 24.1% | 23.6% |

The test file comes pre-split. We train on one file and grade on the other, so the model is always tested on people it has never seen.

### The 14 input features

| Feature | Type | What it means | Example |
|---|---|---|---|
| `age` | number | Age in years (17–90, average 38.6) | 39 |
| `workclass` | category (8) | Employer type | Private, State-gov, Self-emp-not-inc |
| `fnlwgt` | number | Census weight: how many people in the U.S. this row represents | 77516 |
| `education` | category (16) | Highest education level | HS-grad, Bachelors, Masters |
| `education_num` | number | Education as a number (1–16) | 13 = Bachelors |
| `marital_status` | category (7) | Marital status | Never-married, Married-civ-spouse |
| `occupation` | category (14) | Job type | Prof-specialty, Craft-repair, Exec-managerial |
| `relationship` | category (6) | Role in household | Husband, Wife, Own-child, Not-in-family |
| `race` | category (5) | Race | White, Black, Asian-Pac-Islander, … |
| `sex` | category (2) | Sex | Male, Female |
| `capital_gain` | number | Investment income (91.7% of people have 0) | 2174 |
| `capital_loss` | number | Investment losses | 0 |
| `hours_per_week` | number | Hours worked per week (median 40) | 40 |
| `native_country` | category (41) | Country of origin (~90% United-States) | United-States, Mexico |

### The target (what we predict)
`income`: either `<=50K` or `>50K`. This is a **binary classification** problem.

## 3. Quirks we have to clean up

| Quirk | Where | Fix (in `deploymate/data/load.py`) |
|---|---|---|
| No header row | both files | Supply the 15 column names in code |
| Leading spaces (`" State-gov"`) | both files | `skipinitialspace=True` |
| Missing values written as `?` | `workclass` (1,836), `occupation` (1,843), `native_country` (583) in train | Read `?` as missing; fill with `"Unknown"` |
| Junk first line (`\|1x3 Cross validator`) | `adult.test` | Skip the first row |
| Labels end in a period (`>50K.`) | `adult.test` | Strip the trailing `.` |

About 7% of training rows (2,399) have at least one missing value. Instead of dropping them, we keep them as an `"Unknown"` category, because real live data also has gaps.

## 4. Why we picked it

1. **It's real, messy, tabular data.** The kind of data most deployed ML models in industry actually run on.
2. **It mixes number and category columns.** That lets us show both of our drift tests: PSI/KS on numbers, chi-squared/Cramér's V on categories.
3. **It's big enough.** 48k rows gives us a solid training set plus enough held-out rows to replay as hundreds of "live" batches.
4. **It's a known benchmark.** Published models score around 85–87% accuracy, so we can tell if our model is reasonable.
5. **It's small and free.** 6 MB, trains in seconds, no account needed. Good for a live demo on a free host.

## 5. What it does in our project (three jobs)

### Job 1: Train the model that DeployMate watches
We train a classifier (Logistic Regression baseline → Random Forest / Gradient Boosting) on `adult.data` and score it on `adult.test`. This is the "deployed model" in our story. Its results fill the README's **Model** results table.

> ⚠️ **Accuracy alone is misleading here.** 76% of people earn `<=50K`, so a model that always guesses `<=50K` scores 76% accuracy while being useless. That's why we report **F1 and ROC-AUC** alongside accuracy.

### Job 2: Be the "normal" reference
The training data defines what "normal" looks like. Every live batch gets compared against it. If live data stops looking like the training data, that's **drift**, and the model's predictions can't be trusted as much.

### Job 3: Become fake "live traffic", some of it broken on purpose
We can't wait months for real-world drift, so we **simulate** it. The simulator slices `adult.test` into batches (e.g. 500 people each) and streams them in like real traffic. Some scenarios break the batches on purpose in realistic ways:

| Scenario | What we do to the batch | Real-world story |
|---|---|---|
| `healthy` | Nothing | Normal day |
| `gradual_shift` | Ages slowly creep upward batch by batch | Customer base aging over time |
| `sudden_shift` | Hours-per-week suddenly jumps | A policy change or a new data source |
| `category_shift` | Occupation mix changes sharply | Product launched to a new market |
| `missing_values` | Random fields blanked out | A broken upstream data pipeline |
| `noise` | Random noise added to numbers | Sensor or data-entry errors |

**Because we did the breaking, we know exactly which batches are broken.** That lets us grade our detector: what fraction of broken batches it caught (recall), how often it falsely alarmed (precision), and how many batches it took to notice (detection delay). Those numbers fill the README's **Detector** results table. **That's our headline result.**

## 6. Responsible-use note

The dataset includes `race`, `sex`, and `native_country`, and it reflects 1994 income patterns, including real historical inequalities. A model trained on it can learn and repeat those biases. For our project:
- We use it to demonstrate **monitoring**, not to make real decisions about people.
- In the presentation, we should acknowledge the bias and say that a real deployment would also need **fairness monitoring** (comparing error rates across groups). That's a natural "future work" point for DeployMate.

## 7. How to get it

```bash
python scripts/download_data.py     # once the script is written (Sprint 1)
```
Or by hand: download the zip above and unzip it into `data/raw/`. That folder is gitignored, so **the data is never pushed to GitHub**. Everyone (and the live server) downloads their own copy.

## 8. 30-second version (for the presentation)

> "We used the UCI Adult census dataset: about 48,000 people, 14 features like age, education, occupation, and hours worked, and the goal of predicting whether someone earns over $50K. It does three jobs for us. It trains the model we're monitoring. It defines what 'normal' data looks like. And when we replay it as live traffic and deliberately corrupt some batches, it lets us measure exactly how well DeployMate catches problems."
