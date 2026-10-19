# Project Title: ML DeployMate

>>> ML DeployMate monitors deployed machine-learning models and catches problems before users notice them. Statistical drift tests and an Isolation Forest anomaly detector flag degrading models in real time, and a Claude-powered assistant explains each alert in plain English. It also generates deployment configs. **Live demo: [https://ml-deploymate-2tgkbhmailycfxbjgpvyys.streamlit.app](https://ml-deploymate-2tgkbhmailycfxbjgpvyys.streamlit.app)**

## 👥 Team Members & Contributions
As per course requirements, all three members must actively collaborate and commit to this repository.
*   **Obaloluwa Wojuade ([obwoj1](https://github.com/obwoj1))** - 
*   **Marissa Savage ([sadiebird5](https://github.com/sadiebird5))** -
*   **Aubrenesia Tomlinson ([aubrenesia](https://github.com/aubrenesia))** -

---

## 🚀 Live Demo
*   **Application URL:** **[https://ml-deploymate-2tgkbhmailycfxbjgpvyys.streamlit.app](https://ml-deploymate-2tgkbhmailycfxbjgpvyys.streamlit.app)** (Streamlit Community Cloud)
*   *First load after the app has been idle can take ~30 seconds while it wakes up and sets up the model.*
*   *Note: This live demo must be fully functional and accessible during the live project presentation.*

### 🎥 Project Walkthrough
![Project Demo Walkthrough](assets/demo.gif)
*(`demo.gif` will be added to `assets/` before the final submission deadline)*

---

## 🛠️ Installation & Setup

Follow these steps to set up the environment and run the project locally.

### Prerequisites
*   Python 3.13 (what the pinned packages are tested on)
*   `pip` or `conda`
*   Git
*   An Anthropic (Claude) API key (optional; without one, explanations use a built-in template)

### Local Installation
```bash
# Clone the repository
git clone https://github.com/obwoj1/ml-deploymate.git
cd ml-deploymate

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install required dependencies
pip install -r requirements-dev.txt

# Configure environment variables
cp .env.example .env      # then add your API key
```

**How everything works (every part, every decision): [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md)**

Full step-by-step guide (VS Code, notebooks, daily git workflow): [docs/LOCAL_SETUP.md](docs/LOCAL_SETUP.md)

### Run it
```bash
streamlit run app/Home.py            # dashboard → http://localhost:8501
uvicorn api.main:app --reload        # REST API → http://localhost:8000/docs
pytest                               # tests
```
The first start downloads the dataset and trains the model automatically (~15s). Deployment: [docs/DEPLOY.md](docs/DEPLOY.md).

---

## 📊 Dataset
*   **Source:** [UCI Adult / Census Income](https://archive.ics.uci.edu/dataset/2/adult) (1994 U.S. Census, CC BY 4.0)
*   **Description:** 48,842 people (32,561 train / 16,281 test), 14 features: 6 numeric (age, hours per week, capital gain, …) and 8 categorical (workclass, education, occupation, …). Binary target: `income > 50K` (24% positive, so imbalanced).
*   **Role in the project:** (1) trains the monitored classifier, (2) defines the "normal" reference distribution, (3) is replayed as simulated live traffic with labeled, injected drift to evaluate the detector.
*   **Data Pipeline:** Not tracked in git (`data/raw/` is gitignored). Run `python scripts/download_data.py`, or unzip [adult.zip](https://archive.ics.uci.edu/static/public/2/adult.zip) into `data/raw/`. Full walkthrough, column dictionary, and cleaning notes: [docs/DATASET.md](docs/DATASET.md).

---

## 🧠 Model Architecture & Training
*   **Algorithm:** Monitored model: Gradient Boosting (`HistGradientBoostingClassifier`), chosen over Random Forest and a Logistic Regression baseline. Monitor: PSI / KS / chi² drift tests + Isolation Forest.
*   **Frameworks Used:** scikit-learn, FastAPI, Anthropic Claude API, Streamlit, Plotly
*   **Training Process:** scikit-learn `Pipeline`: numeric features → `StandardScaler`, categorical → `OneHotEncoder(handle_unknown="ignore")`, then the classifier. Three classifiers compared with stratified 5-fold cross-validation on the training split; the winner by ROC-AUC (robust to the 76/24 class imbalance) is refit on all training data and scored once on the held-out test split. Run `python scripts/train.py` (~15s) to reproduce; it saves `models/model.joblib` + `models/report.json`.

---

## 📈 Evaluation & Results
*Results will be filled in as models are trained.*

**Monitored model: Gradient Boosting** (selected by 5-fold cross-validated ROC-AUC)

| Metric | Training Set | Test Set |
| :--- | :--- | :--- |
| Accuracy | 0.884 | 0.873 |
| Precision | 0.805 | 0.773 |
| Recall | 0.684 | 0.652 |
| F1-Score | 0.740 | 0.707 |
| ROC-AUC | 0.943 | 0.928 |

**Model comparison** (5-fold cross-validation on the training set)

| Model | CV Accuracy | CV F1 | CV ROC-AUC |
| :--- | :--- | :--- | :--- |
| **Gradient Boosting** | **0.873** | **0.713** | **0.928** |
| Random Forest | 0.866 | 0.690 | 0.919 |
| Logistic Regression (baseline) | 0.852 | 0.662 | 0.907 |

*For reference, always predicting `<=50K` scores 0.764 test accuracy. Most important features (permutation importance): marital status, capital gain, age, education. Full report: [models/report.json](models/report.json). Exploration: [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb).*

**Drift / anomaly monitor** (alerts come from the drift tests; the anomaly detector is an info signal). 10 simulated runs per scenario, 40 batches × 400 rows each, independent random seeds.

| Scenario | Precision | Recall | F1 | False-alarm rate | Detection delay (batches) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| healthy | – | – | – | 0.0% | – |
| gradual_shift | 1.000 | 0.736 | 0.848 | 0.0% | 7.4 |
| sudden_shift | 1.000 | 1.000 | 1.000 | 0.0% | 0.0 |
| category_shift | 1.000 | 1.000 | 1.000 | 0.0% | 0.0 |
| missing_values | 1.000 | 1.000 | 1.000 | 0.0% | 0.0 |
| noise | 1.000 | 1.000 | 1.000 | 0.0% | 0.0 |
| **Overall** | **1.000** | **0.931** | **0.965** | **0.0%** | **1.5** |

**Why alerts use only the drift tests (ablation):**

| Detector | Precision | Recall | F1 | False-alarm rate |
| :--- | :--- | :--- | :--- | :--- |
| **Drift tests (DeployMate alerts)** | **1.000** | **0.931** | **0.965** | **0.0%** |
| Anomaly detector alone | 0.975 | 0.287 | 0.444 | 0.6% |
| Alert if either fires | 0.992 | 0.932 | 0.961 | 0.6% |

*Letting the anomaly detector trigger alerts added 0.6% false alarms for +0.1% recall, so it's shown on the dashboard as an info signal instead. Every broken run was detected (50/50). Sudden problems are caught on the very first broken batch; gradual drift takes ~7 batches because the earliest shifts are tiny. Reproduce with `python scripts/evaluate_monitor.py`; raw numbers in [assets/monitor_results.json](assets/monitor_results.json).*

*Confusion matrices, ROC curves, and drift plots will go in [`assets/`](assets/).*

---

## 🧑‍💻 Git Workflow & Guidelines
To maintain collaboration integrity, remember to:
1.  **Branch Frequently:** Create feature branches (`git checkout -b feature/your-feature`).
2.  **Commit Regularly:** Write descriptive commit messages (`git commit -m "Add data augmentation pipeline"`).
3.  **Pull Before Push:** Always pull latest changes to avoid conflicts (`git pull origin main`).
4.  **Pull Requests Only:** No direct pushes to `main`. Every PR gets one teammate review.

Full team rules: [BLUEPRINT.md §8](BLUEPRINT.md#8-git-workflow-team-rules).
