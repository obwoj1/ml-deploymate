# Project Title: ML DeployMate

>>> ML DeployMate monitors a deployed machine-learning model and detects when incoming data drifts away from the data the model was trained on. Per-feature statistical tests (Population Stability Index, Kolmogorov–Smirnov, chi-squared with Cramér's V) raise alerts, an Isolation Forest tracks unusual model behavior, and a large language model explains each alert in plain language. The monitored model is a gradient-boosted classifier trained on the UCI Adult census dataset.

## 👥 Team Members & Contributions
As per course requirements, all three members must actively collaborate and commit to this repository.
*   **Obaloluwa Wojuade ([obwoj1](https://github.com/obwoj1))** - Data pipeline, model training, drift monitoring, backend services, UI
*   **Marissa Savage ([sadiebird5](https://github.com/sadiebird5))** - Dashboard pages, deployment, model evaluation,  presentation
*   **Aubrenesia Tomlinson ([aubrenesia](https://github.com/aubrenesia))** - Dashboard application shell, drift monitoring, theme and navigation, wireframes

---

## 🚀 Live Demo
*   **Application URL:** https://ml-deploymate-2tgkbhmailycfxbjgpvyys.streamlit.app
*   *Note: This live demo must be fully functional and accessible during the live project presentation.*

The application is hosted on Streamlit Community Cloud. After a period of inactivity, the first load may take up to a minute while the app restarts.

### 🎥 Project Walkthrough
![Project Demo Walkthrough](assets/demo.gif)

---

## 🛠️ Installation & Setup

Follow these steps to set up the environment and run the project locally.

### Prerequisites
*   Python 3.13
*   `pip` or `conda`
*   Git
*   Optional: an Anthropic API key for AI-generated alert explanations. Without a key, explanations are generated from a template.

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

# Optional: add an API key
cp .env.example .env
```

### Running the Project
```bash
streamlit run app/Home.py        # dashboard at http://localhost:8501
uvicorn api.main:app --reload    # REST API at http://localhost:8000/docs
pytest                           # test suite
```
On first start, the application downloads the dataset and trains the model automatically.

Further documentation:
*   [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md): system design and methodology
*   [docs/DATASET.md](docs/DATASET.md): dataset description and cleaning
*   [docs/LOCAL_SETUP.md](docs/LOCAL_SETUP.md): detailed local setup
*   [docs/DEPLOY.md](docs/DEPLOY.md): deployment to Streamlit Community Cloud

---

## 📊 Dataset
*   **Source:** [UCI Adult (Census Income)](https://archive.ics.uci.edu/dataset/2/adult), 1994 U.S. Census, CC BY 4.0
*   **Description:** 48,842 records (32,561 training, 16,281 test). Twelve features are used: five numeric (age, years of education, capital gain, capital loss, hours per week) and seven categorical (workclass, marital status, occupation, relationship, race, sex, native country). The binary target is whether annual income exceeds $50K; 24% of records are positive.
*   **Data Pipeline:** The data is not tracked in git. `python scripts/download_data.py` downloads it into `data/raw/`, and `deploymate/data/load.py` cleans it (missing values are kept as an "Unknown" category; `fnlwgt` and `education` are dropped as a sampling weight and a duplicate of `education_num`).

---

## 🧠 Model Architecture & Training
*   **Algorithm:** Gradient Boosting (`HistGradientBoostingClassifier`), selected over Random Forest and a Logistic Regression baseline. The monitor combines per-feature drift tests with an Isolation Forest over batch-level model metrics.
*   **Frameworks Used:** scikit-learn, pandas, SciPy, Streamlit, Plotly, FastAPI, Anthropic API
*   **Training Process:** Each model is a scikit-learn `Pipeline` (standard scaling for numeric features, one-hot encoding for categorical features). The three candidates were compared with stratified 5-fold cross-validation on the training set and ranked by ROC-AUC, which is robust to the class imbalance. The selected model was refit on the full training set and evaluated once on the held-out test set. Reproduce with `python scripts/train.py`.

---

## 📈 Evaluation & Results

### Monitored model (Gradient Boosting)

| Metric | Training Set | Test Set |
| :--- | :--- | :--- |
| Accuracy | 0.884 | 0.873 |
| Precision | 0.805 | 0.773 |
| Recall | 0.684 | 0.652 |
| F1-Score | 0.740 | 0.707 |
| ROC-AUC | 0.943 | 0.928 |

A majority-class baseline scores 0.764 test accuracy. The close training and test scores indicate no significant overfitting. Hyperparameter tuning did not materially improve accuracy (0.8728 to 0.8731 in cross-validation), which is consistent with published results on this dataset (approximately 85–88%).

### Model comparison (5-fold cross-validation)

| Model | Accuracy | F1 | ROC-AUC |
| :--- | :--- | :--- | :--- |
| Gradient Boosting | 0.873 | 0.713 | 0.928 |
| Random Forest | 0.866 | 0.690 | 0.919 |
| Logistic Regression | 0.852 | 0.662 | 0.907 |

### Drift monitor

The monitor was evaluated on simulated traffic: the test set was replayed as batches of 400 rows, and five types of drift were injected into labeled batches (gradual age shift, sudden hours shift, occupation mix change, missing values, and noise), alongside healthy runs. Each scenario was run 10 times with independent random seeds (2,400 batches in total).

| Scenario | Precision | Recall | F1 | False-alarm rate | Detection delay (batches) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| healthy | – | – | – | 0.0% | – |
| gradual_shift | 1.000 | 0.736 | 0.848 | 0.0% | 7.4 |
| sudden_shift | 1.000 | 1.000 | 1.000 | 0.0% | 0.0 |
| category_shift | 1.000 | 1.000 | 1.000 | 0.0% | 0.0 |
| missing_values | 1.000 | 1.000 | 1.000 | 0.0% | 0.0 |
| noise | 1.000 | 1.000 | 1.000 | 0.0% | 0.0 |
| Overall | 1.000 | 0.931 | 0.965 | 0.0% | 1.5 |

All 50 drifted runs were detected. Missed batches occur only at the start of gradual drift, where the shift is still very small.

**Ablation.** Alerts are raised by the drift tests only. Allowing the Isolation Forest to trigger alerts added false alarms without a meaningful gain in recall, so it is reported as an informational signal:

| Detector | Precision | Recall | F1 | False-alarm rate |
| :--- | :--- | :--- | :--- | :--- |
| Drift tests (used for alerts) | 1.000 | 0.931 | 0.965 | 0.0% |
| Isolation Forest only | 0.975 | 0.287 | 0.444 | 0.6% |
| Either | 0.992 | 0.932 | 0.961 | 0.6% |

Reproduce with `python scripts/evaluate_monitor.py`. Raw results: [assets/monitor_results.json](assets/monitor_results.json) and [models/report.json](models/report.json). Exploratory analysis: [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb).

---

## 🧑‍💻 Git Workflow & Guidelines
To maintain collaboration integrity, remember to:
1.  **Branch Frequently:** Create feature branches (`git checkout -b feature/your-feature`).
2.  **Commit Regularly:** Write descriptive commit messages (`git commit -m "Add data augmentation pipeline"`).
3.  **Pull Before Push:** Always pull latest changes to avoid conflicts (`git pull origin main`).
