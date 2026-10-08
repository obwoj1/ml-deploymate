# Project Title: ML DeployMate

>>> ML DeployMate monitors deployed machine-learning models and catches problems before users notice them. Statistical drift tests and an Isolation Forest anomaly detector flag degrading models in real time, and an OpenAI-powered assistant explains each alert in plain English and generates deployment configs. *(🚧 In progress. See [BLUEPRINT.md](BLUEPRINT.md) for the full plan.)*

## 👥 Team Members & Contributions
As per course requirements, all three members must actively collaborate and commit to this repository.
*   **Obaloluwa Wojuade ([obwoj1](https://github.com/obwoj1))** - ML & backend 
*   **Marissa Savage ([sadiebird5](https://github.com/sadiebird5))** - Live Monitor & Model Performance dashboard pages, demo walkthrough
*   **Aubrenesia Tomilson ([aubrenesia](https://github.com/aubrenesia))** - Dashboard app shell & theme,

---

## 🚀 Live Demo
*   **Application URL:** *TBD (Streamlit Community Cloud / Hugging Face Spaces)*
*   *Note: This live demo must be fully functional and accessible during the live project presentation.*

### 🎥 Project Walkthrough
![Project Demo Walkthrough](assets/demo.gif)
*(`demo.gif` will be added to `assets/` before the final submission deadline)*

---

## 🛠️ Installation & Setup

Follow these steps to set up the environment and run the project locally.

### Prerequisites
*   Python 3.9+
*   `pip` or `conda`
*   Git
*   An OpenAI API key (optional; the app runs in offline mode without one)

### Local Installation
```bash
# Clone the repository
git clone https://github.com/obwoj1/ml-deploymate.git
cd ml-deploymate

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env      # then add your API key
```

Full step-by-step guide (VS Code, notebooks, daily git workflow): [docs/LOCAL_SETUP.md](docs/LOCAL_SETUP.md)

---

## 📊 Dataset
*   **Source:** [UCI Adult / Census Income](https://archive.ics.uci.edu/dataset/2/adult) (1994 U.S. Census, CC BY 4.0)
*   **Description:** 48,842 people (32,561 train / 16,281 test), 14 features: 6 numeric (age, hours per week, capital gain, …) and 8 categorical (workclass, education, occupation, …). Binary target: `income > 50K` (24% positive, so imbalanced).
*   **Role in the project:** (1) trains the monitored classifier, (2) defines the "normal" reference distribution, (3) is replayed as simulated live traffic with labeled, injected drift to evaluate the detector.
*   **Data Pipeline:** Not tracked in git (`data/raw/` is gitignored). Run `python scripts/download_data.py`, or unzip [adult.zip](https://archive.ics.uci.edu/static/public/2/adult.zip) into `data/raw/`. Full walkthrough, column dictionary, and cleaning notes: [docs/DATASET.md](docs/DATASET.md).

---

## 🧠 Model Architecture & Training
*   **Algorithm:** Monitored model: Logistic Regression baseline → Random Forest / Gradient Boosting. Monitor: PSI / KS / chi² drift tests + Isolation Forest.
*   **Frameworks Used:** scikit-learn, FastAPI, SQLAlchemy, OpenAI API, Streamlit
*   **Training Process:** *TBD (Sprints 1–2, see [docs/TASKS.md](docs/TASKS.md))*

---

## 📈 Evaluation & Results
*Results will be filled in as models are trained.*

**Monitored model**

| Metric | Training Set | Test Set |
| :--- | :--- | :--- |
| Accuracy | — | — |
| F1-Score | — | — |
| ROC-AUC | — | — |

**Drift / anomaly detector** (on labeled injected-drift scenarios)

| Metric | Value |
| :--- | :--- |
| Precision | — |
| Recall | — |
| F1-Score | — |
| Mean detection delay (batches) | — |

*Confusion matrices, ROC curves, and drift plots will go in [`assets/`](assets/).*

---

## 🧑‍💻 Git Workflow & Guidelines
To maintain collaboration integrity, remember to:
1.  **Branch Frequently:** Create feature branches (`git checkout -b feature/your-feature`).
2.  **Commit Regularly:** Write descriptive commit messages (`git commit -m "Add data augmentation pipeline"`).
3.  **Pull Before Push:** Always pull latest changes to avoid conflicts (`git pull origin main`).
4.  **Pull Requests Only:** No direct pushes to `main`. Every PR gets one teammate review.

Full team rules: [BLUEPRINT.md §8](BLUEPRINT.md#8-git-workflow-team-rules).
