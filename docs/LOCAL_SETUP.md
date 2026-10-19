# Local Setup

How to set up and run ML DeployMate on your own machine.

## 1. Clone the repository

```bash
git clone https://github.com/obwoj1/ml-deploymate.git
cd ml-deploymate
```

## 2. Create a virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
```

`requirements.txt` contains only what the application needs at runtime (this is what the deployed app installs). `requirements-dev.txt` adds the test and notebook tools.

Activate the environment in every new terminal session. In VS Code or Cursor, select the interpreter at `./venv/bin/python` (`Cmd+Shift+P` → *Python: Select Interpreter*).

## 3. Optional: configure an API key

```bash
cp .env.example .env
```

Add an `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`) to `.env` to enable AI-generated alert explanations. Without a key, explanations are generated from a template. `.env` is gitignored; never commit it.

## 4. Get the data and train the model

```bash
python scripts/download_data.py   # downloads UCI Adult into data/raw/
python scripts/train.py           # compares models, saves models/model.joblib and models/report.json
```

Both steps also run automatically the first time the dashboard or API starts.

## 5. Run

| Task | Command |
|---|---|
| Dashboard | `streamlit run app/Home.py` (run from the project root so the theme in `.streamlit/config.toml` is applied) |
| REST API | `uvicorn api.main:app --reload`, then open http://localhost:8000/docs |
| Tests | `pytest` |
| Watch a drift scenario in the terminal | `python scripts/simulate.py sudden_shift --explain` |
| Re-run the monitor evaluation | `python scripts/evaluate_monitor.py` |
| Exploratory analysis | open `notebooks/01_eda.ipynb` with the `venv` kernel |

## 6. Contributing

1. Create a branch from `main`.
2. Commit focused changes with descriptive messages.
3. Run `pytest` before pushing.
4. Open a pull request into `main`; at least one other team member reviews it before it is merged.

Merges into `main` redeploy the live application automatically.
