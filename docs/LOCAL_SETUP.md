# Local Setup: open and run ML DeployMate in VS Code

Step-by-step for all three of us. Do this once per computer.

---

## 1. Get the code on your computer

**If you don't have it yet** (Marissa, Aubrenesia):
```bash
cd ~/Developer        # or wherever you keep projects
git clone https://github.com/obwoj1/ml-deploymate.git
cd ml-deploymate
```

**If you already have it** (Oba: `~/Desktop/ml-deploymate`), just pull the latest:
```bash
cd ~/Desktop/ml-deploymate
git pull origin main
```

## 2. Open it in VS Code

```bash
code .
```
(Or open VS Code → **File → Open Folder…** → pick `ml-deploymate`.)

Install these VS Code extensions (Extensions sidebar, `Cmd+Shift+X`):
- **Python** (Microsoft)
- **Jupyter** (Microsoft): for the `.ipynb` notebooks

## 3. Create the virtual environment

The virtual environment (`venv`) is a private copy of Python just for this project, so its packages don't mess with anything else on your computer.

Open the VS Code terminal (**Terminal → New Terminal**, or `` Ctrl+` ``), then:
```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
When it worked, your terminal prompt starts with `(venv)`. **Activate it every time you open a new terminal.**

## 4. Point VS Code at the venv

`Cmd+Shift+P` → **Python: Select Interpreter** → pick the one that says `./venv/bin/python`.
Now the editor's autocomplete, the run button, and notebooks all use the project's packages.

## 5. Add your secrets

```bash
cp .env.example .env
```
Open `.env` and paste the API key (ask Oba, and **never commit `.env`**; git already ignores it).

## 6. Get the dataset

Once `scripts/download_data.py` exists (Phase 1):
```bash
python scripts/download_data.py
```
That puts the data in `data/raw/`. Git ignores the folder, so everyone downloads their own copy.

To **look at the data**:
- Open `notebooks/01_eda.ipynb` → pick the `venv` kernel (top right) → run cells with `Shift+Enter`.
- Or open the CSV in `data/raw/` directly. The **Data Wrangler** or **Rainbow CSV** extensions make it readable as a table.

## 7. Run things (as they get built)

| What | Command |
|---|---|
| Tests | `pytest` |
| API | `uvicorn api.main:app --reload` → http://localhost:8000/docs |
| Dashboard | `streamlit run app/Home.py` → opens in your browser |

## 8. Daily workflow

```bash
git checkout main && git pull origin main      # get teammates' work
git checkout -b feature/your-thing             # new branch for your task
# ...work, save, test...
git add <files> && git commit -m "Describe what you did"
git push -u origin feature/your-thing          # then open a PR on GitHub
```
Then log what you did in [PROGRESS.md](../PROGRESS.md).
