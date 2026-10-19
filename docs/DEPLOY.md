# Deploying the dashboard (Streamlit Community Cloud)

Goal: a public URL for the live demo (required in the README and during the presentation).
Streamlit Community Cloud is free and deploys straight from GitHub.

## Already checked (2026-10-19)
We tested a **fresh copy of the repo exactly like the server sees it** (no dataset, no trained
model, no API key, only `requirements.txt` installed, Python 3.13):
- Install: ✅ ~20 seconds, no errors
- The app **downloads the dataset and trains the model by itself** on first start
- All 6 pages load with no errors, on real data (not demo data)
- No API key → AI explanations use the template (still work)

## Step by step
1. Go to **https://share.streamlit.io** and sign in **with GitHub**.
2. Click **Create app** → **Deploy a public app from GitHub** (Yes, from a private repo is fine. Streamlit will ask for permission to read it; the repo owner, Oba, may need to approve).
3. Fill in:
   | Field | Value |
   |---|---|
   | Repository | `obwoj1/ml-deploymate` |
   | Branch | `main` |
   | Main file path | `app/Home.py` |
   | App URL | e.g. `ml-deploymate` (gives `ml-deploymate.streamlit.app`) |
4. Click **Advanced settings**:
   - **Python version: 3.13** (the packages are pinned for 3.13)
   - **Secrets** (optional, only for real AI explanations; **Oba should type this in himself**):
     ```toml
     ANTHROPIC_API_KEY = "sk-ant-..."
     ```
     Use a key that belongs to a **workspace** (console.anthropic.com → Settings → API keys).
     An organization-level key also needs `ANTHROPIC_WORKSPACE_ID = "wrkspc_..."`.
     Never put the key in the repo or in a chat. Without it, explanations use the template.
5. Click **Deploy**. The first build takes 2–4 minutes. Then the first page load takes
   ~10–15 seconds (downloading the data + training the model); after that it's instant.
6. Click through all 5 pages. Run a scenario on Live Monitor and check the Alerts page.
7. Put the URL in the README under **🚀 Live Demo** and open a PR.

## Before the presentation ⚠️
- **Free apps go to sleep after a few days with no visitors.** Open the URL **10 minutes before
  presenting**. If you see "This app has gone to sleep", click **Yes, get this app back up** and wait.
- Every push to `main` redeploys automatically. **Don't merge anything right before presenting.**
- Backup plan if the internet fails: run it locally with `streamlit run app/Home.py`.

## If something goes wrong
- **"ModuleNotFoundError"** → a package is missing from `requirements.txt` (only that file is
  installed on the server, not `requirements-dev.txt`).
- **Page shows "Showing demo data"** → the real backend failed to start. Open
  **Manage app → logs** (bottom right) to see why. The most likely cause is that the dataset
  download failed; reboot the app.
- **Changes don't show up** → **Manage app → ⋮ → Reboot app**.
