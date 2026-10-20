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

**Steps to fix:**
1. Open the application logs.
2. Identify the missing module and check the installation output.
3. Confirm that its package is listed in `requirements.txt`.
   The deployment installs this file, not `requirements-dev.txt`.
4. If a dependency is missing, add a compatible version.
5. Commit the correction and open a pull request.
6. After the change is merged and deployed, confirm the app starts.

### The app shows "Showing demo data"

**Symptom:** The application loads but displays a demo-data banner.

**Possible cause:** The real backend could not initialize. A failed
dataset download is one possible reason.

**Steps to fix:**
1. Open Manage app and inspect the logs.
2. Find the error associated with backend initialization.
3. If the logs indicate a temporary download failure, reboot the app.
4. Wait for initialization to finish.
5. Confirm that the demo-data banner disappears.
6. Run a monitoring scenario to check that the backend works.

Demo data keeps the interface available, but it should not be described
as a successful run of the real backend.

### UI changes do not appear

**Symptom:** The hosted application still shows an older version.

**Steps to investigate:**
1. Confirm the updated file was saved and committed.
2. Confirm the commit was pushed to GitHub.
3. Confirm the pull request was merged into `main`.
4. Check the deployment logs for errors.
5. Wait for deployment to finish, then refresh the browser.
6. If the app still shows old content, use the app management controls
   to reboot it.

Changes on a separate branch will not appear in the hosted app while
the deployment is configured to use `main`.

### AI explanations use a template

**Symptom:** Alerts receive template explanations instead of
AI-generated explanations.

**Possible cause:** No API key is configured, or the AI request failed.

**Steps to investigate:**
1. Confirm whether an API key was intentionally configured.
2. If AI explanations are required, ask the key owner to verify the
   deployment's secret settings.
3. Check the logs for provider errors.
4. Request an explanation again after resolving the reported issue.

Template explanations are an expected fallback. Monitoring can still
work without an AI API key.

Never include secret values in commits, screenshots, or troubleshooting notes.

### Local presentation backup

If the hosted application is unavailable:

1. Open a terminal in the project folder.
2. Activate the project's virtual environment.
3. Run:

   streamlit run app/Home.py

4. Open the local address printed in the terminal.
5. Check the pages needed for the demonstration.

Prepare and test this backup before the presentation. The first local
startup may need internet access to download the dataset and install
dependencies.

