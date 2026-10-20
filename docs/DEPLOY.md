# Deployment (Streamlit Community Cloud)

The dashboard is deployed at https://ml-deploymate-2tgkbhmailycfxbjgpvyys.streamlit.app

## Requirements
- The app installs only `requirements.txt` (pinned, runtime-only) on Python 3.13.
- No dataset or trained model needs to be committed: on first start the app downloads the UCI Adult dataset and trains the model automatically (about 10–15 seconds), then caches both in memory.
- An API key is optional. Without one, alert explanations use the built-in template.

This was verified on a fresh clone with no data, no model and no key: installation completed in about 20 seconds and every page loaded on real data.

## Steps
1. Sign in at https://share.streamlit.io with GitHub. If the repository is private, grant Streamlit access to private repositories.
2. Select **Create app** → **Deploy a public app from GitHub**.
3. Settings:

   | Field | Value |
   |---|---|
   | Repository | `obwoj1/ml-deploymate` |
   | Branch | `main` |
   | Main file path | `app/Home.py` |

4. Under **Advanced settings**, set the Python version to **3.13** and, optionally, add secrets:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
   Use a workspace-scoped key. An organization-level key also requires `ANTHROPIC_WORKSPACE_ID`. Enter secrets only in the Streamlit settings; never commit them.
5. Select **Deploy**. The first build takes a few minutes.

Secrets can be added or changed later under **Settings → Secrets**. The Python version can only be changed by redeploying.

## Operational notes
- Every merge into `main` triggers an automatic redeploy.
- Apps on the free tier sleep after a period of inactivity; the first visit afterwards restarts the app. Open the app shortly before a demo.
- The theme is read from `.streamlit/config.toml` in the repository root. Deploying from a branch with a different config changes the theme.
- Local fallback: `streamlit run app/Home.py` from the project root.

## Troubleshooting
| Symptom | Likely cause |
|---|---|
| `ModuleNotFoundError` in the logs | A package is missing from `requirements.txt` |
| "Showing demo data" banner | The real backend failed to start (usually the dataset download); check **Manage app → logs** and reboot |
| "Template explanation" on the Alerts page | No valid API key in Secrets |
| Changes not visible | **Manage app → Reboot app** |

## Troubleshooting in detail

### `ModuleNotFoundError` in the logs

**Symptom:** The app fails to start and the logs show `ModuleNotFoundError`.

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

   ```bash
   streamlit run app/Home.py
   ```

4. Open the local address printed in the terminal.
5. Check the pages needed for the demonstration.

Prepare and test this backup before the presentation. The first local
startup may need internet access to download the dataset and install
dependencies.
