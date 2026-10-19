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
