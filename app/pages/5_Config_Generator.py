import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # lets the page find the `deploymate` code

import streamlit as st

from deploymate import service

st.set_page_config(page_title="Config Generator · ML DeployMate", layout="wide")
st.title("Config Generator")
st.caption("Fill in a few details about your model and get the files you need to deploy it.")

with st.form("config"):
    c1, c2 = st.columns(2)
    model_name = c1.text_input("Model name", "my-model")
    framework = c2.radio("Serve it with", ["fastapi", "streamlit"], horizontal=True,
                         format_func=lambda f: "FastAPI (an API)" if f == "fastapi" else "Streamlit (a dashboard)")
    python_version = c1.selectbox("Python version", ["3.13", "3.12", "3.11"])
    port = c2.number_input("Port", min_value=1, max_value=65535,
                           value=8000 if framework == "fastapi" else 8501, step=1)
    submitted = st.form_submit_button("Generate files", type="primary")

if submitted:
    st.session_state["config_files"] = service.generate_config({
        "model_name": model_name, "python_version": python_version,
        "framework": framework, "port": int(port)})

files = st.session_state.get("config_files")
if files:
    FILES = [
        ("Dockerfile", "dockerfile", "docker", "Dockerfile",
         "Packages your model and its dependencies into a container that runs anywhere."),
        ("GitHub Actions", "github_actions", "yaml", "ci.yml",
         "Runs your tests automatically on every push. Save it as .github/workflows/ci.yml."),
        (".env.example", "env_example", "bash", ".env.example",
         "The settings your app needs. Copy it to .env and fill in real values (never commit .env)."),
    ]
    for tab, (title, key, lang, filename, about) in zip(st.tabs([f[0] for f in FILES]), FILES):
        with tab:
            st.caption(about)
            st.code(files[key], language=lang)
            st.download_button(f"⬇ Download {filename}", files[key], file_name=filename, key=f"dl-{key}")
else:
    st.info("Fill in the form and press **Generate files**.")
