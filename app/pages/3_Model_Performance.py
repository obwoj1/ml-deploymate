import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # lets the page find the `deploymate` code

import streamlit as st
from deploymate import service

st.title("Model Performance")
st.write("Coming soon")
