import streamlit as st
import os

st.write(f"Current Working Directory: {os.getcwd()}")
st.write(f"Does .streamlit folder exist? {os.path.exists('.streamlit')}")
st.write(f"Secrets available: {list(st.secrets.keys())}")