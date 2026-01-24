"""Quick test to verify tab structure"""
import streamlit as st

st.title("Tab Test")

tab1, tab2, tab3, tab4 = st.tabs(["Tab 1", "Tab 2", "Tab 3", "Tab 4"])

with tab2:
    st.write("This is tab 2")
    st.success("Tab 2 works!")

with tab3:
    st.write("This is tab 3")
    st.success("Tab 3 works!")

with tab4:
    st.write("This is tab 4")
    st.success("Tab 4 works!")

with tab1:
    st.write("This is tab 1")
    st.success("Tab 1 works!")
