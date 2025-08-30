#!/usr/bin/env python3
"""
Minimal radio button test to verify debug messages
"""
import streamlit as st

st.title("🔍 Radio Button Debug Test")

# Test radio button
viz_type = st.radio(
    "Select Visualization Type",
    ["Charts", "Maps"],
    horizontal=True
)

# Debug message - this should ALWAYS show
st.write(f"🔍 **RADIO DEBUG: Selected = {viz_type}**")

if viz_type == "Maps":
    st.error("🔍 **MAPS BRANCH ENTERED!**")
    st.success("Maps radio button is working!")
elif viz_type == "Charts":
    st.info("🔍 **CHARTS BRANCH ENTERED!**")
    st.success("Charts radio button is working!")

st.write("---")
st.write("**Test Results:**")
st.write("1. You should see the radio debug message above")
st.write("2. When you click Maps, you should see 'MAPS BRANCH ENTERED!'")
st.write("3. When you click Charts, you should see 'CHARTS BRANCH ENTERED!'")
