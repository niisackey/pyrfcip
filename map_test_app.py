import streamlit as st
from rfcip.summary import get_summary_data
from simple_working_map import display_simple_maps
import pandas as pd

st.set_page_config(page_title="Map Test", layout="wide")
st.title("🗺️ Simple Map Test")

# Basic inputs
crop_input = st.text_input("Crop:", "CORN")
state_input = st.text_input("State:", "IA") 
year = st.number_input("Year:", value=2024)

if st.button("Test Maps"):
    with st.spinner("Getting data..."):
        df = get_summary_data(crop_input, state_input, [year])
    
    if df is not None and not df.empty:
        st.success(f"Got {len(df)} rows")
        
        # Clean column names
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        st.dataframe(df.head())
        
        # Test the maps
        display_simple_maps(df)
    else:
        st.error("No data")
