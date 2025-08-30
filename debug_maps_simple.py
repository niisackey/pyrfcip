"""
Ultra simple debug map for immediate testing
"""
import streamlit as st
import plotly.express as px
import pandas as pd

def debug_maps(df):
    """
    Debug version that shows exactly what's happening
    """
    st.write("🔍 **DEBUG: Map function called!**")
    st.write("🔍 **DEBUG: Function is running!**")
    
    # Always show a map no matter what
    st.write("🗺️ **Creating test map...**")
    
    # Create a simple test map that always works
    test_data = pd.DataFrame({
        'state': ['IA', 'IL', 'CA', 'TX', 'FL'],
        'value': [100, 90, 80, 70, 60]
    })
    
    fig = px.choropleth(
        test_data,
        locations='state',
        color='value',
        locationmode='USA-states',
        scope='usa',
        title='DEBUG: This map should ALWAYS appear',
        color_continuous_scale='Blues'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.success("✅ DEBUG: If you see this, maps are working!")
    
    # Show your data info
    if df is not None:
        st.write("🔍 **Your data info:**")
        st.write(f"Shape: {df.shape}")
        st.write(f"Columns: {list(df.columns)}")
    else:
        st.write("🔍 **No data provided to map function**")
