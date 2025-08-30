import streamlit as st
import pandas as pd
import plotly.express as px

def simple_map_test(df):
    """
    Very simple map function for testing
    """
    st.write("🗺️ **Simple Map Test**")
    
    if df is None or df.empty:
        st.error("No data provided")
        return
    
    st.write(f"Data shape: {df.shape}")
    st.write(f"Columns: {list(df.columns)}")
    
    # Create a very simple state map
    try:
        # Aggregate by state if possible
        if 'state_abbrv' in df.columns and 'total_liability' in df.columns:
            state_data = df.groupby('state_abbrv')['total_liability'].sum().reset_index()
            
            # Create simple choropleth
            fig = px.choropleth(
                state_data,
                locations='state_abbrv',
                color='total_liability',
                locationmode='USA-states',
                scope='usa',
                title='Simple State Map Test'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.success("✅ Simple map displayed!")
        else:
            st.warning("Cannot create map - missing required columns")
            st.dataframe(df.head())
            
    except Exception as e:
        st.error(f"Map error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
