"""
Ultra-simple map that ALWAYS works
"""
import streamlit as st
import pandas as pd
import plotly.express as px

def ultra_simple_map():
    """
    Create the simplest possible US map that always works
    """
    st.write("🗺️ **Ultra Simple Map Test**")
    
    # Create guaranteed data that will always work
    sample_data = pd.DataFrame({
        'state': ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI'],
        'value': [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    })
    
    st.write("Creating basic US map...")
    
    try:
        # Create the simplest possible choropleth map
        fig = px.choropleth(
            sample_data,
            locations='state',
            color='value',
            locationmode='USA-states',
            scope='usa',
            title='Basic US Map Test'
        )
        
        # Display the map
        st.plotly_chart(fig, use_container_width=True)
        st.success("✅ Basic map displayed!")
        
        return True
        
    except Exception as e:
        st.error(f"❌ Even simple map failed: {str(e)}")
        return False

def guaranteed_working_maps(df):
    """
    Maps that are guaranteed to work
    """
    st.subheader("🗺️ Map Test")
    
    # First, try the ultra simple map
    st.write("**Step 1: Testing basic map capability...**")
    basic_works = ultra_simple_map()
    
    if not basic_works:
        st.error("❌ Basic mapping capability failed")
        return
    
    # If basic works, try with your data
    st.write("**Step 2: Trying with your data...**")
    
    if df is None or df.empty:
        st.warning("⚠️ No data provided, using sample only")
        return
    
    st.write(f"Data shape: {df.shape}")
    st.write(f"Columns: {list(df.columns)}")
    
    # Try to create a map with your data
    try:
        if 'state_abbrv' in df.columns:
            # Show first few rows
            st.write("Sample data:")
            st.dataframe(df.head(3))
            
            # Try to aggregate by state
            if 'total_premium' in df.columns:
                state_data = df.groupby('state_abbrv')['total_premium'].sum().reset_index()
                
                fig = px.choropleth(
                    state_data,
                    locations='state_abbrv',
                    color='total_premium',
                    locationmode='USA-states',
                    scope='usa',
                    title='Premium by State (Your Data)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.success("✅ Your data map displayed!")
            else:
                st.warning("No total_premium column found")
        else:
            st.warning("No state_abbrv column found")
            
    except Exception as e:
        st.error(f"❌ Your data map failed: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
