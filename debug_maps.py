#!/usr/bin/env python3
"""
Debug version to see what happens when Maps is selected
"""
import streamlit as st
import pandas as pd

# Create some test data that looks like real USDA data
test_data = {
    'county_name': ['Adams County', 'Brown County', 'Clay County', 'Davis County'],
    'total_premium': [100000, 150000, 120000, 80000],
    'total_liability': [500000, 750000, 600000, 400000],
    'indemnity': [25000, 45000, 30000, 15000],
    'state_abbrv': ['IA', 'IA', 'IA', 'IA'],
    'fips': ['19001', '19017', '19035', '19051']
}

df = pd.DataFrame(test_data)

st.title("🔧 DEBUG: Map Selection Issue")
st.write("This is a debug version to see exactly what happens when Maps is selected")

st.write("**Test Data:**")
st.dataframe(df)

# Test the radio button with detailed debugging
st.write("---")
st.subheader("🎛️ Visualization Type Selection")

viz_type = st.radio(
    "Select Visualization Type",
    ["Charts", "Maps"],
    horizontal=True,
    key="viz_type_debug"
)

st.write(f"**Current selection:** `{viz_type}`")

# Add debugging info
if viz_type == "Charts":
    st.success("✅ Charts selected - this should work normally")
    st.bar_chart(df.set_index('county_name')['total_premium'])
    
elif viz_type == "Maps":
    st.warning("🗺️ Maps selected - let's see what happens...")
    
    # Step-by-step debugging
    st.write("**Step 1: Checking imports...**")
    try:
        st.write("Importing map_viz...")
        from map_viz import display_map_dashboard
        st.success("✅ map_viz.display_map_dashboard imported")
        
        st.write("Importing simple_map...")
        from simple_map import simple_map_test
        st.success("✅ simple_map.simple_map_test imported")
        
    except Exception as e:
        st.error(f"❌ Import failed: {str(e)}")
        st.stop()
    
    st.write("**Step 2: Calling map functions...**")
    
    # Try the simple map first
    with st.expander("🔍 Simple Map Test", expanded=True):
        try:
            st.write("Calling simple_map_test(df)...")
            simple_map_test(df)
            st.success("✅ Simple map completed")
        except Exception as e:
            st.error(f"❌ Simple map failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    # Try the full dashboard
    with st.expander("🗺️ Full Map Dashboard Test", expanded=True):
        try:
            st.write("Calling display_map_dashboard(df)...")
            display_map_dashboard(df)
            st.success("✅ Full map dashboard completed")
        except Exception as e:
            st.error(f"❌ Full map dashboard failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

st.write("---")
st.write("**End of debug script. Did you see any maps above?**")
