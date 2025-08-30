#!/usr/bin/env python3
"""
Final verification that Streamlit app will run properly
"""
import streamlit as st
import pandas as pd
from rfcip.summary import get_summary_data

def main():
    st.title("🧪 Map Functionality Test")
    
    # Test data
    with st.spinner("Getting test data..."):
        df = get_summary_data("CORN", "IA", [2024])
    
    if df is not None and not df.empty:
        st.success(f"✅ Got {len(df)} rows of data")
        
        # Visualization type selector
        viz_type = st.radio(
            "Select Visualization Type",
            ["Charts", "Maps"],
            horizontal=True
        )
        
        if viz_type == "Maps":
            # Test maps functionality
            from map_viz import display_map_dashboard
            from simple_map import simple_map_test
            
            st.write("🗺️ **Loading Map Visualizations...**")
            try:
                display_map_dashboard(df)
                st.success("✅ Maps working correctly!")
            except Exception as e:
                st.error(f"❌ Map error: {str(e)}")
                st.write("🔧 **Trying simple map fallback...**")
                try:
                    simple_map_test(df)
                    st.success("✅ Simple map fallback working!")
                except Exception as e2:
                    st.error(f"❌ Simple map also failed: {str(e2)}")
        else:
            st.write("📊 Charts would be displayed here")
    else:
        st.error("❌ No test data available")

if __name__ == "__main__":
    main()
