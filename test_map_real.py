#!/usr/bin/env python3
"""
Test script to verify map functionality with real USDA data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_map_with_real_data():
    print("🧪 Testing map functionality with real USDA data...")
    
    try:
        # Import required modules
        import pandas as pd
        from rfcip.summary import get_summary_data_from_excel
        from map_viz import prepare_county_data, create_state_summary_map
        import plotly.express as px
        
        print("✅ All imports successful")
        
        # Get real data
        print("📊 Fetching real USDA data for 2023...")
        df = get_summary_data_from_excel(2023)
        
        if df.empty:
            print("❌ No data retrieved")
            return False
        
        print(f"✅ Data retrieved: {df.shape[0]:,} rows, {df.shape[1]} columns")
        print(f"Columns: {list(df.columns)}")
        
        # Test data preparation
        print("\n🛠️ Testing data preparation...")
        df_clean = prepare_county_data(df)
        
        if df_clean.empty:
            print("❌ Data preparation failed")
            return False
        
        print(f"✅ Data prepared: {df_clean.shape[0]:,} rows")
        print(f"Cleaned columns: {list(df_clean.columns)}")
        
        # Test state map creation
        print("\n🗺️ Testing state map creation...")
        try:
            fig = create_state_summary_map(
                df_clean,
                color_column='total_liability',
                title="Test State Map"
            )
            
            if fig is not None:
                print("✅ State map created successfully")
                # Try to save it as HTML to verify it works
                fig.write_html("test_map.html")
                print("✅ Map saved as test_map.html")
                return True
            else:
                print("❌ Map creation returned None")
                return False
                
        except Exception as e:
            print(f"❌ Map creation failed: {e}")
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_map_with_real_data()
    if success:
        print("\n🎉 All tests passed! Map functionality should work in Streamlit.")
    else:
        print("\n💥 Tests failed. Check the errors above.")
