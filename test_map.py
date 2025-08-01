#!/usr/bin/env python3
"""
Quick test script to verify map visualization imports and basic functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    print("Testing imports...")
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except Exception as e:
        print(f"❌ pandas import failed: {e}")
        return False
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        print("✅ plotly imported successfully")
    except Exception as e:
        print(f"❌ plotly import failed: {e}")
        return False
    
    try:
        import streamlit as st
        print("✅ streamlit imported successfully")
    except Exception as e:
        print(f"❌ streamlit import failed: {e}")
        return False
    
    try:
        from map_viz import display_map_dashboard
        print("✅ map_viz imported successfully")
    except Exception as e:
        print(f"❌ map_viz import failed: {e}")
        print(f"Error details: {e}")
        return False
    
    try:
        from fips_mapping import add_fips_to_dataframe
        print("✅ fips_mapping imported successfully")
    except Exception as e:
        print(f"❌ fips_mapping import failed: {e}")
        return False
    
    return True

def test_sample_data():
    print("\nTesting with sample data...")
    
    try:
        import pandas as pd
        
        # Create sample data similar to what the app would have
        sample_data = pd.DataFrame({
            'commodity_year': [2023] * 5,
            'commodity_code': [41] * 5,
            'commodity_name': ['CORN'] * 5,
            'state_abbrv': ['IA', 'IL', 'NE', 'MN', 'IN'],
            'county_name': ['STORY', 'MCLEAN', 'LANCASTER', 'BLUE EARTH', 'TIPPECANOE'],
            'total_liability': [1000000, 800000, 600000, 500000, 400000],
            'total_premium': [50000, 40000, 30000, 25000, 20000],
            'subsidy': [25000, 20000, 15000, 12500, 10000],
            'indemnity': [10000, 8000, 6000, 5000, 4000]
        })
        
        print(f"✅ Sample data created: {sample_data.shape}")
        print("Sample data columns:", list(sample_data.columns))
        
        return sample_data
        
    except Exception as e:
        print(f"❌ Failed to create sample data: {e}")
        return None

if __name__ == "__main__":
    print("🧪 Testing map visualization components...\n")
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Cannot proceed.")
        sys.exit(1)
    
    # Test sample data creation
    sample_df = test_sample_data()
    if sample_df is None:
        print("\n❌ Sample data test failed.")
        sys.exit(1)
    
    print("\n✅ All tests passed! Map visualization should work.")
