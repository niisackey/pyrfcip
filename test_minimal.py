#!/usr/bin/env python3
"""
Minimal test to reproduce the maps issue
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_minimal_map():
    print("🧪 Testing minimal map functionality...")
    
    try:
        # Import required modules
        import pandas as pd
        from rfcip.summary import get_summary_data_from_excel
        
        print("✅ Imports successful")
        
        # Get real data (use cached data if available)
        print("📊 Fetching USDA data...")
        df = get_summary_data_from_excel(2023)
        
        if df.empty:
            print("❌ No data retrieved")
            return False
        
        print(f"✅ Data retrieved: {df.shape[0]:,} rows, {df.shape[1]} columns")
        print(f"Columns: {list(df.columns)}")
        
        # Test if display_map_dashboard can be imported
        print("🗺️ Testing map dashboard import...")
        try:
            from map_viz import display_map_dashboard
            print("✅ Map dashboard imported successfully")
        except Exception as e:
            print(f"❌ Map dashboard import failed: {e}")
            return False
        
        # Test if we can call prepare_county_data directly
        print("🛠️ Testing data preparation...")
        try:
            from map_viz import prepare_county_data
            df_clean = prepare_county_data(df)
            print(f"✅ Data preparation successful: {df_clean.shape}")
        except Exception as e:
            print(f"❌ Data preparation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test state map creation
        print("🗺️ Testing state map...")
        try:
            from map_viz import create_state_summary_map
            fig = create_state_summary_map(
                df_clean,
                color_column='total_liability',
                title="Test State Map"
            )
            
            if fig is not None:
                print("✅ State map created successfully")
                return True
            else:
                print("❌ State map returned None")
                return False
                
        except Exception as e:
            print(f"❌ State map creation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_minimal_map()
    if success:
        print("\n🎉 Minimal test passed!")
    else:
        print("\n💥 Minimal test failed!")
