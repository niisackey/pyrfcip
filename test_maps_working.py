#!/usr/bin/env python3
"""
Test script to verify Maps functionality is working
"""
import pandas as pd
from rfcip.summary import get_summary_data
from map_viz import display_map_dashboard
from simple_map import simple_map_test

def test_maps():
    """Test the maps functionality"""
    print("🧪 Testing Maps Functionality...")
    
    # Get some test data
    print("📊 Getting test data...")
    try:
        df = get_summary_data("CORN", "IA", [2024])
        if df is None or df.empty:
            print("❌ No test data available")
            return False
        
        print(f"✅ Got {len(df)} rows of data")
        print(f"Columns: {list(df.columns)}")
        
        # Test simple map
        print("\n🗺️ Testing simple map...")
        simple_map_test(df)
        print("✅ Simple map test completed")
        
        # Test full map dashboard
        print("\n🗺️ Testing full map dashboard...")
        display_map_dashboard(df)
        print("✅ Full map dashboard test completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during test: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_maps()
    if success:
        print("\n🎉 All map tests passed!")
    else:
        print("\n💥 Some map tests failed!")
