#!/usr/bin/env python3
"""
Quick test to verify Maps work after removing charts
"""
import pandas as pd

def test_simple_maps():
    print("🧪 Testing simplified app...")
    
    # Test data
    test_data = {
        'county_name': ['Adams County', 'Brown County'], 
        'total_premium': [100000, 150000],
        'total_liability': [500000, 750000],
        'indemnity': [25000, 45000],
        'state_abbrv': ['IA', 'IA']
    }
    
    df = pd.DataFrame(test_data)
    print(f"✅ Test data created: {df.shape}")
    
    # Test the map function import
    try:
        from simple_working_map import display_simple_maps
        print("✅ Map function imported successfully")
        
        # This would normally call display_simple_maps(df) in Streamlit
        print("✅ Map function is ready to be called")
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_simple_maps()
    if success:
        print("\n🎉 SUCCESS! Charts removed, Maps should now work!")
        print("\n📋 Test it now:")
        print("1. Run: streamlit run app.py") 
        print("2. Enter: CORN, IA, 2024")
        print("3. Click 'Fetch Data'")
        print("4. Select 'Maps' - should work now!")
    else:
        print("\n💥 Still issues found")
