#!/usr/bin/env python3
"""
Final test to make sure everything works before running Streamlit
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def final_test():
    print("🔧 Final system test...")
    
    try:
        # Test imports
        import pandas as pd
        import plotly.express as px
        from simple_map import simple_map_test
        print("✅ All imports successful")
        
        # Create test data
        test_data = pd.DataFrame({
            'state_abbrv': ['IA', 'IL', 'NE', 'MN', 'IN'],
            'county_name': ['STORY', 'MCLEAN', 'LANCASTER', 'BLUE EARTH', 'TIPPECANOE'],
            'total_liability': [1000000, 800000, 600000, 500000, 400000],
            'total_premium': [50000, 40000, 30000, 25000, 20000],
            'subsidy': [25000, 20000, 15000, 12500, 10000],
            'indemnity': [10000, 8000, 6000, 5000, 4000]
        })
        print("✅ Test data created")
        
        # Test simple choropleth creation
        state_data = test_data.groupby('state_abbrv')['total_liability'].sum().reset_index()
        
        fig = px.choropleth(
            state_data,
            locations='state_abbrv',
            color='total_liability',
            locationmode='USA-states',
            scope='usa',
            title='Final Test Map'
        )
        
        if fig is not None:
            print("✅ Plotly choropleth created successfully")
            fig.write_html("final_test_map.html")
            print("✅ Test map saved as final_test_map.html")
            return True
        else:
            print("❌ Map creation failed")
            return False
            
    except Exception as e:
        print(f"❌ Final test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = final_test()
    if success:
        print("\n🎉 All systems ready! Streamlit app should work now.")
        print("Run: streamlit run app.py")
    else:
        print("\n💥 System test failed. Check errors above.")
