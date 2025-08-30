import streamlit as st
import plotly.express as px
import pandas as pd

st.title("🔧 Radio Button Debug Test")

# Test the radio button at the top level
st.write("Testing radio button functionality...")

viz_type = st.radio(
    "Select Visualization Type",
    ["Charts", "Maps"],
    horizontal=True
)

st.write(f"**Current selection: {viz_type}**")

if viz_type == "Maps":
    st.success("✅ MAPS SELECTED!")
    st.write("🗺️ Creating test map...")
    
    # Create simple test map
    test_data = pd.DataFrame({
        'state': ['IA', 'IL', 'CA', 'TX'],
        'value': [100, 90, 80, 70]
    })
    
    fig = px.choropleth(
        test_data,
        locations='state',
        color='value',
        locationmode='USA-states',
        scope='usa',
        title='TEST MAP - This should appear!'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.success("✅ MAP DISPLAYED!")
    
else:
    st.info("Charts selected - no map shown")
    st.write("Select 'Maps' to see a test map")
