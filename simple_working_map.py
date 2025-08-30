"""
Simplified map visualization that WILL work
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def simple_working_map(df):
    """
    Create a simple map that definitely works
    """
    st.write("🗺️ **Simple Working Map**")
    
    if df is None or df.empty:
        st.error("No data provided")
        return
    
    # Show what we have
    st.write(f"Data shape: {df.shape}")
    
    try:
        # Create a basic state-level map
        if 'state_abbrv' in df.columns:
            # Aggregate by state
            state_cols = ['state_abbrv']
            
            # Find numeric columns to aggregate
            numeric_cols = []
            for col in ['total_premium', 'total_liability', 'indemnity']:
                if col in df.columns:
                    numeric_cols.append(col)
            
            if numeric_cols:
                agg_dict = {col: 'sum' for col in numeric_cols}
                state_data = df.groupby('state_abbrv').agg(agg_dict).reset_index()
                
                # Create map with first numeric column
                metric = numeric_cols[0]
                
                fig = px.choropleth(
                    state_data,
                    locations='state_abbrv',
                    color=metric,
                    locationmode='USA-states',
                    scope='usa',
                    title=f'{metric.replace("_", " ").title()} by State',
                    color_continuous_scale='Viridis'
                )
                
                fig.update_layout(
                    geo=dict(
                        showlakes=True,
                        lakecolor='rgb(255, 255, 255)'
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.success("✅ Map displayed successfully!")
                
                # Show the data used
                with st.expander("📊 Data used for map"):
                    st.dataframe(state_data)
            else:
                st.warning("No numeric columns found for mapping")
        else:
            st.warning("No state_abbrv column found for mapping")
            # Fallback: show a sample map
            st.write("Creating sample US map...")
            
            # Create a simple sample map
            sample_states = ['IA', 'IL', 'IN', 'OH', 'NE']
            sample_values = [100, 150, 120, 180, 90]
            
            sample_df = pd.DataFrame({
                'state': sample_states,
                'value': sample_values
            })
            
            fig = px.choropleth(
                sample_df,
                locations='state',
                color='value',
                locationmode='USA-states',
                scope='usa',
                title='Sample Map (using test data)',
                color_continuous_scale='Blues'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.info("ℹ️ Showing sample map since no state data found")
            
    except Exception as e:
        st.error(f"Map error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

def display_simple_maps(df):
    """
    Display simple, reliable maps
    """
    st.subheader("🗺️ Geographic Visualizations")
    
    # Just create the simplest possible working map
    simple_working_map(df)
