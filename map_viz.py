"""
US County Map Visualization for USDA Crop Insurance Data
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from urllib.request import urlopen
import json
import numpy as np
from typing import Optional, Dict, Any
from fips_mapping import add_fips_to_dataframe

# Load US counties GeoJSON data
@st.cache_data
def load_counties_geojson():
    """Load US counties GeoJSON data from plotly datasets"""
    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        counties = json.load(response)
    return counties

def prepare_county_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare county data for visualization with robust error handling
    
    Args:
        df: Raw DataFrame from USDA data source
        
    Returns:
        Cleaned DataFrame ready for mapping
    """
    st.write("🔍 **DEBUG: prepare_county_data called**")
    
    if df is None:
        st.error("❌ Input DataFrame is None")
        return pd.DataFrame()
    
    if df.empty:
        st.error("❌ Input DataFrame is empty")
        return pd.DataFrame()
    
    st.write(f"✅ Input data: {df.shape[0]} rows, {df.shape[1]} columns")
    st.write(f"Input columns: {list(df.columns)}")
    
    try:
        # Make a copy to avoid modifying original
        result_df = df.copy()
        st.write("✅ DataFrame copied successfully")
        
        # Ensure basic required columns exist
        required_cols = ['total_liability', 'total_premium', 'indemnity']
        missing_cols = []
        for col in required_cols:
            if col not in result_df.columns:
                result_df[col] = 0.0
                missing_cols.append(col)
        
        if missing_cols:
            st.warning(f"⚠️ Added missing columns with zeros: {missing_cols}")
        
        # Ensure state abbreviation column exists
        if 'state_abbrv' not in result_df.columns:
            if 'state_abbreviation' in result_df.columns:
                result_df['state_abbrv'] = result_df['state_abbreviation']
                st.write("✅ Used state_abbreviation for state_abbrv")
            elif 'locationstateabbreviation' in result_df.columns:
                result_df['state_abbrv'] = result_df['locationstateabbreviation']
                st.write("✅ Used locationstateabbreviation for state_abbrv")
            else:
                result_df['state_abbrv'] = 'Unknown'
                st.warning("⚠️ No state column found, set to 'Unknown'")
        
        # Ensure county name column exists
        if 'county_name' not in result_df.columns:
            if 'county' in result_df.columns:
                result_df['county_name'] = result_df['county']
                st.write("✅ Used county for county_name")
            else:
                result_df['county_name'] = 'Unknown'
                st.warning("⚠️ No county column found, set to 'Unknown'")
        
        # Clean up data types
        for col in required_cols:
            old_type = result_df[col].dtype
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0.0)
            st.write(f"✅ Converted {col} from {old_type} to numeric")
        
        # Remove rows with all zero values for key metrics
        before_filter = len(result_df)
        result_df = result_df[
            (result_df['total_liability'] > 0) | 
            (result_df['total_premium'] > 0) | 
            (result_df['indemnity'] > 0)
        ]
        after_filter = len(result_df)
        st.write(f"✅ Filtered data: {before_filter} → {after_filter} rows")
        
        # Calculate derived metrics if possible
        if 'subsidy' not in result_df.columns:
            result_df['subsidy'] = 0.0
            st.write("✅ Added subsidy column with zeros")
        else:
            result_df['subsidy'] = pd.to_numeric(result_df['subsidy'], errors='coerce').fillna(0.0)
            st.write("✅ Cleaned subsidy column")
        
        st.write(f"✅ Final data: {result_df.shape[0]} rows, {result_df.shape[1]} columns")
        st.write(f"Final columns: {list(result_df.columns)}")
        
        return result_df
        
    except Exception as e:
        st.error(f"❌ Error preparing data: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        # Return original data if preparation fails
        return df

def create_county_choropleth(
    df: pd.DataFrame, 
    color_column: str = 'total_liability',
    title: str = "US County-Level Insurance Data",
    color_scale: str = 'Blues'
) -> go.Figure:
    """
    Create a choropleth map of US counties
    
    Args:
        df: DataFrame with county data including FIPS codes
        color_column: Column name to use for color mapping
        title: Map title
        color_scale: Plotly color scale name
        
    Returns:
        Plotly figure object
    """
    # Load counties GeoJSON
    counties = load_counties_geojson()
    
    # Create the choropleth map
    fig = px.choropleth(
        df,
        geojson=counties,
        locations='fips',
        color=color_column,
        color_continuous_scale=color_scale,
        scope="usa",
        hover_data={
            'county_name': True,
            'state_abbrv': True,
            'total_liability': ':,.0f',
            'total_premium': ':,.0f',
            'indemnity': ':,.0f',
            'fips': False
        },
        labels={
            'total_liability': 'Total Liability ($)',
            'total_premium': 'Total Premium ($)',
            'indemnity': 'Indemnity ($)',
            'county_name': 'County',
            'state_abbrv': 'State'
        },
        title=title
    )
    
    # Update layout
    fig.update_layout(
        title_x=0.5,
        geo=dict(
            showframe=False,
            showcoastlines=True,
            showland=True,
            landcolor='rgb(243, 243, 243)',
            projection_type='albers usa'
        ),
        width=1000,
        height=600
    )
    
    return fig

def create_state_summary_map(
    df: pd.DataFrame,
    color_column: str = 'total_liability',
    title: str = "US State-Level Insurance Summary"
) -> go.Figure:
    """
    Create a state-level summary map
    
    Args:
        df: DataFrame with county data
        color_column: Column to aggregate and color by
        title: Map title
        
    Returns:
        Plotly figure object
    """
    # Aggregate data by state
    if 'state_abbrv' not in df.columns:
        # Try to create state abbreviations from state names if available
        if 'state_name' in df.columns:
            state_mapping = {
                'ALABAMA': 'AL', 'ALASKA': 'AK', 'ARIZONA': 'AZ', 'ARKANSAS': 'AR', 'CALIFORNIA': 'CA',
                'COLORADO': 'CO', 'CONNECTICUT': 'CT', 'DELAWARE': 'DE', 'FLORIDA': 'FL', 'GEORGIA': 'GA',
                'HAWAII': 'HI', 'IDAHO': 'ID', 'ILLINOIS': 'IL', 'INDIANA': 'IN', 'IOWA': 'IA',
                'KANSAS': 'KS', 'KENTUCKY': 'KY', 'LOUISIANA': 'LA', 'MAINE': 'ME', 'MARYLAND': 'MD',
                'MASSACHUSETTS': 'MA', 'MICHIGAN': 'MI', 'MINNESOTA': 'MN', 'MISSISSIPPI': 'MS', 'MISSOURI': 'MO',
                'MONTANA': 'MT', 'NEBRASKA': 'NE', 'NEVADA': 'NV', 'NEW HAMPSHIRE': 'NH', 'NEW JERSEY': 'NJ',
                'NEW MEXICO': 'NM', 'NEW YORK': 'NY', 'NORTH CAROLINA': 'NC', 'NORTH DAKOTA': 'ND', 'OHIO': 'OH',
                'OKLAHOMA': 'OK', 'OREGON': 'OR', 'PENNSYLVANIA': 'PA', 'RHODE ISLAND': 'RI', 'SOUTH CAROLINA': 'SC',
                'SOUTH DAKOTA': 'SD', 'TENNESSEE': 'TN', 'TEXAS': 'TX', 'UTAH': 'UT', 'VERMONT': 'VT',
                'VIRGINIA': 'VA', 'WASHINGTON': 'WA', 'WEST VIRGINIA': 'WV', 'WISCONSIN': 'WI', 'WYOMING': 'WY'
            }
            df['state_abbrv'] = df['state_name'].str.upper().map(state_mapping)
        else:
            # Create sample data if no state information available
            df['state_abbrv'] = 'CA'  # Default to California for demonstration
    
    state_df = df.groupby('state_abbrv').agg({
        'total_liability': 'sum',
        'total_premium': 'sum',
        'indemnity': 'sum',
        'subsidy': 'sum' if 'subsidy' in df.columns else 'total_premium'
    }).reset_index()
    
    # Calculate derived metrics
    state_df['loss_ratio'] = np.where(
        state_df['total_premium'] > 0,
        state_df['indemnity'] / state_df['total_premium'],
        0
    )
    
    # Create choropleth map
    fig = px.choropleth(
        state_df,
        locations='state_abbrv',
        color=color_column,
        locationmode='USA-states',
        color_continuous_scale='Blues',
        scope="usa",
        hover_data={
            'total_liability': ':,.0f',
            'total_premium': ':,.0f',
            'indemnity': ':,.0f',
            'loss_ratio': ':.3f'
        },
        labels={
            'total_liability': 'Total Liability ($)',
            'total_premium': 'Total Premium ($)',
            'indemnity': 'Indemnity ($)',
            'loss_ratio': 'Loss Ratio',
            'state_abbrv': 'State'
        },
        title=title
    )
    
    fig.update_layout(
        title_x=0.5,
        geo=dict(
            showframe=False,
            showcoastlines=True,
            showland=True,
            landcolor='rgb(243, 243, 243)'
        ),
        width=1000,
        height=600
    )
    
    return fig

def create_bubble_map(
    df: pd.DataFrame,
    size_column: str = 'total_liability',
    color_column: str = 'loss_ratio',
    title: str = "US County Insurance Metrics"
) -> go.Figure:
    """
    Create a bubble map showing counties as circles
    
    Args:
        df: DataFrame with county data including lat/lon
        size_column: Column for bubble size
        color_column: Column for bubble color
        title: Map title
        
    Returns:
        Plotly figure object
    """
    # For demonstration, create sample lat/lon coordinates
    # In production, you'd want actual county centroids
    np.random.seed(42)
    df_map = df.copy()
    df_map['lat'] = np.random.uniform(25, 49, len(df))  # Continental US latitude range
    df_map['lon'] = np.random.uniform(-125, -66, len(df))  # Continental US longitude range
    
    # Ensure positive values for size
    df_map[size_column] = df_map[size_column].abs()
    
    # Create scatter mapbox
    fig = px.scatter_mapbox(
        df_map,
        lat='lat',
        lon='lon',
        size=size_column,
        color=color_column,
        hover_data={
            'county_name': True,
            'state_abbrv': True,
            'total_liability': ':,.0f',
            'total_premium': ':,.0f',
            'indemnity': ':,.0f'
        },
        color_continuous_scale='RdYlBu_r',
        size_max=15,
        zoom=3,
        mapbox_style='open-street-map',
        title=title
    )
    
    fig.update_layout(
        title_x=0.5,
        mapbox=dict(
            center=dict(lat=39.8283, lon=-98.5795),  # Geographic center of US
            zoom=3
        ),
        width=1000,
        height=600
    )
    
    return fig

def display_map_dashboard(df: pd.DataFrame):
    """
    Display a complete map dashboard with multiple visualization options
    
    Args:
        df: DataFrame with county-level insurance data
    """
    st.write("🔍 **DEBUG: display_map_dashboard called**")
    st.subheader("🗺️ Geographic Visualization Dashboard")
    
    # Basic data validation first
    if df is None:
        st.error("❌ DataFrame is None")
        return
    
    if df.empty:
        st.error("❌ DataFrame is empty")
        return
    
    st.success(f"✅ DataFrame received: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    # Show available columns for debugging
    st.write("**Available columns:**", list(df.columns))
    
    with st.expander("🔍 Sample Data"):
        st.dataframe(df.head(3))
    
    try:
        st.write("🛠️ **Preparing data...**")
        # Prepare data
        df_clean = prepare_county_data(df)
        
        if df_clean.empty:
            st.error("❌ No data available after cleaning/preparation")
            return
        
        st.success(f"✅ Data prepared: {df_clean.shape[0]:,} rows")
        
        # Check if we have real county data or just "Unknown" counties
        has_real_counties = (
            'county_name' in df_clean.columns and 
            not df_clean['county_name'].isin(['Unknown', 'None', '']).all()
        )
        
        st.write(f"**Has real counties:** {has_real_counties}")
        
        # Map type selection based on available data
        if has_real_counties:
            map_options = ["County Choropleth", "State Summary", "Bubble Map"]
            default_map = "State Summary"  # Changed default to more reliable option
            st.info("ℹ️ County data detected. All map types available.")
        else:
            map_options = ["State Summary", "Bubble Map"]
            default_map = "State Summary"
            st.info("ℹ️ County data shows 'Unknown' values. Showing state-level visualizations.")
        
        map_type = st.selectbox(
            "Select Map Type",
            map_options,
            index=map_options.index(default_map)
        )
        
        st.write(f"**Selected map type:** {map_type}")
        
        # Metric selection
        metric_options = {
            'total_liability': 'Total Liability ($)',
            'total_premium': 'Total Premium ($)',
            'indemnity': 'Indemnity ($)',
            'subsidy': 'Subsidy ($)'
        }
        
        # Only include metrics that exist in the data
        available_metrics = {k: v for k, v in metric_options.items() if k in df_clean.columns}
        
        st.write(f"**Available metrics:** {list(available_metrics.keys())}")
        
        if not available_metrics:
            st.error("❌ No suitable metrics found for mapping")
            st.write("Looking for columns:", list(metric_options.keys()))
            st.write("Available columns:", list(df_clean.columns))
            return
        
        selected_metric = st.selectbox(
            "Select Metric to Display",
            options=list(available_metrics.keys()),
            format_func=lambda x: available_metrics[x]
        )
        
        st.write(f"**Selected metric:** {selected_metric}")
        
        # Generate and display map
        with st.spinner("Generating map..."):
            st.write(f"🗺️ **Creating {map_type} map...**")
            try:
                if map_type == "State Summary":
                    fig = create_state_summary_map(
                        df_clean,
                        color_column=selected_metric,
                        title=f"US States: {available_metrics[selected_metric]}"
                    )
                    st.write("✅ State map created")
                else:
                    st.write(f"⚠️ Map type {map_type} not implemented in debug mode")
                    fig = None
                
                if fig is not None:
                    st.write("� **Displaying map...**")
                    st.plotly_chart(fig, use_container_width=True)
                    st.success("✅ Map displayed successfully!")
                else:
                    st.error("❌ Failed to generate map figure")
                        
            except Exception as map_error:
                st.error(f"❌ Error generating {map_type}: {str(map_error)}")
                import traceback
                st.code(traceback.format_exc())
                    
    except Exception as e:
        st.error(f"❌ Critical error in map dashboard: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def display_summary_stats(df_clean, selected_metric, available_metrics, has_real_counties):
    """Display summary statistics for the selected metric"""
    try:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if has_real_counties:
                st.metric("Total Counties", len(df_clean))
            else:
                st.metric("Total Records", len(df_clean))
        
        with col2:
            if selected_metric in df_clean.columns:
                total_val = df_clean[selected_metric].sum()
                if total_val > 1000000:
                    st.metric(
                        f"Total {available_metrics[selected_metric]}", 
                        f"${total_val/1000000:.1f}M"
                    )
                else:
                    st.metric(
                        f"Total {available_metrics[selected_metric]}", 
                        f"${total_val:,.0f}"
                    )
        
        with col3:
            if selected_metric in df_clean.columns:
                avg_val = df_clean[selected_metric].mean()
                if avg_val > 1000000:
                    st.metric(
                        f"Average {available_metrics[selected_metric]}", 
                        f"${avg_val/1000000:.1f}M"
                    )
                else:
                    st.metric(
                        f"Average {available_metrics[selected_metric]}", 
                        f"${avg_val:,.0f}"
                    )
        
        with col4:
            if selected_metric in df_clean.columns:
                max_val = df_clean[selected_metric].max()
                if max_val > 1000000:
                    st.metric(
                        f"Max {available_metrics[selected_metric]}", 
                        f"${max_val/1000000:.1f}M"
                    )
                else:
                    st.metric(
                        f"Max {available_metrics[selected_metric]}", 
                        f"${max_val:,.0f}"
                    )
    except Exception as e:
        st.warning(f"Could not display statistics: {str(e)}")

# Example usage function
def demo_map_with_sample_data():
    """Demo function to show map with sample data"""
    # Create sample data
    sample_data = {
        'county_name': ['Cook County', 'Los Angeles County', 'Harris County', 'Maricopa County'],
        'state_abbrv': ['IL', 'CA', 'TX', 'AZ'],
        'total_liability': [1000000, 2000000, 1500000, 800000],
        'total_premium': [100000, 200000, 150000, 80000],
        'indemnity': [50000, 250000, 100000, 40000],
        'subsidy': [30000, 60000, 45000, 24000]
    }
    
    df_sample = pd.DataFrame(sample_data)
    display_map_dashboard(df_sample)

if __name__ == "__main__":
    demo_map_with_sample_data()
