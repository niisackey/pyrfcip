import streamlit as st
from rfcip.summary import get_summary_data
from rfcip.col import get_col_data
from rfcip.livestock import get_livestock_data
from rfcip.codes import get_crop_codes, get_insurance_plan_codes, get_price_data
from rfcip.helpers import valid_state, valid_crop
from rfcip.reinsurance_reports import build_reinsurance_datasets
from map_viz import display_map_dashboard
from simple_working_map import display_simple_maps
from guaranteed_maps import guaranteed_working_maps
from debug_maps_simple import debug_maps
from datetime import datetime
import pandas as pd
import altair as alt
import numpy as np

st.set_page_config(page_title="USDA Crop Insurance Explorer", layout="wide")
st.title("🌽 USDA Crop Insurance Explorer")

# --- Sidebar controls ---
st.sidebar.header("🔍 Query Parameters")

data_type = st.sidebar.selectbox("Select Data Type", [
    "Summary of Business",
    "County-Level Loss",
    "Livestock Insurance",
    "Price Discovery Data",
    "Reinsurance Reports"
])

# Year range - current year back to 2000
current_year = datetime.now().year
year_selection = st.sidebar.multiselect("Select Year(s)", list(range(current_year, 1999, -1)), default=[current_year])
year = year_selection if year_selection else [current_year]


# Dynamic inputs based on data type
if data_type == "Livestock Insurance":
    program = st.sidebar.selectbox("Select Program", ["DRP", "LGM", "LRP"])
    crop_input = ""
else:
    program = None
    crop_input = st.sidebar.text_input("Enter Crop Name (e.g., CORN or FEEDER CATTLE):", "CORN")

state_input = st.sidebar.text_input("Enter State (Name, Abbrev, or FIPS):", "IA")

# Preload reinsurance data when selected
reinsurance_data_loaded = False
reinsurance_df = None
fund_col = None
report_col = None

if data_type == "Reinsurance Reports":
    with st.spinner("Loading reinsurance options..."):
        try:
            base_url = "https://www.rma.usda.gov/tools-reports/reinsurance-reports"
            df_dict = build_reinsurance_datasets(base_url)
            reinsurance_df = pd.concat(df_dict.values(), ignore_index=True)
            reinsurance_data_loaded = True
            
            # Detect appropriate columns for filtering
            possible_fund_cols = ['fund', 'fund_name', 'fund name', 'funds', 'fundtype', 'fund type']
            possible_report_cols = ['report', 'report_type', 'report name', 'reporttype', 'type', 'report category']
            
            for col in reinsurance_df.columns:
                col_lower = col.lower()
                if not fund_col and any(term in col_lower for term in possible_fund_cols):
                    fund_col = col
                if not report_col and any(term in col_lower for term in possible_report_cols):
                    report_col = col
                    
        except Exception as e:
            st.error(f"Error loading reinsurance options: {str(e)}")
            reinsurance_data_loaded = False

# Initialize filters
fund_filter = []
report_type_filter = []

# Only show reinsurance filters if we found appropriate columns
if reinsurance_data_loaded and reinsurance_df is not None and not reinsurance_df.empty:
    if fund_col:
        fund_options = reinsurance_df[fund_col].unique().tolist()
        fund_filter = st.sidebar.multiselect(f"Filter by {fund_col}", fund_options)
    
    if report_col:
        report_options = reinsurance_df[report_col].unique().tolist()
        report_type_filter = st.sidebar.multiselect(f"Filter by {report_col}", report_options)

# --- Main content ---
# Initialize session state for data persistence
if 'df' not in st.session_state:
    st.session_state.df = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

df = st.session_state.df

if st.sidebar.button("Fetch Data"):
    # Validate inputs before fetching data
    validation_failed = False

    if state_input and not valid_state(state_input):
        st.error("❌ Invalid state entered. Please enter a valid state name, abbreviation, or FIPS code.")
        validation_failed = True

    if data_type in ["Summary of Business", "Price Discovery Data"] and crop_input:
        if not valid_crop(crop_input):
            st.error("❌ Invalid crop name entered. Please enter a valid crop name.")
            validation_failed = True

    if data_type == "County-Level Loss" and crop_input:
        if not valid_crop(crop_input):
            st.error("❌ Invalid crop name entered. Please enter a valid crop name.")
            validation_failed = True

    if validation_failed:
        st.stop()

    with st.spinner("Fetching data..."):
        try:
            if data_type == "Summary of Business":
                df = get_summary_data(crop_input, state_input, year)
                if df is not None and not df.empty:
                    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
                    # Store in session state
                    st.session_state.df = df
                    st.session_state.data_loaded = True


            elif data_type == "County-Level Loss":
                df = get_col_data(year=year)
                if crop_input:
                    df = df[df['commodity_name'].str.upper() == crop_input.upper()]
                if state_input:
                    state_upper = state_input.upper()
                    if 'state_abbrv' in df.columns:
                        df = df[df['state_abbrv'] == state_upper]
                    elif 'state_code' in df.columns:
                        try:
                            state_code = int(state_upper)
                            df = df[df['state_code'] == state_code]
                        except ValueError:
                            df = df[df['state_code'].astype(str) == state_upper]
                # Store in session state
                st.session_state.df = df
                st.session_state.data_loaded = True

            elif data_type == "Livestock Insurance":
                df = get_livestock_data(year=year, program=program)
                if state_input:
                    if 'location_state_abbreviation' in df.columns:
                        # FIXED: Removed extra bracket at the end of this line
                        df = df[df['location_state_abbreviation'] == state_input.upper()]
                # Store in session state
                st.session_state.df = df
                st.session_state.data_loaded = True

            elif data_type == "Price Discovery Data":
                df = get_price_data(year=year, crop=crop_input, state=state_input)
                # Store in session state
                st.session_state.df = df
                st.session_state.data_loaded = True

            elif data_type == "Reinsurance Reports":
                # Use preloaded data and apply filters
                df = reinsurance_df.copy()
                if fund_col and fund_filter:
                    df = df[df[fund_col].isin(fund_filter)]
                if report_col and report_type_filter:
                    df = df[df[report_col].isin(report_type_filter)]
                
                # Year filtering doesn't apply to reinsurance reports
                st.info("ℹ️ Year selection is not applicable for Reinsurance Reports")
                # Store in session state
                st.session_state.df = df
                st.session_state.data_loaded = True

        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")

# Display data if it exists in session state (moved outside the button click)
df = st.session_state.df  # Get the current data from session state
if df is None or df.empty:
    if st.session_state.data_loaded:
        st.warning("⚠️ No data found for the given inputs.")
else:
    st.success(f"✅ Loaded {len(df)} rows.")
    st.dataframe(df, use_container_width=True)

    # ======================
    # VISUALIZATIONS SECTION
    # ======================
    st.subheader("📊 Data Visualizations")
    
    # Add visualization type selector
    viz_type = st.radio(
        "Select Visualization Type",
        ["Charts", "Maps"],
        horizontal=True
    )
    
    if viz_type == "Maps":
        st.subheader("🗺️ Geographic Visualizations")
        
        # Check if we have geographic data for mapping
        if data_type == "Summary of Business":
            # For Summary of Business, we need state-level data
            if 'state_code' in df.columns or 'state_name' in df.columns or 'state_abbrv' in df.columns:
                # Create state-level aggregations
                if 'state_code' in df.columns:
                    state_col = 'state_code'
                elif 'state_abbrv' in df.columns:
                    state_col = 'state_abbrv'
                else:
                    state_col = 'state_name'
                
                # Aggregate data by state
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    map_data = df.groupby(state_col)[numeric_cols].sum().reset_index()
                    
                    # Create map visualization options
                    metric_options = [col for col in numeric_cols if col not in [state_col]]
                    if metric_options:
                        selected_metric = st.selectbox("Select metric to visualize:", metric_options)
                        
                        # Create the choropleth map
                        try:
                            import plotly.express as px
                            
                            fig = px.choropleth(
                                map_data,
                                locations=state_col,
                                color=selected_metric,
                                locationmode='USA-states',
                                scope='usa',
                                title=f'{selected_metric.replace("_", " ").title()} by State',
                                color_continuous_scale="Viridis"
                            )
                            
                            fig.update_layout(
                                geo=dict(
                                    showframe=False,
                                    showcoastlines=True,
                                    showland=True,
                                    landcolor="rgb(243, 243, 243)"
                                )
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show data table for the map
                            with st.expander("📊 Map Data Details"):
                                st.dataframe(map_data, use_container_width=True)
                                
                        except Exception as e:
                            st.error(f"Error creating map: {str(e)}")
                            st.info("📊 Showing data table instead:")
                            st.dataframe(map_data, use_container_width=True)
                    else:
                        st.warning("No numeric columns available for mapping.")
                else:
                    st.warning("No numeric data available for state-level mapping.")
            else:
                st.warning("No state information found in the data for geographic mapping.")
                
        elif data_type == "County-Level Loss":
            # For County-Level Loss, we have county-level data
            if 'county_name' in df.columns and 'state_abbrv' in df.columns:
                st.info("🗺️ County-level mapping available - showing state-level summary")
                
                # Aggregate to state level for mapping
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    state_data = df.groupby('state_abbrv')[numeric_cols].sum().reset_index()
                    
                    metric_options = [col for col in numeric_cols]
                    if metric_options:
                        selected_metric = st.selectbox("Select metric to visualize:", metric_options)
                        
                        try:
                            import plotly.express as px
                            
                            fig = px.choropleth(
                                state_data,
                                locations='state_abbrv',
                                color=selected_metric,
                                locationmode='USA-states',
                                scope='usa',
                                title=f'{selected_metric.replace("_", " ").title()} by State (County Data Aggregated)',
                                color_continuous_scale="Reds"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            with st.expander("� State Summary Data"):
                                st.dataframe(state_data, use_container_width=True)
                                
                        except Exception as e:
                            st.error(f"Error creating map: {str(e)}")
            else:
                st.warning("No geographic information found for mapping.")
        else:
            st.info(f"�️ Geographic mapping not yet implemented for {data_type}")
            st.info("💡 Available for: Summary of Business, County-Level Loss")
    
    else:  # Charts
        st.subheader("📊 Chart Visualizations")
        
        # Get numeric columns for charting
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(numeric_cols) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("� Summary Statistics")
                # Show key metrics
                for col in numeric_cols[:5]:  # Show top 5 numeric columns
                    if col in df.columns:
                        total_val = df[col].sum()
                        avg_val = df[col].mean()
                        st.metric(
                            label=col.replace('_', ' ').title(),
                            value=f"${total_val:,.0f}" if 'premium' in col.lower() or 'liability' in col.lower() or 'indemnity' in col.lower() else f"{total_val:,.0f}",
                            delta=f"Avg: ${avg_val:,.0f}" if 'premium' in col.lower() or 'liability' in col.lower() or 'indemnity' in col.lower() else f"Avg: {avg_val:,.0f}"
                        )
            
            with col2:
                st.subheader("📊 Data Distribution")
                # Create a simple bar chart
                if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                    group_col = st.selectbox("Group by:", categorical_cols)
                    value_col = st.selectbox("Sum by:", numeric_cols)
                    
                    if group_col and value_col:
                        chart_data = df.groupby(group_col)[value_col].sum().reset_index()
                        chart_data = chart_data.sort_values(value_col, ascending=False).head(10)
                        
                        # Create Altair chart
                        chart = alt.Chart(chart_data).mark_bar().encode(
                            x=alt.X(f'{value_col}:Q', title=value_col.replace('_', ' ').title()),
                            y=alt.Y(f'{group_col}:N', sort='-x', title=group_col.replace('_', ' ').title()),
                            color=alt.Color(f'{value_col}:Q', scale=alt.Scale(scheme='viridis'))
                        ).properties(
                            width=400,
                            height=300,
                            title=f'{value_col.replace("_", " ").title()} by {group_col.replace("_", " ").title()}'
                        )
                        
                        st.altair_chart(chart, use_container_width=True)
            
            # Additional charts row
            st.subheader("📈 Trend Analysis")
            
            # Time series if we have date columns
            date_cols = [col for col in df.columns if 'year' in col.lower() or 'date' in col.lower()]
            if date_cols and len(numeric_cols) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    date_col = st.selectbox("Time dimension:", date_cols)
                    metric_col = st.selectbox("Metric:", numeric_cols, key="trend_metric")
                    
                with col2:
                    if date_col and metric_col:
                        trend_data = df.groupby(date_col)[metric_col].sum().reset_index()
                        
                        trend_chart = alt.Chart(trend_data).mark_line(point=True).encode(
                            x=alt.X(f'{date_col}:O', title=date_col.replace('_', ' ').title()),
                            y=alt.Y(f'{metric_col}:Q', title=metric_col.replace('_', ' ').title()),
                            tooltip=[date_col, metric_col]
                        ).properties(
                            width=400,
                            height=200,
                            title=f'{metric_col.replace("_", " ").title()} Over Time'
                        )
                        
                        st.altair_chart(trend_chart, use_container_width=True)
            
            # Correlation heatmap for numeric columns
            if len(numeric_cols) > 2:
                st.subheader("🔗 Correlation Analysis")
                corr_cols = st.multiselect("Select columns for correlation:", numeric_cols, default=numeric_cols[:5])
                
                if len(corr_cols) > 1:
                    corr_data = df[corr_cols].corr()
                    
                    # Create correlation heatmap using Altair
                    corr_df = corr_data.reset_index().melt('index')
                    corr_df.columns = ['var1', 'var2', 'correlation']
                    
                    heatmap = alt.Chart(corr_df).mark_rect().encode(
                        x=alt.X('var1:N', title=None),
                        y=alt.Y('var2:N', title=None),
                        color=alt.Color('correlation:Q', scale=alt.Scale(scheme='redblue', domain=[-1, 1])),
                        tooltip=['var1', 'var2', 'correlation']
                    ).properties(
                        width=400,
                        height=400,
                        title='Correlation Matrix'
                    )
                    
                    st.altair_chart(heatmap, use_container_width=True)
        else:
            st.warning("No numeric columns found for chart visualization.")

    # Download button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"usda_{data_type.replace(' ', '_')}_{state_input}.csv",
        mime='text/csv'
    )

# --- Codes Reference ---
st.markdown("---")
with st.expander("📘 Reference Tables"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌱 Crop Codes")
        crop_codes_df = get_crop_codes(year=max(year))
        st.dataframe(crop_codes_df, use_container_width=True)

        st.subheader("🐄 Livestock Programs")
        livestock_programs = pd.DataFrame({
            "Program": ["DRP", "LGM", "LRP"],
            "Full Name": [
                "Dairy Revenue Protection",
                "Livestock Gross Margin",
                "Livestock Risk Protection"
            ]
        })
        st.dataframe(livestock_programs, use_container_width=True)

    with col2:
        st.subheader("📋 Insurance Plan Codes")
        plan_codes_df = get_insurance_plan_codes(year=max(year))
        st.dataframe(plan_codes_df, use_container_width=True)

        st.subheader("📊 Cause of Loss Codes")
        col_codes = pd.DataFrame({
            "Code": list(range(1, 15)),
            "Description": [
                "Drought", "Excess Moisture/Precipitation/Rain", "Freeze", "Hail",
                "Hurricane/Typhoon", "Failure of Irrigation Water Supply", "Fire",
                "Insect Damage", "Plant Disease", "Other", "Volcanic Eruption",
                "Decline in Price", "Earthquake", "Wind"
            ]
        })
        st.dataframe(col_codes, use_container_width=True)
                        # DEBUG VERSION - This should definitely work
                        st.write("� **DEBUG: Maps selected!**")
                        debug_maps(df)
                    else:
                        # Charts removed - only showing basic data summary
                        st.write("📊 **Data Summary**")
                        st.write(f"Total rows: {len(df):,}")
                        
                        # Show basic statistics
                        if 'total_premium' in df.columns:
                            total_premium = df['total_premium'].sum()
                            st.metric("Total Premium", f"${total_premium:,.2f}")
                        
                        if 'total_liability' in df.columns:
                            total_liability = df['total_liability'].sum()
                            st.metric("Total Liability", f"${total_liability:,.2f}")
                        
                        if 'indemnity' in df.columns:
                            total_indemnity = df['indemnity'].sum()
                            st.metric("Total Indemnity", f"${total_indemnity:,.2f}")
                else:
                    missing = [col for col in required_cols if col not in df.columns]
                    st.warning(f"⚠️ Cannot generate visualizations. Missing columns: {', '.join(missing)}. Available columns: {', '.join(df.columns)}")
            
            elif data_type == "County-Level Loss":
                if viz_type == "Maps":
                    # Display guaranteed working maps
                    st.write("🗺️ **Loading County Loss Maps...**")
                    guaranteed_working_maps(df)
                else:
                    # Simple summary instead of complex charts
                    st.write("📊 **County-Level Loss Summary**")
                    st.write(f"Total rows: {len(df):,}")
                    if 'indem_amount' in df.columns:
                        total_indemnity = df['indem_amount'].sum()
                        st.metric("Total Indemnity", f"${total_indemnity:,.2f}")
                    if 'total_premium' in df.columns:
                        total_premium = df['total_premium'].sum()
                        st.metric("Total Premium", f"${total_premium:,.2f}")
            
            elif data_type == "Livestock Insurance":
                if viz_type == "Maps":
                    # Display guaranteed working maps
                    st.write("🗺️ **Loading Livestock Maps...**")
                    guaranteed_working_maps(df)
                else:
                    # Simple summary for livestock
                    st.write("📊 **Livestock Insurance Summary**")
                    st.write(f"Total rows: {len(df):,}")
                    if 'total_premium' in df.columns:
                        total_premium = df['total_premium'].sum()
                        st.metric("Total Premium", f"${total_premium:,.2f}")
                    if 'total_liability' in df.columns:
                        total_liability = df['total_liability'].sum()
                        st.metric("Total Liability", f"${total_liability:,.2f}")
            
            elif data_type == "Price Discovery Data":
                if viz_type == "Maps":
                    # Display guaranteed working maps
                    st.write("🗺️ **Loading Price Discovery Maps...**")
                    guaranteed_working_maps(df)
                else:
                    # Simple summary for price data
                    st.write("📊 **Price Discovery Summary**")
                    st.write(f"Total rows: {len(df):,}")
                    if 'price' in df.columns:
                        avg_price = df['price'].mean()
                        st.metric("Average Price", f"${avg_price:.2f}")
            
            elif data_type == "Reinsurance Reports":
                if viz_type == "Maps":
                    # Display guaranteed working maps
                    st.write("🗺️ **Loading Reinsurance Maps...**")
                    guaranteed_working_maps(df)
                else:
                    # Simple summary for reinsurance
                    st.write("📊 **Reinsurance Reports Summary**")
                    st.write(f"Total rows: {len(df):,}")
                    
                    # Find dollar column
                    dollar_col = None
                    possible_dollar_cols = ['dollars', 'amount', 'value', 'dollar amount', 'fund amount', 'total']
                    for col in df.columns:
                        col_lower = col.lower()
                        if any(term in col_lower for term in possible_dollar_cols):
                            dollar_col = col
                            break
                    
                    if dollar_col:
                        total_amount = df[dollar_col].sum()
                        st.metric(f"Total {dollar_col}", f"${total_amount:,.2f}")
        
        except Exception as e:
            st.error(f"Visualization error: {str(e)}")

        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"usda_{data_type.replace(' ', '_')}_{state_input}.csv",
            mime='text/csv'
        )

# --- Codes Reference ---
st.markdown("---")
with st.expander("📘 Reference Tables"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌱 Crop Codes")
        crop_codes_df = get_crop_codes(year=max(year))
        st.dataframe(crop_codes_df, use_container_width=True)

        st.subheader("🐄 Livestock Programs")
        livestock_programs = pd.DataFrame({
            "Program": ["DRP", "LGM", "LRP"],
            "Full Name": [
                "Dairy Revenue Protection",
                "Livestock Gross Margin",
                "Livestock Risk Protection"
            ]
        })
        st.dataframe(livestock_programs, use_container_width=True)

    with col2:
        st.subheader("📋 Insurance Plan Codes")
        plan_codes_df = get_insurance_plan_codes(year=max(year))
        st.dataframe(plan_codes_df, use_container_width=True)

        st.subheader("📊 Cause of Loss Codes")
        col_codes = pd.DataFrame({
            "Code": list(range(1, 15)),
            "Description": [
                "Drought", "Excess Moisture/Precipitation/Rain", "Freeze", "Hail",
                "Hurricane/Typhoon", "Failure of Irrigation Water Supply", "Fire",
                "Insect Damage", "Plant Disease", "Other", "Volcanic Eruption",
                "Decline in Price", "Earthquake", "Wind"
            ]
        })
        st.dataframe(col_codes, use_container_width=True)