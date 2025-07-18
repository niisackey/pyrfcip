import streamlit as st
from rfcip.summary import get_summary_data
from rfcip.col import get_col_data
from rfcip.livestock import get_livestock_data
from rfcip.codes import get_crop_codes, get_insurance_plan_codes, get_price_data
from rfcip.helpers import valid_state, valid_crop
from rfcip.reinsurance_reports import build_reinsurance_datasets
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
df = None
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

            elif data_type == "Livestock Insurance":
                df = get_livestock_data(year=year, program=program)
                if state_input:
                    if 'location_state_abbreviation' in df.columns:
                        # FIXED: Removed extra bracket at the end of this line
                        df = df[df['location_state_abbreviation'] == state_input.upper()]

            elif data_type == "Price Discovery Data":
                df = get_price_data(year=year, crop=crop_input, state=state_input)

            elif data_type == "Reinsurance Reports":
                # Use preloaded data and apply filters
                df = reinsurance_df.copy()
                if fund_col and fund_filter:
                    df = df[df[fund_col].isin(fund_filter)]
                if report_col and report_type_filter:
                    df = df[df[report_col].isin(report_type_filter)]
                
                # Year filtering doesn't apply to reinsurance reports
                st.info("ℹ️ Year selection is not applicable for Reinsurance Reports")

        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")

    if df is None or df.empty:
        st.warning("⚠️ No data found for the given inputs.")
    else:
        st.success(f"✅ Loaded {len(df)} rows.")
        st.dataframe(df, use_container_width=True)

        # ======================
        # VISUALIZATIONS SECTION
        # ======================
        st.subheader("📊 Data Visualizations")
        
        try:
            if data_type == "Summary of Business":
                # Check if we have required columns
                required_cols = ['county_name', 'total_premium', 'total_liability', 'indemnity']
                if all(col in df.columns for col in required_cols):
                    # Aggregate data by county
                    county_df = df.groupby('county_name').agg({
                        'total_premium': 'sum',
                        'total_liability': 'sum',
                        'indemnity': 'sum'
                    }).reset_index()
                    
                    # Calculate loss ratio
                    county_df['loss_ratio'] = county_df['indemnity'] / county_df['total_premium']
                    county_df['loss_ratio'] = county_df['loss_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
                    
                    # Sort and get top 10 counties
                    top_counties = county_df.sort_values('total_premium', ascending=False).head(10)
                    
                    # Create tabs for different visualizations
                    tab1, tab2, tab3 = st.tabs(["Premium by County", "Loss Ratio by County", "Liability vs Premium"])
                    
                    with tab1:
                        st.write("**Top 10 Counties by Premium**")
                        # Bar chart - Premium by County
                        chart = alt.Chart(top_counties).mark_bar().encode(
                            x=alt.X('county_name:N', sort='-y', title='County'),
                            y=alt.Y('total_premium:Q', title='Total Premium ($)'),
                            tooltip=['county_name', 'total_premium']
                        ).properties(
                            width=600,
                            height=400
                        )
                        st.altair_chart(chart, use_container_width=True)
                    
                    with tab2:
                        st.write("**Loss Ratio by County**")
                        # Bar chart - Loss Ratio by County
                        loss_chart = alt.Chart(top_counties).mark_bar().encode(
                            x=alt.X('county_name:N', sort='-y', title='County'),
                            y=alt.Y('loss_ratio:Q', title='Loss Ratio (Indemnity/Premium)'),
                            color=alt.Color('loss_ratio:Q', scale=alt.Scale(scheme='redblue')),
                            tooltip=['county_name', 'loss_ratio', 'indemnity', 'total_premium']
                        ).properties(
                            width=600,
                            height=400
                        )
                        st.altair_chart(loss_chart, use_container_width=True)
                    
                    with tab3:
                        st.write("**Liability vs Premium**")
                        # Scatter plot - Liability vs Premium
                        scatter = alt.Chart(top_counties).mark_circle(size=60).encode(
                            x='total_liability:Q',
                            y='total_premium:Q',
                            color=alt.Color('loss_ratio:Q', scale=alt.Scale(scheme='goldred')),
                            tooltip=['county_name', 'total_liability', 'total_premium', 'loss_ratio']
                        ).properties(
                            width=600,
                            height=400
                        )
                        st.altair_chart(scatter, use_container_width=True)
                else:
                    missing = [col for col in required_cols if col not in df.columns]
                    st.warning(f"⚠️ Cannot generate visualizations. Missing columns: {', '.join(missing)}. Available columns: {', '.join(df.columns)}")
            
            elif data_type == "County-Level Loss":
                # Create tabs
                tab1, tab2, tab3 = st.tabs(["Premium & Indemnity", "Loss Causes", "Geographic Distribution"])
                
                with tab1:
                    # Premium and Indemnity over time
                    if 'year_of_loss' in df.columns:
                        # Group by year and sum
                        year_df = df.groupby('year_of_loss').agg({
                            'total_premium': 'sum',
                            'indem_amount': 'sum'
                        }).reset_index()
                        
                        # Create a melted dataframe for plotting
                        melted_df = year_df.melt(id_vars=['year_of_loss'], 
                                                value_vars=['total_premium', 'indem_amount'],
                                                var_name='metric', 
                                                value_name='value')
                        
                        # Create a bar chart
                        bar_chart = alt.Chart(melted_df).mark_bar().encode(
                            x='year_of_loss:O',
                            y=alt.Y('value:Q', title='Amount ($)'),
                            color='metric:N',
                            tooltip=['year_of_loss', 'metric', 'value']
                        ).properties(
                            title='Premium vs Indemnity Over Time',
                            width=600,
                            height=400
                        )
                        st.altair_chart(bar_chart, use_container_width=True)
                    else:
                        st.warning("⚠️ 'year_of_loss' column not found in data")
                
                with tab2:
                    # Loss causes analysis
                    if 'col_name' in df.columns and 'indem_amount' in df.columns:
                        # Group by cause of loss
                        col_df = df.groupby('col_name').agg({
                            'indem_amount': 'sum',
                            'policies_indemnified': 'sum'
                        }).reset_index().sort_values('indem_amount', ascending=False).head(10)
                        
                        # Create a bar chart for top loss causes
                        cause_chart = alt.Chart(col_df).mark_bar().encode(
                            x=alt.X('col_name:N', title='Cause of Loss', sort='-y'),
                            y=alt.Y('indem_amount:Q', title='Total Indemnity ($)'),
                            tooltip=['col_name', 'indem_amount', 'policies_indemnified']
                        ).properties(
                            title='Top 10 Causes of Loss by Indemnity Amount',
                            width=600,
                            height=400
                        )
                        st.altair_chart(cause_chart, use_container_width=True)
                    else:
                        st.warning("⚠️ Required columns for loss causes analysis not found")
                
                with tab3:
                    # Geographic distribution
                    if 'county_name' in df.columns and 'indem_amount' in df.columns:
                        # Group by county
                        county_df = df.groupby('county_name').agg({
                            'indem_amount': 'sum',
                            'total_premium': 'sum'
                        }).reset_index().sort_values('indem_amount', ascending=False).head(10)
                        
                        # Calculate loss ratio
                        county_df['loss_ratio'] = county_df['indem_amount'] / county_df['total_premium']
                        county_df['loss_ratio'] = county_df['loss_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
                        
                        # Create bar chart for top counties by indemnity
                        county_chart = alt.Chart(county_df).mark_bar().encode(
                            x=alt.X('county_name:N', title='County', sort='-y'),
                            y=alt.Y('indem_amount:Q', title='Total Indemnity ($)'),
                            color=alt.Color('loss_ratio:Q', scale=alt.Scale(scheme='redblue')),
                            tooltip=['county_name', 'indem_amount', 'total_premium', 'loss_ratio']
                        ).properties(
                            title='Top 10 Counties by Indemnity Amount',
                            width=600,
                            height=400
                        )
                        st.altair_chart(county_chart, use_container_width=True)
                    else:
                        st.warning("⚠️ Required columns for geographic analysis not found")
            
            elif data_type == "Livestock Insurance":
                if 'year' in df.columns:
                    # Create tabs
                    tab1, tab2 = st.tabs(["Yearly Trends", "Program Distribution"])
                    
                    with tab1:
                        # Group by year and sum
                        year_df = df.groupby('year').sum().reset_index()
                        
                        # Line chart for trends
                        metrics = st.multiselect(
                            'Select metrics to plot', 
                            options=['total_premium', 'total_liability', 'indemnity'],
                            default=['total_premium', 'indemnity'],
                            key='livestock_metrics'
                        )
                        
                        if metrics:
                            melted_df = year_df.melt(id_vars=['year'], 
                                                    value_vars=metrics,
                                                    var_name='metric', 
                                                    value_name='value')
                            
                            line_chart = alt.Chart(melted_df).mark_line(point=True).encode(
                                x='year:O',
                                y='value:Q',
                                color='metric:N',
                                tooltip=['year', 'metric', 'value']
                            ).properties(
                                title=f'Livestock Insurance Trends ({program})',
                                width=600,
                                height=400
                            )
                            st.altair_chart(line_chart, use_container_width=True)
                    
                    with tab2:
                        # Distribution by state
                        if 'location_state_abbreviation' in df.columns:
                            state_df = df.groupby('location_state_abbreviation').sum().reset_index()
                            
                            bar_chart = alt.Chart(state_df).mark_bar().encode(
                                x=alt.X('location_state_abbreviation:N', title='State'),
                                y=alt.Y('total_premium:Q', title='Total Premium'),
                                color=alt.Color('location_state_abbreviation:N', legend=None),
                                tooltip=['location_state_abbreviation', 'total_premium']
                            ).properties(
                                title='Livestock Premium by State',
                                width=600,
                                height=400
                            )
                            st.altair_chart(bar_chart, use_container_width=True)
            
            elif data_type == "Price Discovery Data":
                if 'price' in df.columns and 'date' in df.columns:
                    # Convert date to datetime
                    df['date'] = pd.to_datetime(df['date'])
                    
                    # Line chart for price trends
                    line_chart = alt.Chart(df).mark_line().encode(
                        x=alt.X('date:T', title='Date'),
                        y=alt.Y('price:Q', title='Price ($)'),
                        tooltip=['date', 'price']
                    ).properties(
                        title=f'Price Discovery for {crop_input}',
                        width=600,
                        height=400
                    )
                    st.altair_chart(line_chart, use_container_width=True)
            
            elif data_type == "Reinsurance Reports":
                # Create tabs
                tab1, tab2 = st.tabs(["Dollars by Year", "Fund Distribution"])
                
                with tab1:
                    # Find dollar column if it exists
                    dollar_col = None
                    possible_dollar_cols = ['dollars', 'amount', 'value', 'dollar amount', 'fund amount', 'total']
                    for col in df.columns:
                        col_lower = col.lower()
                        if any(term in col_lower for term in possible_dollar_cols):
                            dollar_col = col
                            break
                    
                    # Find year column if it exists
                    year_col = None
                    possible_year_cols = ['year', 'reinsurance_year', 'report_year', 'fiscal_year', 'date_year', 'reporting_year']
                    for col in df.columns:
                        col_lower = col.lower()
                        if any(term in col_lower for term in possible_year_cols):
                            year_col = col
                            break
                    
                    if dollar_col:
                        if year_col:
                            # Group by year and sum
                            year_df = df.groupby(year_col)[dollar_col].sum().reset_index()
                            
                            chart = alt.Chart(year_df).mark_bar().encode(
                                x=f'{year_col}:O',
                                y=f'sum({dollar_col}):Q',
                                tooltip=[year_col, f'sum({dollar_col})']
                            ).properties(
                                width=800, 
                                height=400, 
                                title=f"{dollar_col} by {year_col}"
                            )
                            st.altair_chart(chart, use_container_width=True)
                        else:
                            # Just show total dollars if no year column found
                            total = df[dollar_col].sum()
                            st.metric(f"Total {dollar_col}", f"${total:,.2f}")
                    else:
                        st.warning("⚠️ Could not find dollar amount column for visualization")
                
                with tab2:
                    if fund_col:
                        if dollar_col:
                            fund_df = df.groupby(fund_col)[dollar_col].sum().reset_index()
                            
                            # Show bar chart instead of pie for better readability
                            bar_chart = alt.Chart(fund_df).mark_bar().encode(
                                x=alt.X(f'{fund_col}:N', sort='-y', title=fund_col),
                                y=alt.Y(f'sum({dollar_col}):Q', title=dollar_col),
                                color=alt.Color(f'{fund_col}:N', legend=None),
                                tooltip=[fund_col, f'sum({dollar_col})']
                            ).properties(
                                title=f"{dollar_col} by {fund_col}",
                                width=600,
                                height=400
                            )
                            st.altair_chart(bar_chart, use_container_width=True)
                        else:
                            # Show fund distribution without dollar amounts
                            fund_counts = df[fund_col].value_counts().reset_index()
                            fund_counts.columns = [fund_col, 'Count']
                            
                            chart = alt.Chart(fund_counts).mark_bar().encode(
                                x=f'{fund_col}:N',
                                y='Count:Q',
                                color=alt.Color(f'{fund_col}:N', legend=None)
                            ).properties(
                                title=f"Distribution by {fund_col}",
                                width=600,
                                height=400
                            )
                            st.altair_chart(chart, use_container_width=True)
                    else:
                        st.warning("⚠️ Could not find fund column for visualization")
        
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