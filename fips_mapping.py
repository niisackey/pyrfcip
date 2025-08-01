"""
US County FIPS Codes mapping
This file provides mapping between county names, state abbreviations, and FIPS codes
"""
import pandas as pd
import requests
import streamlit as st
from typing import Dict, Optional

# Sample FIPS data - in production, you'd want the complete dataset
SAMPLE_FIPS_DATA = {
    'fips': [
        '01001', '01003', '01005', '01007', '01009', '01011', '01013', '01015', '01017', '01019',
        '02013', '02016', '02020', '02050', '02060', '02068', '02070', '02090', '02100', '02110',
        '04001', '04003', '04005', '04007', '04009', '04011', '04012', '04013', '04015', '04017',
        '05001', '05003', '05005', '05007', '05009', '05011', '05013', '05015', '05017', '05019',
        '06001', '06003', '06005', '06007', '06009', '06011', '06013', '06015', '06017', '06019',
        '17031', '17043', '17097', '17111', '17197',  # Illinois counties including Cook
        '48001', '48003', '48005', '48007', '48009', '48011', '48013', '48015', '48017', '48019',
        '19001', '19003', '19005', '19007', '19009', '19011', '19013', '19015', '19017', '19019'  # Iowa counties
    ],
    'county_name': [
        'Autauga County', 'Baldwin County', 'Barbour County', 'Bibb County', 'Blount County',
        'Bullock County', 'Butler County', 'Calhoun County', 'Chambers County', 'Cherokee County',
        'Aleutians East Borough', 'Aleutians West Census Area', 'Anchorage Municipality', 
        'Bethel Census Area', 'Bristol Bay Borough', 'Denali Borough', 'Dillingham Census Area',
        'Fairbanks North Star Borough', 'Haines Borough', 'Juneau City and Borough',
        'Apache County', 'Cochise County', 'Coconino County', 'Gila County', 'Graham County',
        'Greenlee County', 'La Paz County', 'Maricopa County', 'Mohave County', 'Navajo County',
        'Arkansas County', 'Ashley County', 'Baxter County', 'Benton County', 'Boone County',
        'Bradley County', 'Calhoun County', 'Carroll County', 'Chicot County', 'Clark County',
        'Alameda County', 'Alpine County', 'Amador County', 'Butte County', 'Calaveras County',
        'Colusa County', 'Contra Costa County', 'Del Norte County', 'El Dorado County', 'Fresno County',
        'Cook County', 'DuPage County', 'Lake County', 'McHenry County', 'Will County',
        'Anderson County', 'Andrews County', 'Angelina County', 'Aransas County', 'Archer County',
        'Armstrong County', 'Atascosa County', 'Austin County', 'Bailey County', 'Bandera County',
        'Adair County', 'Adams County', 'Allamakee County', 'Appanoose County', 'Audubon County',
        'Benton County', 'Black Hawk County', 'Boone County', 'Bremer County', 'Buchanan County'
    ],
    'state_name': [
        'Alabama', 'Alabama', 'Alabama', 'Alabama', 'Alabama', 'Alabama', 'Alabama', 'Alabama', 'Alabama', 'Alabama',
        'Alaska', 'Alaska', 'Alaska', 'Alaska', 'Alaska', 'Alaska', 'Alaska', 'Alaska', 'Alaska', 'Alaska',
        'Arizona', 'Arizona', 'Arizona', 'Arizona', 'Arizona', 'Arizona', 'Arizona', 'Arizona', 'Arizona', 'Arizona',
        'Arkansas', 'Arkansas', 'Arkansas', 'Arkansas', 'Arkansas', 'Arkansas', 'Arkansas', 'Arkansas', 'Arkansas', 'Arkansas',
        'California', 'California', 'California', 'California', 'California', 'California', 'California', 'California', 'California', 'California',
        'Illinois', 'Illinois', 'Illinois', 'Illinois', 'Illinois',
        'Texas', 'Texas', 'Texas', 'Texas', 'Texas', 'Texas', 'Texas', 'Texas', 'Texas', 'Texas',
        'Iowa', 'Iowa', 'Iowa', 'Iowa', 'Iowa', 'Iowa', 'Iowa', 'Iowa', 'Iowa', 'Iowa'
    ],
    'state_abbrv': [
        'AL', 'AL', 'AL', 'AL', 'AL', 'AL', 'AL', 'AL', 'AL', 'AL',
        'AK', 'AK', 'AK', 'AK', 'AK', 'AK', 'AK', 'AK', 'AK', 'AK',
        'AZ', 'AZ', 'AZ', 'AZ', 'AZ', 'AZ', 'AZ', 'AZ', 'AZ', 'AZ',
        'AR', 'AR', 'AR', 'AR', 'AR', 'AR', 'AR', 'AR', 'AR', 'AR',
        'CA', 'CA', 'CA', 'CA', 'CA', 'CA', 'CA', 'CA', 'CA', 'CA',
        'IL', 'IL', 'IL', 'IL', 'IL',
        'TX', 'TX', 'TX', 'TX', 'TX', 'TX', 'TX', 'TX', 'TX', 'TX',
        'IA', 'IA', 'IA', 'IA', 'IA', 'IA', 'IA', 'IA', 'IA', 'IA'
    ]
}

@st.cache_data
def get_fips_mapping() -> pd.DataFrame:
    """
    Get FIPS codes mapping dataframe
    
    Returns:
        DataFrame with fips, county_name, state_name, state_abbrv columns
    """
    return pd.DataFrame(SAMPLE_FIPS_DATA)

@st.cache_data
def load_full_fips_data() -> Optional[pd.DataFrame]:
    """
    Attempt to load full FIPS data from external source
    Falls back to sample data if external source is unavailable
    
    Returns:
        DataFrame with complete FIPS mapping or None if failed
    """
    try:
        # Try to load from a reliable external source
        url = "https://raw.githubusercontent.com/kjhealy/fips-codes/master/state_and_county_fips_master.csv"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            # Read the CSV content
            import io
            df = pd.read_csv(io.StringIO(response.text))
            
            # Standardize column names
            df.columns = df.columns.str.lower().str.strip()
            
            # Map to our expected column names
            column_mapping = {
                'fips': 'fips',
                'name': 'county_name',
                'state': 'state_name',
                'state_abbrv': 'state_abbrv'
            }
            
            # Try different possible column name variations
            for old_col, new_col in column_mapping.items():
                if old_col not in df.columns:
                    # Try common variations
                    alternatives = {
                        'fips': ['fips_code', 'county_fips', 'geoid'],
                        'name': ['county_name', 'county', 'area_name'],
                        'state': ['state_name', 'state_full'],
                        'state_abbrv': ['state_abbr', 'state_code', 'abbr']
                    }
                    
                    for alt_name in alternatives.get(old_col, []):
                        if alt_name in df.columns:
                            df = df.rename(columns={alt_name: new_col})
                            break
            
            # Ensure we have the required columns
            required_cols = ['fips', 'county_name', 'state_abbrv']
            if all(col in df.columns for col in required_cols):
                # Ensure FIPS codes are properly formatted (5 digits with leading zeros)
                df['fips'] = df['fips'].astype(str).str.zfill(5)
                return df
            else:
                st.warning(f"External FIPS data missing required columns. Available: {list(df.columns)}")
                return None
            
    except Exception as e:
        st.warning(f"Could not load full FIPS data: {str(e)}. Using sample data.")
        return None

def create_county_fips_lookup(df: pd.DataFrame) -> Dict[str, str]:
    """
    Create a lookup dictionary from county name + state to FIPS code
    
    Args:
        df: DataFrame with county data
        
    Returns:
        Dictionary mapping "County Name, ST" to FIPS code
    """
    fips_df = load_full_fips_data()
    if fips_df is None:
        fips_df = get_fips_mapping()
    
    lookup = {}
    for _, row in fips_df.iterrows():
        try:
            # Check if required columns exist
            if 'county_name' not in row.index or 'state_abbrv' not in row.index or 'fips' not in row.index:
                continue
                
            # Create keys in multiple formats to match potential data variations
            county_clean = str(row['county_name']).replace(' County', '').replace(' Parish', '').replace(' Borough', '')
            state_abbr = str(row['state_abbrv'])
            fips = str(row['fips'])
            
            # Multiple key formats for better matching
            lookup[f"{county_clean}, {state_abbr}"] = fips
            lookup[f"{county_clean.upper()}, {state_abbr}"] = fips
            lookup[f"{row['county_name']}, {state_abbr}"] = fips
            lookup[f"{row['county_name'].upper()}, {state_abbr}"] = fips
        except (KeyError, AttributeError):
            # Skip rows with missing data
            continue
    
    return lookup

def add_fips_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add FIPS codes to a dataframe that has county_name and state_abbrv columns
    
    Args:
        df: DataFrame with county-level data
        
    Returns:
        DataFrame with fips column added
    """
    if 'county_name' not in df.columns or 'state_abbrv' not in df.columns:
        # If we don't have the required columns, create dummy FIPS for demo
        df['fips'] = pd.Series(range(len(df))).astype(str).str.zfill(5)
        return df
    
    # Create lookup dictionary
    fips_lookup = create_county_fips_lookup(df)
    
    # Add FIPS codes
    def get_fips(row):
        county = str(row['county_name']).strip()
        state = str(row['state_abbrv']).strip()
        
        # Try different combinations
        for key_format in [
            f"{county}, {state}",
            f"{county.upper()}, {state}",
            f"{county.replace(' County', '')}, {state}",
            f"{county.replace(' County', '').upper()}, {state}"
        ]:
            if key_format in fips_lookup:
                return fips_lookup[key_format]
        
        # If no match found, create a dummy FIPS (this should be rare with good data)
        return f"99{hash(f'{county}{state}') % 1000:03d}"
    
    df['fips'] = df.apply(get_fips, axis=1)
    return df

# State codes for reference
STATE_CODES = {
    'AL': '01', 'AK': '02', 'AZ': '04', 'AR': '05', 'CA': '06', 'CO': '08', 'CT': '09', 'DE': '10',
    'FL': '12', 'GA': '13', 'HI': '15', 'ID': '16', 'IL': '17', 'IN': '18', 'IA': '19', 'KS': '20',
    'KY': '21', 'LA': '22', 'ME': '23', 'MD': '24', 'MA': '25', 'MI': '26', 'MN': '27', 'MS': '28',
    'MO': '29', 'MT': '30', 'NE': '31', 'NV': '32', 'NH': '33', 'NJ': '34', 'NM': '35', 'NY': '36',
    'NC': '37', 'ND': '38', 'OH': '39', 'OK': '40', 'OR': '41', 'PA': '42', 'RI': '44', 'SC': '45',
    'SD': '46', 'TN': '47', 'TX': '48', 'UT': '49', 'VT': '50', 'VA': '51', 'WA': '53', 'WV': '54',
    'WI': '55', 'WY': '56'
}
