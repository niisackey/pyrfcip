# rfcip/utils.py
import pandas as pd
import re
import os
import logging
from pathlib import Path
from typing import List, Union, Any, Optional, Dict
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# State to FIPS mapping (50 states + DC)
STATE_FIPS_TO_NAME = {
    "01": "Alabama",
    "02": "Alaska",
    "04": "Arizona",
    "05": "Arkansas",
    "06": "California",
    "08": "Colorado",
    "09": "Connecticut",
    "10": "Delaware",
    "11": "District of Columbia",
    "12": "Florida",
    "13": "Georgia",
    "15": "Hawaii",
    "16": "Idaho",
    "17": "Illinois",
    "18": "Indiana",
    "19": "Iowa",
    "20": "Kansas",
    "21": "Kentucky",
    "22": "Louisiana",
    "23": "Maine",
    "24": "Maryland",
    "25": "Massachusetts",
    "26": "Michigan",
    "27": "Minnesota",
    "28": "Mississippi",
    "29": "Missouri",
    "30": "Montana",
    "31": "Nebraska",
    "32": "Nevada",
    "33": "New Hampshire",
    "34": "New Jersey",
    "35": "New Mexico",
    "36": "New York",
    "37": "North Carolina",
    "38": "North Dakota",
    "39": "Ohio",
    "40": "Oklahoma",
    "41": "Oregon",
    "42": "Pennsylvania",
    "44": "Rhode Island",
    "45": "South Carolina",
    "46": "South Dakota",
    "47": "Tennessee",
    "48": "Texas",
    "49": "Utah",
    "50": "Vermont",
    "51": "Virginia",
    "53": "Washington",
    "54": "West Virginia",
    "55": "Wisconsin",
    "56": "Wyoming"
}



STATE_TO_FIPS = {
    'AL': '01', 'AK': '02', 'AZ': '04', 'AR': '05', 'CA': '06',
    'CO': '08', 'CT': '09', 'DE': '10', 'DC': '11', 'FL': '12',
    'GA': '13', 'HI': '15', 'ID': '16', 'IL': '17', 'IN': '18',
    'IA': '19', 'KS': '20', 'KY': '21', 'LA': '22', 'ME': '23',
    'MD': '24', 'MA': '25', 'MI': '26', 'MN': '27', 'MS': '28',
    'MO': '29', 'MT': '30', 'NE': '31', 'NV': '32', 'NH': '33',
    'NJ': '34', 'NM': '35', 'NY': '36', 'NC': '37', 'ND': '38',
    'OH': '39', 'OK': '40', 'OR': '41', 'PA': '42', 'RI': '44',
    'SC': '45', 'SD': '46', 'TN': '47', 'TX': '48', 'UT': '49',
    'VT': '50', 'VA': '51', 'WA': '53', 'WV': '54', 'WI': '55',
    'WY': '56'
}

# Inverse mapping for state names
STATE_NAMES_TO_ABBR = {
    'alabama': 'AL', 'alaska': 'AK', 'arizona': 'AZ', 'arkansas': 'AR',
    'california': 'CA', 'colorado': 'CO', 'connecticut': 'CT', 'delaware': 'DE',
    'district of columbia': 'DC', 'florida': 'FL', 'georgia': 'GA',
    'hawaii': 'HI', 'idaho': 'ID', 'illinois': 'IL', 'indiana': 'IN',
    'iowa': 'IA', 'kansas': 'KS', 'kentucky': 'KY', 'louisiana': 'LA',
    'maine': 'ME', 'maryland': 'MD', 'massachusetts': 'MA', 'michigan': 'MI',
    'minnesota': 'MN', 'mississippi': 'MS', 'missouri': 'MO', 'montana': 'MT',
    'nebraska': 'NE', 'nevada': 'NV', 'new hampshire': 'NH', 'new jersey': 'NJ',
    'new mexico': 'NM', 'new york': 'NY', 'north carolina': 'NC',
    'north dakota': 'ND', 'ohio': 'OH', 'oklahoma': 'OK', 'oregon': 'OR',
    'pennsylvania': 'PA', 'rhode island': 'RI', 'south carolina': 'SC',
    'south dakota': 'SD', 'tennessee': 'TN', 'texas': 'TX', 'utah': 'UT',
    'vermont': 'VT', 'virginia': 'VA', 'washington': 'WA', 'west virginia': 'WV',
    'wisconsin': 'WI', 'wyoming': 'WY'
}

# Global variable for county data
COUNTY_DATA = None

def get_county_data() -> pd.DataFrame:
    """
    Get county FIPS data - either from cache, download, or fallback
    
    Returns:
        DataFrame with columns: fips, name
    """
    global COUNTY_DATA
    if COUNTY_DATA is not None:
        return COUNTY_DATA
    
    # Create cache directory
    cache_dir = Path.home() / ".rfcip_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / "county_fips.csv"
    
    # Try to load from local cache
    if csv_path.exists():
        try:
            COUNTY_DATA = pd.read_csv(csv_path, dtype={'fips': str})
            logger.info("Loaded county data from local cache")
            return COUNTY_DATA
        except Exception as e:
            logger.warning(f"Error reading county CSV: {e}")
    
    # Try to download from US Census (primary source)
    try:
        url = "https://www2.census.gov/geo/docs/reference/codes/files/national_county.txt"
        headers = ["state", "state_fips", "county_fips", "county_name", "fips_class"]
        df = pd.read_csv(url, sep=",", names=headers, dtype=str, encoding='latin1')
        
        # Create 5-digit FIPS code
        df["fips"] = df["state_fips"] + df["county_fips"]
        df["name"] = df["county_name"] + " County"
        
        # Filter out territories (keep only states + DC)
        df = df[df["state_fips"].isin(STATE_TO_FIPS.values())]
        
        COUNTY_DATA = df[["fips", "name"]].copy()
        
        # Save to cache
        COUNTY_DATA.to_csv(csv_path, index=False)
        logger.info("Downloaded and saved county data from US Census")
        return COUNTY_DATA
    except Exception as e:
        logger.error(f"Failed to download county data from US Census: {e}")
    
    # Try to download from GitHub fallback
    try:
        url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Extract county names from GeoJSON
        data = response.json()
        counties = []
        for feature in data['features']:
            fips = feature['id']
            name = feature['properties']['name']
            counties.append({"fips": fips, "name": name + " County"})
        
        COUNTY_DATA = pd.DataFrame(counties)
        
        # Save to cache
        COUNTY_DATA.to_csv(csv_path, index=False)
        logger.info("Downloaded and saved county data from GitHub")
        return COUNTY_DATA
    except Exception as e:
        logger.error(f"Failed to download county data from GitHub: {e}")
    
    # Ultimate fallback to minimal dataset
    logger.warning("Using minimal county dataset")
    counties = [
        {"fips": "01001", "name": "Autauga County"},
        {"fips": "01003", "name": "Baldwin County"},
        {"fips": "19153", "name": "Polk County"},
        {"fips": "19155", "name": "Pottawattamie County"},
        {"fips": "56045", "name": "Uinta County"},
    ]
    COUNTY_DATA = pd.DataFrame(counties)
    
    # Save fallback to CSV for future
    try:
        COUNTY_DATA.to_csv(csv_path, index=False)
    except:
        pass
    
    return COUNTY_DATA

def get_county_name(fips_code: Union[str, int]) -> str:
    """
    Convert FIPS code to county name
    
    Args:
        fips_code: 5-digit FIPS code
        
    Returns:
        County name if found, else empty string
    """
    county_data = get_county_data()
    fips_str = str(fips_code).zfill(5)
    
    match = county_data[county_data['fips'] == fips_str]
    if not match.empty:
        return match.iloc[0]['name']
    return ""






def convert_state_to_fips(state_input: Union[str, int]) -> str:
    """
    Convert state identifier to 2-digit FIPS code
    
    Args:
        state_input: Can be state name, abbreviation, or FIPS code
        
    Returns:
        2-digit FIPS code as string
        
    Raises:
        ValueError if input can't be matched
    """
    # Handle integer inputs
    if isinstance(state_input, int):
        fips_str = str(state_input).zfill(2)
        if fips_str in STATE_TO_FIPS.values():
            return fips_str
        raise ValueError(f"Invalid state FIPS code: {state_input}")
    
    # Convert to uppercase for consistency
    state_clean = state_input.strip().upper()
    
    # Check if it's already a valid FIPS code
    if state_clean in STATE_TO_FIPS.values():
        return state_clean
    
    # Check if it's a state abbreviation
    if state_clean in STATE_TO_FIPS:
        return STATE_TO_FIPS[state_clean]
    
    # Check if it's a full state name (case-insensitive)
    state_lower = state_input.strip().lower()
    if state_lower in STATE_NAMES_TO_ABBR:
        return STATE_TO_FIPS[STATE_NAMES_TO_ABBR[state_lower]]
    
    # Try to match without spaces/special characters
    state_normalized = re.sub(r'[^a-z]', '', state_lower)
    for name, abbr in STATE_NAMES_TO_ABBR.items():
        name_normalized = re.sub(r'[^a-z]', '', name)
        if name_normalized == state_normalized:
            return STATE_TO_FIPS[abbr]
    
    # If no match found
    raise ValueError(f"Invalid state identifier: {state_input}")

def clean_fips(fips_input: Union[str, int]) -> str:
    """
    Clean and validate FIPS code to 5 digits (state + county)
    
    Args:
        fips_input: County FIPS code (string or integer)
        
    Returns:
        5-digit FIPS code as string
        
    Raises:
        ValueError for invalid FIPS codes
    """
    # Remove non-digit characters
    fips_str = ''.join(filter(str.isdigit, str(fips_input)))
    
    # Pad to 5 digits
    if len(fips_str) == 4:
        fips_str = '0' + fips_str  # Pad leading zero if only 4 digits
    elif len(fips_str) == 3:
        fips_str = '00' + fips_str  # Pad two zeros if only 3 digits
    elif len(fips_str) == 2:
        fips_str = '000' + fips_str  # State-only code, but we need county
    
    # Validate length
    if len(fips_str) != 5:
        raise ValueError(f"FIPS code must be 4 or 5 digits, got {fips_input} (cleaned: {fips_str})")
    
    # Validate state portion
    state_part = fips_str[:2]
    if state_part not in STATE_TO_FIPS.values():
        raise ValueError(f"Invalid state FIPS code: {state_part}")
    
    return fips_str

def convert_to_list(value: Union[Any, List[Any]], none_value: Any = None) -> List[Any]:
    """
    Convert a value to a list if it's not already a list.
    
    Args:
        value: Input value to convert
        none_value: Value to return if input is None
        
    Returns:
        List containing the input value(s)
    """
    if value is None:
        return [] if none_value is None else [none_value]
    if isinstance(value, list):
        return value
    return [value]

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean DataFrame column names to snake_case
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with cleaned column names
    """
    if df.empty:
        return df
        
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r'\s+', '_', regex=True)  # Replace spaces with underscores
        .str.replace(r'[^\w_]', '', regex=True)  # Remove special characters
        .str.replace(r'_{2,}', '_', regex=True)  # Replace multiple underscores
        .str.strip('_')
    )
    return df



# Add to rfcip/utils.py after the existing functions

def convert_state_to_alpha(state_input: Union[str, int]) -> str:
    """
    Convert state identifier to 2-letter abbreviation
    
    Args:
        state_input: Can be state name, abbreviation, or FIPS code
        
    Returns:
        2-letter state abbreviation
        
    Raises:
        ValueError if input can't be matched
    """
    # Handle integer inputs
    if isinstance(state_input, int):
        fips_str = str(state_input).zfill(2)
        if fips_str in STATE_TO_FIPS.values():
            # Create inverse mapping
            fips_to_alpha = {v: k for k, v in STATE_TO_FIPS.items()}
            return fips_to_alpha[fips_str]
        raise ValueError(f"Invalid state FIPS code: {state_input}")
    
    # Convert to uppercase for consistency
    state_clean = state_input.strip().upper()
    
    # Check if it's already a valid abbreviation
    if state_clean in STATE_TO_FIPS:
        return state_clean
    
    # Check if it's a full state name (case-insensitive)
    state_lower = state_input.strip().lower()
    if state_lower in STATE_NAMES_TO_ABBR:
        return STATE_NAMES_TO_ABBR[state_lower]
    
    # Check if it's a FIPS code
    if state_clean in STATE_TO_FIPS.values():
        # Create inverse mapping
        fips_to_alpha = {v: k for k, v in STATE_TO_FIPS.items()}
        return fips_to_alpha[state_clean]
    
    # Try to match without spaces/special characters
    state_normalized = re.sub(r'[^a-z]', '', state_lower)
    for name, abbr in STATE_NAMES_TO_ABBR.items():
        name_normalized = re.sub(r'[^a-z]', '', name)
        if name_normalized == state_normalized:
            return abbr
    
    # If no match found
    raise ValueError(f"Invalid state identifier: {state_input}")

def convert_to_numeric(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Convert specified columns to numeric type. If no columns specified,
    converts all columns that appear to contain numeric data.
    
    Args:
        df: Input DataFrame
        columns: List of columns to convert (optional)
        
    Returns:
        DataFrame with converted columns
    """
    if df.empty:
        return df
        
    if columns is None:
        # Automatically detect numeric columns
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    df[col] = pd.to_numeric(df[col], errors='raise')
                except:
                    pass
    else:
        for col in columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def convert_to_datetime(df: pd.DataFrame, columns: List[str], format: Optional[str] = None) -> pd.DataFrame:
    """
    Convert specified columns to datetime type
    
    Args:
        df: Input DataFrame
        columns: List of columns to convert
        format: Optional datetime format string
        
    Returns:
        DataFrame with converted columns
    """
    if df.empty:
        return df
        
    for col in columns:
        if col in df.columns:
            try:
                if format:
                    df[col] = pd.to_datetime(df[col], format=format, errors='coerce')
                else:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception as e:
                logger.warning(f"Error converting {col} to datetime: {str(e)}")
    return df

def create_cache_dir(cache_dir: Union[str, Path]) -> Path:
    """
    Create cache directory if it doesn't exist
    
    Args:
        cache_dir: Path to cache directory
        
    Returns:
        Path object for the cache directory
    """
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path

def generate_cache_key(params: Dict[str, Any]) -> str:
    """
    Generate a unique cache key based on parameters
    
    Args:
        params: Dictionary of parameters
        
    Returns:
        MD5 hash string
    """
    import hashlib
    import json
    
    # Create a stable string representation
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(param_str.encode()).hexdigest()

def filter_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Automatically detect and convert numeric columns
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with numeric columns converted
    """
    if df.empty:
        return df
        
    for col in df.columns:
        # Skip if already numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
            
        # Try converting to numeric
        try:
            converted = pd.to_numeric(df[col], errors='raise')
            df[col] = converted
        except:
            # Try handling percentage values
            if df[col].astype(str).str.endswith('%').any():
                try:
                    df[col] = pd.to_numeric(df[col].str.rstrip('%'), errors='coerce') / 100.0
                except:
                    pass
    return df

def safe_read_excel(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Safely read Excel file with multiple fallback methods
    
    Args:
        file_path: Path to Excel file
        kwargs: Additional arguments to pass to pd.read_excel
        
    Returns:
        DataFrame with contents of Excel file
    """
    try:
        # First try with openpyxl
        return pd.read_excel(file_path, engine='openpyxl', **kwargs)
    except Exception as e1:
        try:
            # Try with xlrd
            return pd.read_excel(file_path, engine='xlrd', **kwargs)
        except Exception as e2:
            try:
                # Try without specifying engine
                return pd.read_excel(file_path, **kwargs)
            except Exception as e3:
                # Try skipping headers
                try:
                    return pd.read_excel(file_path, skiprows=1, **kwargs)
                except Exception as e4:
                    raise RuntimeError(
                        f"Failed to read Excel file: {str(e1)}\n{str(e2)}\n{str(e3)}\n{str(e4)}"
                    )
                    
# Add this at the bottom of utils.py
__all__ = [
    'convert_state_to_fips', 
    'clean_fips', 
    'convert_to_list',
    'clean_column_names',
    'convert_to_numeric',
    'convert_to_datetime',
    'create_cache_dir',
    'generate_cache_key',
    'filter_numeric_columns',
    'safe_read_excel',
    'convert_state_to_alpha',  # Add this
    'STATE_TO_FIPS'  # Add this to expose the constant
]