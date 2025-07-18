# codes.py
import pandas as pd
import requests
import zipfile
import io
import os
import re
import logging
import sys
import warnings
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Union, List, Optional
from pathlib import Path
import tempfile
import numpy as np
from io import BytesIO, StringIO
import time
from .utils import (convert_state_to_fips, convert_state_to_alpha, 
                   STATE_TO_FIPS, convert_to_list, clean_fips, get_county_name)

QUICK_STATS_API_KEY = "6100ED4E-087C-30F5-BB90-E7EABB501E51"

FIPS_TO_ALPHA = {v: k for k, v in STATE_TO_FIPS.items()}



# Configure logging
logger = logging.getLogger("rfcip.codes")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

LIVESTOCK_COMMODITIES = {
    "CATTLE": "ANIMALS & PRODUCTS",
    "HOGS": "ANIMALS & PRODUCTS",
    "CHICKENS": "ANIMALS & PRODUCTS",
    "TURKEYS": "ANIMALS & PRODUCTS",
    "MILK": "DAIRY",
    "EGGS": "POULTRY",
    "SHEEP": "ANIMALS & PRODUCTS",
    "GOATS": "ANIMALS & PRODUCTS"
}

COMMODITY_ALIASES = {
    "RICE": ["RICE", "RICE, LONG GRAIN", "RICE, SHORT & MEDIUM GRAIN"],
    "CATTLE": ["CATTLE", "CATTLE, (EXCL COWS)"],
    "MILK": ["MILK", "MILK, ALL CLASSES"],
    "EGGS": ["EGGS", "EGGS, HENS & BROILERS"],
    "HOGS": ["HOGS", "HOGS, ALL"]
}

SOURCE_ALTERNATIVES = ["SURVEY", "CENSUS", "ADMINISTRATIVE"]
AGG_LEVELS = ["STATE", "NATIONAL", "COUNTY"]

# ------------------- RMA SPECIFIC UTILITIES -------------------
# State to FIPS mapping (50 states + DC)
RMA_STATE_TO_FIPS = {
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
RMA_STATE_NAMES_TO_ABBR = {
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

def _rma_convert_state_to_fips(state_input: Union[str, int]) -> str:
    """Convert state input to two-digit FIPS code for RMA data"""
    if isinstance(state_input, int):
        return str(state_input).zfill(2)
    
    state_clean = state_input.strip().upper()
    
    # Check if it's already a FIPS code
    if state_clean in RMA_STATE_TO_FIPS.values():
        return state_clean
    
    # Check if it's a state abbreviation
    if state_clean in RMA_STATE_TO_FIPS:
        return RMA_STATE_TO_FIPS[state_clean]
    
    # Check if it's a state name
    state_lower = state_input.strip().lower()
    if state_lower in RMA_STATE_NAMES_TO_ABBR:
        return RMA_STATE_TO_FIPS[RMA_STATE_NAMES_TO_ABBR[state_lower]]
    
    # Try to match without spaces/special characters
    state_normalized = re.sub(r'[^a-z]', '', state_lower)
    for name, abbr in RMA_STATE_NAMES_TO_ABBR.items():
        if re.sub(r'[^a-z]', '', name) == state_normalized:
            return RMA_STATE_TO_FIPS[abbr]
    
    raise ValueError(f"Invalid state identifier for RMA: {state_input}")

# ------------------- RMA DATA FETCHING UTILITIES -------------------
def _rma_fetch_usda_data(
    url: str,
    required_columns: List[str],
    fallback_path: Union[str, Path],
    cache_dir: Union[str, Path],
    years: List[int],
    max_retries: int = 3
) -> pd.DataFrame:
    """Core function to fetch and process RMA data with caching and fallback"""
    # Create cache directory
    Path(cache_dir).mkdir(exist_ok=True, parents=True)
    
    # Generate cache key
    cache_key = re.sub(r'\W+', '_', url.split('?')[-1]) + f"_{'_'.join(map(str, years))}"
    cache_file = Path(cache_dir) / f"{cache_key}.parquet"
    
    # Use cached data if available
    if cache_file.exists():
        try:
            return pd.read_parquet(cache_file)
        except Exception as e:
            logger.warning(f"Error reading RMA cache: {e}. Fetching fresh data.")
    
    # Try to fetch fresh data with retries
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Read Excel file
            with BytesIO(response.content) as bio:
                # First attempt to read
                df = pd.read_excel(bio, engine="openpyxl")
                
                # Check for header issues (common in USDA files)
                if len(df.columns) > 1 and df.columns[1].lower() in ['x2', 'unnamed: 1']:
                    bio.seek(0)
                    df = pd.read_excel(bio, skip=1, engine="openpyxl")
                
                # Clean column names
                df.columns = (
                    df.columns
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .str.replace(r'\W+', '_', regex=True)
                    .str.strip('_')
                )
                
                # Select required columns
                df = df.loc[:, [c for c in required_columns if c in df.columns]]
                
                # Cache the data
                df.to_parquet(cache_file)
                return df
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
        except Exception as e:
            logger.warning(f"Processing error: {e}")
            break
    
    # Fallback to local file if download fails
    warnings.warn("Using fallback data for RMA")
    try:
        return pd.read_csv(fallback_path)
    except Exception as e:
        warnings.warn(f"Fallback failed: {e}")
        return pd.DataFrame(columns=required_columns)

# ------------------- CROP CODE FUNCTIONS -------------------
def get_crop_codes(
    crop: Optional[Union[str, int, List[str], List[int]]] = None,
    year: Union[int, List[int]] = datetime.now().year,
    fallback_path: Union[str, Path] = "rma_crop_codes_fallback.csv",
    cache_dir: Union[str, Path] = "rma_cache"
) -> pd.DataFrame:
    """
    Fetch crop codes from USDA Summary of Business report
    
    Args:
        crop: Commodity names (case-insensitive) or codes to filter
        year: Year(s) to fetch data for (default: current year)
        fallback_path: Local CSV path if download fails
        cache_dir: Directory to cache downloaded files
        
    Returns:
        DataFrame with columns: commodity_year, commodity_code, commodity_name
    """
    # Validate and normalize inputs
    years = [year] if isinstance(year, int) else year
    crops = [crop] if crop is not None and not isinstance(crop, list) else crop
    
    # Construct URL
    base_url = "https://public-rma.fpac.usda.gov/apps/SummaryOfBusiness/ReportGenerator/ExportToExcel"
    params = {
        "CY": ",".join(map(str, years)),
        "ORD": "CY,CM",
        "CC": "S",
        "VisibleColumns": "CommodityYear,CommodityCode,CommodityName",
        "SortField": "",
        "SortDir": ""
    }
    url = f"{base_url}?{'&'.join(f'{k}={v}' for k,v in params.items())}"
    
    # Fetch data
    df = _rma_fetch_usda_data(
        url,
        required_columns=['commodity_year', 'commodity_code', 'commodity_name'],
        fallback_path=fallback_path,
        cache_dir=cache_dir,
        years=years
    )
    
    # Apply crop filters if specified
    if crop is not None:
        if all(isinstance(c, str) for c in crops):
            # Case-insensitive name matching
            name_mask = df['commodity_name'].str.lower().isin([c.lower() for c in crops])
            result = df[name_mask]
        elif all(isinstance(c, int) for c in crops):
            # Numeric code matching
            result = df[df['commodity_code'].isin(crops)]
        else:
            raise TypeError("Crop must be all strings or all integers")
        
        # Handle no matches
        if result.empty:
            warnings.warn("No matching crops found. Returning all crops")
            result = df
    else:
        result = df
    
    # Add backward compatible columns
    result = result.rename(columns={
        'commodity_name': 'CROP_NAME',
        'commodity_code': 'CROP_CODE'
    })
    result['commodity_name'] = result['CROP_NAME']
    
    return result

# ------------------- INSURANCE PLAN CODE FUNCTIONS -------------------
def get_insurance_plan_codes(
    plan: Optional[Union[str, int, List[str], List[int]]] = None,
    year: Union[int, List[int]] = datetime.now().year,
    fallback_path: Union[str, Path] = "rma_plan_codes_fallback.csv",
    cache_dir: Union[str, Path] = "rma_cache"
) -> pd.DataFrame:
    """
    Fetch insurance plan codes from USDA Summary of Business report
    
    Args:
        plan: Plan names (case-insensitive), abbreviations, or codes to filter
        year: Year(s) to fetch data for (default: current year)
        fallback_path: Local CSV path if download fails
        cache_dir: Directory to cache downloaded files
        
    Returns:
        DataFrame with columns: commodity_year, insurance_plan_code, 
        insurance_plan, insurance_plan_abbrv
    """
    # Validate and normalize inputs
    years = [year] if isinstance(year, int) else year
    plans = [plan] if plan is not None and not isinstance(plan, list) else plan
    
    # Construct URL
    base_url = "https://public-rma.fpac.usda.gov/apps/SummaryOfBusiness/ReportGenerator/ExportToExcel"
    params = {
        "CY": ",".join(map(str, years)),
        "ORD": "CY,IP",
        "CC": "B",
        "VisibleColumns": "CommodityYear,InsurancePlanCode,InsurancePlanName,InsurancePlanAbbreviation",
        "SortField": "",
        "SortDir": ""
    }
    url = f"{base_url}?{'&'.join(f'{k}={v}' for k,v in params.items())}"
    
    # Fetch data
    df = _rma_fetch_usda_data(
        url,
        required_columns=[
            'commodity_year', 
            'insurance_plan_code', 
            'insurance_plan', 
            'insurance_plan_abbreviation'
        ],
        fallback_path=fallback_path,
        cache_dir=cache_dir,
        years=years
    )
    
    # Standardize column names
    col_map = {
        'insurance_plan_abbreviation': 'insurance_plan_abbrv',
        'insurance_plan_abbrv': 'insurance_plan_abbrv',
        'abbreviation': 'insurance_plan_abbrv'
    }
    df = df.rename(columns={c: col_map.get(c, c) for c in df.columns})
    
    # Apply plan filters if specified
    if plan is not None:
        # Try matching by abbreviation first
        if all(isinstance(p, str) for p in plans):
            abbr_mask = df['insurance_plan_abbrv'].str.lower().isin([p.lower() for p in plans])
            result = df[abbr_mask]
            
            # If no abbreviation matches, try full name
            if result.empty:
                name_mask = df['insurance_plan'].str.lower().isin([p.lower() for p in plans])
                result = df[name_mask]
        elif all(isinstance(p, int) for p in plans):
            # Numeric code matching
            result = df[df['insurance_plan_code'].isin(plans)]
        else:
            raise TypeError("Plan must be all strings or all integers")
        
        # Handle no matches
        if result.empty:
            # Try numeric matching if string matching failed
            try:
                numeric_plans = [int(p) for p in plans]
                result = df[df['insurance_plan_code'].isin(numeric_plans)]
            except:
                pass
            
            if result.empty:
                warnings.warn("No matching plans found. Returning all plans")
                return df
        return result
    
    return df

# ------------------- RMA PRICE DATA FUNCTION -------------------
def get_rma_price_data(
    year: Optional[Union[int, List[int]]] = None,
    crop: Optional[Union[str, int, List[Union[str, int]]]] = None,
    state: Optional[Union[str, int, List[Union[str, int]]]] = None
) -> pd.DataFrame:
    """
    Download RMA projected and harvest prices for revenue protection insurance plans
    
    Args:
        year: Commodity year(s) (default: current year)
        crop: Crop name(s) or code(s) (use get_crop_codes() for reference)
        state: State identifier(s) (name, abbreviation, or FIPS code)
    
    Returns:
        DataFrame with projected and harvest prices
    """
    logger.info("Downloading RMA price data")
    
    # Set default to current year if not specified
    if year is None:
        year = datetime.now().year
    
    # Base URL
    base_url = "https://public-rma.fpac.usda.gov/apps/PriceDiscovery/Services/RevenuePriceDataService.svc/RevenuePrices?"
    
    # Build URL parameters
    params = []
    
    # Handle year parameter
    if isinstance(year, int):
        year = [year]
    if year:
        params.append(f"commodityYears={','.join(map(str, year))}")
    
    # Handle crop parameter
    if crop is not None:
        if not isinstance(crop, list):
            crop = [crop]
        
        # Get crop codes
        crop_df = get_crop_codes(crop=crop)
        if crop_df.empty:
            warnings.warn("No valid crops found. Returning empty DataFrame")
            return pd.DataFrame()
        
        crop_codes = crop_df['commodity_code'].unique().tolist()
        params.append(f"commodityCodes={','.join(map(str, crop_codes))}")
    
    # Handle state parameter
    if state is not None:
        if not isinstance(state, list):
            state = [state]
        
        state_fips = []
        for s in state:
            try:
                state_fips.append(_rma_convert_state_to_fips(s))
            except ValueError as e:
                warnings.warn(str(e))
        
        if state_fips:
            params.append(f"stateCodes={','.join(state_fips)}")
    
    # Construct full URL
    url = base_url + "&".join(params)
    logger.debug(f"RMA Price API URL: {url}")
    
    try:
        # Fetch XML data
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Parse XML
        root = ET.fromstring(response.content)
        ns = {'d': 'http://schemas.datacontract.org/2004/07/RMA.Common.Model'}
        
        records = []
        for elem in root.findall('.//d:RevenuePrice', ns):
            record = {}
            for child in elem:
                # Remove namespace from tag
                tag = child.tag.split('}', 1)[-1] if '}' in child.tag else child.tag
                record[tag] = child.text
            records.append(record)
        
        # Create DataFrame
        df = pd.DataFrame(records)
        
        if df.empty:
            logger.warning("No RMA price data found for specified parameters")
            return df
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Remove unnecessary columns (matches R implementation)
        cols_to_remove = df.columns[df.columns.str.contains(
            'RSS|Display|Composite|Actuarial', 
            case=False, 
            regex=True
        )]
        df = df.drop(columns=cols_to_remove, errors='ignore')
        
        # Convert date columns
        date_cols = df.columns[df.columns.str.contains('Date')]
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Convert numeric columns
        non_date_cols = df.columns.difference(date_cols)
        for col in non_date_cols:
            # Skip columns that are already strings or dates
            if df[col].dtype == object:
                # Try converting to numeric
                df[col] = pd.to_numeric(df[col], errors='ignore')
        
        return df
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error: {str(e)}")
        return pd.DataFrame()
    except ET.ParseError as e:
        logger.error(f"XML parsing error: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return pd.DataFrame()

# ------------------- QUICK STATS PRICE DATA FUNCTION -------------------
def get_valid_states_for_crop(commodity: str, year: int, source: str) -> List[str]:
    try:
        params = {
            "key": QUICK_STATS_API_KEY,
            "param": "state_alpha",
            "year": year,
            "commodity_desc": commodity,
            "source_desc": source
        }
        r = requests.get("https://quickstats.nass.usda.gov/api/get_param_values/", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        valid_states = data.get("valid_values", [])
        if not valid_states:
            logger.warning(f"⚠️ USDA param API returned no valid states for {commodity} ({year}) with {source}")
        logger.debug(f"Valid states for {commodity} ({year}) with {source}: {valid_states}")
        return valid_states
    except Exception as e:
        logger.warning(f"Could not fetch valid states for {commodity}, {year}, {source}: {e}")
        return []

def get_valid_counties_for_crop(commodity: str, year: int, source: str, state_alpha: str) -> List[str]:
    try:
        params = {
            "key": QUICK_STATS_API_KEY,
            "param": "county_name",
            "year": year,
            "commodity_desc": commodity,
            "source_desc": source,
            "state_alpha": state_alpha
        }
        r = requests.get("https://quickstats.nass.usda.gov/api/get_param_values/", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        valid_counties = data.get("valid_values", [])
        logger.debug(f"Valid counties for {commodity} in {state_alpha} ({year}) with {source}: {valid_counties}")
        return valid_counties
    except Exception as e:
        logger.warning(f"Could not fetch valid counties for {commodity}, {year}, {source}, {state_alpha}: {e}")
        return []

def resolve_commodity_alias(crop_name: str) -> str:
    for key, aliases in COMMODITY_ALIASES.items():
        if crop_name in aliases:
            logger.info(f"Mapped alias '{crop_name}' to commodity '{key}'")
            return key
    return crop_name

def get_price_data(
    year: Union[int, List[int]] = None,
    crop: Union[str, List[str]] = None,
    state: Union[str, List[str]] = None,
    max_retries: int = 3,
    retry_delay: int = 5,
    request_delay: int = 1,
    granularity: str = "auto"
) -> pd.DataFrame:
    if year is None:
        year = datetime.now().year

    years = convert_to_list(year)
    current_year = datetime.now().year
    valid_years = [y for y in years if isinstance(y, int) and 1990 <= y <= current_year + 1]
    if not valid_years:
        logger.warning("No valid years provided, using current year")
        valid_years = [current_year]

    crops = convert_to_list(crop) or ["CORN"]

    all_results = []
    attempted_requests = 0
    successful_requests = 0

    for y in valid_years:
        for crop_raw in crops:
            c_str = str(crop_raw).strip().upper()
            commodity_std = resolve_commodity_alias(c_str)
            commodity_variants = COMMODITY_ALIASES.get(commodity_std, [commodity_std])

            for variant in commodity_variants:
                for source_desc in SOURCE_ALTERNATIVES:
                    valid_states = get_valid_states_for_crop(variant, y, source_desc)

                    if state:
                        requested_states = []
                        for s in convert_to_list(state):
                            try:
                                fips_code = convert_state_to_fips(s)
                                state_alpha = FIPS_TO_ALPHA.get(fips_code)
                                if state_alpha:
                                    requested_states.append(state_alpha)
                            except Exception as e:
                                logger.error(f"State conversion failed for {s}: {str(e)}")
                    else:
                        requested_states = valid_states

                    levels_to_try = ["STATE"]
                    if granularity == "auto":
                        levels_to_try += ["COUNTY", "NATIONAL"]
                    elif granularity in AGG_LEVELS:
                        levels_to_try = [granularity]

                    for level in levels_to_try:
                        if level == "NATIONAL":
                            location_pairs = [(None, None)]
                        elif level == "COUNTY":
                            location_pairs = []
                            for st in requested_states:
                                counties = get_valid_counties_for_crop(variant, y, source_desc, st)
                                location_pairs.extend([(st, county) for county in counties])
                        else:
                            location_pairs = [(st, None) for st in requested_states]

                        for st, county in location_pairs:
                            params = {
                                "key": QUICK_STATS_API_KEY,
                                "source_desc": source_desc,
                                "statisticcat_desc": "PRICE RECEIVED",
                                "format": "JSON",
                                "commodity_desc": variant,
                                "year": y,
                                "agg_level_desc": level
                            }
                            if st:
                                params["state_alpha"] = st
                            if county:
                                params["county_name"] = county

                            attempted_requests += 1

                            try:
                                logger.debug(f"Requesting: {params}")
                                r = requests.get("https://quickstats.nass.usda.gov/api/api_GET", params=params, timeout=30)

                                if r.status_code == 400:
                                    logger.warning(f"400 Bad Request for {variant} in {st}/{county} ({y}) with {source_desc} at {level} level.")
                                    continue

                                r.raise_for_status()
                                data = r.json()

                                if 'data' not in data or not data['data']:
                                    logger.info(f"✅ Valid request but no data found: {variant} in {st}/{county} ({y}) via {source_desc} at {level} level")
                                    continue

                                df = pd.DataFrame(data['data'])
                                if df.empty:
                                    logger.info(f"Empty DataFrame returned for {variant} in {st}/{county} ({y}) at {level} level")
                                    continue

                                df['price'] = pd.to_numeric(df['Value'].str.replace(r'[^\d.]', '', regex=True), errors='coerce')
                                df['year'] = pd.to_numeric(df['year'], errors='coerce')
                                df['commodity'] = df['commodity_desc']
                                df['state'] = df.get('state_alpha', 'US')

                                all_results.append(df[['year', 'state', 'commodity', 'price']])
                                successful_requests += 1

                            except Exception as e:
                                logger.error(f"Final failure for {variant} in {st}/{county} ({y}) with {source_desc} at {level} level: {str(e)}")

                            time.sleep(request_delay)

    logger.info(f"✅ Attempted {attempted_requests} requests | ✅ Successful data pulls: {successful_requests}")
    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
