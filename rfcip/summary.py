# rfcip/summary.py
import pandas as pd
import requests
import tempfile
import warnings
import os
import time
import logging
import json
import hashlib
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from datetime import datetime
from .codes import get_crop_codes, get_insurance_plan_codes
from .utils import convert_state_to_fips, clean_fips, clean_column_names
from .helpers import get_sob_url, get_sobtpu_data
from .helpers import extract_first_text_file_from_zip

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def resolve_crop_variants(crop_input: Union[str, int, List[Union[str, int]]]) -> List[str]:
    try:
        crop_df = get_crop_codes()
        crop_df.columns = crop_df.columns.str.lower()
        crop_df['alias'] = crop_df['crop_name'].str.lower()

        if isinstance(crop_input, (str, int)):
            crop_input = [crop_input]

        resolved = []
        for item in crop_input:
            item_str = str(item).strip().lower()
            matches = crop_df[crop_df['alias'].str.contains(item_str, na=False)]
            if not matches.empty:
                resolved.extend(matches['crop_code'].astype(str).tolist())
            elif item_str.isdigit():
                resolved.append(item_str)
            else:
                logger.warning(f"Unrecognized crop: {item}")

        return list(set(resolved))[:5]

    except Exception as e:
        warnings.warn(f"Failed to resolve crop variants: {str(e)}")
        return []

def resolve_insurance_plan_variants(plan_input: Union[str, int, List[Union[str, int]]]) -> List[str]:
    try:
        plan_df = get_insurance_plan_codes()
        plan_df.columns = plan_df.columns.str.lower()

        if 'insurance_plan_abbrv' not in plan_df.columns or 'insurance_plan' not in plan_df.columns:
            raise ValueError("insurance_plan_abbrv or insurance_plan missing from plan code DataFrame")

        plan_df['alias'] = plan_df['insurance_plan_abbrv'].str.lower()
        plan_df['name'] = plan_df['insurance_plan'].str.lower()

        if isinstance(plan_input, (str, int)):
            plan_input = [plan_input]

        resolved = []
        for item in plan_input:
            item_str = str(item).strip().lower()
            matches = plan_df[plan_df['alias'].str.contains(item_str, na=False) | plan_df['name'].str.contains(item_str, na=False)]
            if not matches.empty:
                resolved.extend(matches['insurance_plan_code'].astype(str).tolist())
            elif item_str.isdigit():
                resolved.append(item_str)
            else:
                logger.warning(f"Unrecognized insurance plan: {item}")

        return list(set(resolved))[:5]

    except Exception as e:
        warnings.warn(f"Failed to resolve insurance plan variants: {str(e)}")
        return []


def _generate_cache_key(**params: Any) -> str:
    return hashlib.md5(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()

def get_summary_data(
    crop: Union[str, List[str]] = None,
    state: Union[str, List[str]] = None,
    year: Union[int, List[int]] = datetime.now().year,
    delivery_type: Optional[Union[str, List[str]]] = None,
    insurance_plan: Optional[Union[str, int, List[Union[str, int]]]] = None,
    county: Optional[Union[str, int]] = None,
    fips: Optional[Union[str, int]] = None,
    cov_lvl: Optional[Union[float, List[float]]] = None,
    comm_cat: str = "B",
    dest_file: Optional[str] = None,
    group_by: Optional[Union[str, List[str]]] = None,
    sob_version: str = "sob",
    cache_dir: str = "sob_cache",
    max_retries: int = 3,
    retry_delay: int = 5
) -> Optional[pd.DataFrame]:
    """
    Download USDA Summary of Business data with enhanced error handling
    
    Args:
        crop: Crop name(s) or code(s)
        state: State name, abbreviation, or FIPS
        year: Crop year(s) to retrieve
        delivery_type: "RBUP" (buyup) or "RCAT" (catastrophic)
        insurance_plan: Insurance plan name or code
        county: County name or 5-digit FIPS
        fips: Direct 5-digit county FIPS
        cov_lvl: Coverage level (0.5-0.95)
        comm_cat: "S" (standard), "L" (livestock), "B" (both)
        dest_file: Export path for Excel file
        group_by: Additional grouping columns
        sob_version: "sob" (standard) or "sobtpu" (detailed)
        cache_dir: Cache directory path
        max_retries: Maximum number of retry attempts for transient errors
        retry_delay: Delay between retry attempts in seconds
        
    Returns:
        DataFrame with summary of business data
    """
    # Validate inputs
    if isinstance(year, int):
        years = [year]
    else:
        years = year
        
    # Validate years
    current_year = datetime.now().year
    years = [y for y in years if 2000 <= y <= current_year + 1]
    if not years:
        warnings.warn(f"No valid years specified. Using current year: {current_year}")
        years = [current_year]
        
    # Warn about current year data availability
    recent_years = [y for y in years if y >= current_year]
    if recent_years and sob_version in ["sob", "sobtpu"]:
        years_str = ', '.join(map(str, recent_years))
        warnings.warn(f"Data for recent years ({years_str}) may not be fully available yet")
        
    valid_comm_cats = ["S", "L", "B"]
    if comm_cat not in valid_comm_cats:
        raise ValueError(f"comm_cat must be one of {valid_comm_cats}")
        
    # Create cache directory
    Path(cache_dir).mkdir(exist_ok=True, parents=True)
    
    # Initialize results
    all_data = []
    available_years = []  # Track years we successfully get data for
    
    # Handle SOBTPU version separately (process all years at once)
    if sob_version == "sobtpu":
        # SOBTPU data is only available from 2015 onward
        valid_sobtpu_years = [y for y in years if y >= 2015]
        if not valid_sobtpu_years:
            warnings.warn("SOBTPU data is only available from 2015 onward. Switching to standard SOB processing")
            sob_version = "sob"
        else:
            try:
                df = get_sobtpu_data(
                    year=valid_sobtpu_years,
                    crop=crop,
                    insurance_plan=insurance_plan,
                    state=state,
                    county=county,
                    fips=fips,
                    cov_lvl=cov_lvl,
                    cache_dir=cache_dir,
                    max_retries=max_retries,
                    retry_delay=retry_delay
                )
                if df is not None and not df.empty:
                    all_data.append(df)
                    available_years.extend(valid_sobtpu_years)
                else:
                    warnings.warn("SOBTPU data is empty. Switching to standard SOB processing")
                    sob_version = "sob"
            except Exception as e:
                warnings.warn(f"Failed to get SOBTPU data: {str(e)}. Switching to standard SOB processing")
                sob_version = "sob"
    
    # Process standard SOB data
    if sob_version == "sob":
        # First try using SOBTPU as a detailed source for standard SOB format (only for years >= 2015)
        sobtpu_years = [y for y in years if y >= 2015]
        if sobtpu_years:
            try:
                df = get_sobtpu_data(
                    year=sobtpu_years,
                    crop=crop,
                    insurance_plan=insurance_plan,
                    state=state,
                    county=county,
                    fips=fips,
                    cov_lvl=cov_lvl,
                    cache_dir=cache_dir,
                    max_retries=max_retries,
                    retry_delay=retry_delay
                )
                if df is not None and not df.empty:
                    # Aggregate to match standard SOB format
                    df = df.groupby([
                        'commodity_year', 'state_code', 'county_code', 'commodity_code',
                        'insurance_plan_code', 'coverage_level_percent'
                    ]).agg({
                        'liability_amount': 'sum',
                        'total_premium_amount': 'sum',
                        'subsidy_amount': 'sum',
                        'indemnity_amount': 'sum'
                    }).reset_index()
                    df = df.rename(columns={
                        'liability_amount': 'liability',
                        'total_premium_amount': 'total_premium',
                        'subsidy_amount': 'subsidy',
                        'indemnity_amount': 'indemnity'
                    })
                    all_data.append(df)
                    available_years.extend(sobtpu_years)
            except Exception as e:
                warnings.warn(f"Failed to get SOBTPU fallback data: {str(e)}")
        
        # Process all years with standard SOB method (including years < 2015)
        remaining_years = [y for y in years if y not in available_years]
        if not remaining_years:
            # If we already have all years from SOBTPU, skip standard processing
            pass
        else:
            for y in remaining_years:
                cache_file = None
                data_retrieved = False
                for attempt in range(max_retries + 1):
                    try:
                        # Generate cache key based on parameters
                        cache_key = _generate_cache_key(
                            year=y,
                            crop=crop,
                            delivery_type=delivery_type,
                            insurance_plan=insurance_plan,
                            state=state,
                            county=county,
                            fips=fips,
                            cov_lvl=cov_lvl,
                            comm_cat=comm_cat,
                            group_by=group_by
                        )
                        cache_file = Path(cache_dir) / f"sob_{cache_key}.parquet"
                        
                        # Check cache first
                        if cache_file.exists():
                            df = pd.read_parquet(cache_file)
                            if not df.empty:
                                all_data.append(df)
                                available_years.append(y)
                                data_retrieved = True
                                break
                                
                        # Fetch new data
                        df = _get_standard_sob_data(
                            year=y,
                            crop=crop,
                            delivery_type=delivery_type,
                            insurance_plan=insurance_plan,
                            state=state,
                            county=county,
                            fips=fips,
                            cov_lvl=cov_lvl,
                            comm_cat=comm_cat,
                            group_by=group_by,
                            max_retries=max_retries,
                            retry_delay=retry_delay
                        )
                        
                        if df is not None and not df.empty:
                            # Cache the data
                            df.to_parquet(cache_file)
                            all_data.append(df)
                            available_years.append(y)
                            data_retrieved = True
                            break
                        elif attempt == max_retries:
                            warnings.warn(f"No data available for year {y}")
                            break
                            
                    except Exception as e:
                        if attempt < max_retries:
                            logger.warning(f"Attempt {attempt+1} failed for year {y}: {str(e)}")
                            time.sleep(retry_delay)
                        else:
                            warnings.warn(f"Error processing year {y}: {str(e)}")
                            # Remove potentially corrupt cache file
                            if cache_file and cache_file.exists():
                                try:
                                    cache_file.unlink()
                                except Exception:
                                    pass
    
    # Process and return combined data
    result = _process_combined_data(all_data, dest_file)
    
    # Warn if we didn't get data for all requested years
    missing_years = set(years) - set(available_years)
    if missing_years:
        years_str = ', '.join(map(str, sorted(missing_years)))
        warnings.warn(f"Could not retrieve data for the following years: {years_str}")
        
    return result


def _process_combined_data(all_data: List[pd.DataFrame], dest_file: Optional[str]) -> Optional[pd.DataFrame]:
    """Helper function to process and output combined data"""
    if not all_data:
        warnings.warn("No data retrieved for specified parameters")
        return pd.DataFrame()
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Convert data types
    numeric_cols = [
        'commodity_year', 'commodity_code', 'insurance_plan_code', 
        'state_code', 'county_code', 'coverage_level_percent',
        'policies_earning_premium', 'net_planted_quantity',
        'net_endorsed_acres', 'liability', 'total_premium',
        'subsidy', 'producer_premium', 'indemnity'
    ]
    
    for col in numeric_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors='coerce')
    
    # Export or return
    if dest_file:
        # Create directory if needed
        dest_path = Path(dest_file)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        combined.to_excel(dest_file, index=False)
        return None
    else:
        return combined
        

def _generate_cache_key(**params: Any) -> str:
    """Generate unique cache key based on parameters"""
    # Create hashable representation of parameters
    param_str = json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(param_str.encode()).hexdigest()

def _get_standard_sob_data(
    year: int,
    crop: Optional[Union[str, int, List[Union[str, int]]]] = None,
    delivery_type: Optional[Union[str, List[str]]] = None,
    insurance_plan: Optional[Union[str, int, List[Union[str, int]]]] = None,
    state: Optional[Union[str, int]] = None,
    county: Optional[Union[str, int]] = None,
    fips: Optional[Union[str, int]] = None,
    cov_lvl: Optional[Union[float, List[float]]] = None,
    comm_cat: str = "B",
    group_by: Optional[Union[str, List[str]]] = None,
    max_retries: int = 3,
    retry_delay: int = 5
) -> pd.DataFrame:
    state_fips = None
    if state:
        try:
            if isinstance(state, list):
                state_fips = [convert_state_to_fips(s) for s in state[:5]]
            else:
                state_fips = [convert_state_to_fips(state)]
        except ValueError as e:
            warnings.warn(f"Invalid state identifier: {str(e)}")
            return pd.DataFrame()

    clean_fips_code = None
    if fips:
        try:
            clean_fips_code = clean_fips(fips)
        except ValueError as e:
            warnings.warn(str(e))
            return pd.DataFrame()

    cov_percent = None
    if cov_lvl:
        if isinstance(cov_lvl, float):
            cov_percent = [int(cov_lvl * 100)]
        elif isinstance(cov_lvl, list):
            valid_cov = [int(cl * 100) for cl in cov_lvl if 0.5 <= cl <= 0.95][:5]
            cov_percent = valid_cov if valid_cov else None

    crop_codes = resolve_crop_variants(crop)
    plan_codes = resolve_insurance_plan_variants(insurance_plan)

    try:
        url = get_sob_url(
            year=[year],
            crop=crop_codes,
            delivery_type=delivery_type,
            insurance_plan=plan_codes,
            state=state_fips,
            county=county,
            fips=clean_fips_code,
            cov_lvl=cov_percent,
            comm_cat=comm_cat,
            group_by=group_by
        )
        logger.info(f"Requesting URL: {url}")
    except Exception as e:
        warnings.warn(f"Failed to construct URL: {str(e)}")
        return pd.DataFrame()

    for attempt in range(max_retries + 1):
        tmp_path = None
        try:
            response = requests.get(url, timeout=30)

            if response.status_code == 500:
                if year >= datetime.now().year:
                    warnings.warn(f"Server error 500 for year {year}. This year may not be available yet.")
                else:
                    warnings.warn(f"Server error 500 for year {year}. Data may not be available.")
                return pd.DataFrame()

            response.raise_for_status()

            if not response.content:
                warnings.warn("Empty response content")
                return pd.DataFrame()

            with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name

            try:
                df = pd.read_excel(tmp_path)
            except Exception as e:
                try:
                    df = pd.read_excel(tmp_path, skiprows=1)
                except:
                    try:
                        df = pd.read_csv(tmp_path)
                    except:
                        warnings.warn(f"Failed to read downloaded file: {str(e)}")
                        return pd.DataFrame()

            df = clean_column_names(df)

            if 'commodity_year' not in df.columns:
                df['commodity_year'] = year

            return df

        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                logger.warning(f"Attempt {attempt+1} failed: {str(e)} - Retrying in {retry_delay} seconds")
                time.sleep(retry_delay)
            else:
                warnings.warn(f"Network error after {max_retries} attempts: {str(e)}")
        except Exception as e:
            if attempt < max_retries:
                logger.warning(f"Attempt {attempt+1} failed: {str(e)} - Retrying in {retry_delay} seconds")
                time.sleep(retry_delay)
            else:
                warnings.warn(f"Error processing data after {max_retries} attempts: {str(e)}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass

    return pd.DataFrame()
