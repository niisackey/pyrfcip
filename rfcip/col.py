import pandas as pd
import zipfile
import io
import os
from datetime import datetime
from typing import List, Union
import logging
from tqdm import tqdm
import tempfile
from .helpers import (
    locate_col_links,
    download_and_verify
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


COL_COLUMNS = [
    "commodity_year", "state_code", "state_abbrv", "county_code", "county_name",
    "commodity_code", "commodity_name", "insurance_plan_code", "insurance_plan_abbrv",
    "delivery_type", "stage_code", "col_code", "col_name", "month_of_loss_code",
    "month_of_loss_name", "year_of_loss", "policies_earning_prem", "policies_indemnified",
    "net_planted_qty", "net_endorsed_acres", "liability", "total_premium",
    "producer_paid_premium", "subsidy", "state_subsidy", "addnl_subsidy",
    "efa_prem_discount", "indemnified_quantity", "indem_amount", "loss_ratio"
]


NUMERIC_COLS = [
    "commodity_year", "state_code", "county_code", "commodity_code", 
    "insurance_plan_code", "col_code", "month_of_loss_code", "year_of_loss", 
    "policies_earning_prem", "policies_indemnified", "net_planted_qty", 
    "net_endorsed_acres", "liability", "total_premium", "producer_paid_premium", 
    "subsidy", "state_subsidy", "addnl_subsidy", "efa_prem_discount", 
    "indemnified_quantity", "indem_amount", "loss_ratio"
]

# Define string columns for trimming
STRING_COLS = [
    "state_abbrv", "county_name", "commodity_name", "insurance_plan_abbrv",
    "delivery_type", "stage_code", "col_name", "month_of_loss_name"
]

def get_col_data(
    year: Union[int, List[int]] = None,
    cache_dir: str = "col_cache",
    max_retries: int = 3
) -> pd.DataFrame:
    """
    Download and process USDA Cause of Loss (COL) data
    
    Args:
        year: Single year or list of years to retrieve (default: last 5 years)
        cache_dir: Directory to cache downloaded files
        max_retries: Number of download retry attempts
        
    Returns:
        Processed DataFrame with cause of loss data
    """
    # Set default years (current year and previous 4 years)
    if year is None:
        current_year = datetime.now().year
        years = list(range(current_year - 4, current_year + 1))
    elif isinstance(year, int):
        years = [year]
    else:
        years = year
        
    # Validate years
    current_year = datetime.now().year
    valid_years = list(range(1989, current_year + 1))  # USDA COL data available since 1989
    invalid_years = [y for y in years if y not in valid_years]
    
    if invalid_years:
        logger.warning("Invalid years detected: %s. Using valid range: 1989-%d", 
                      invalid_years, current_year)
        years = [y for y in years if y in valid_years]
        
    if not years:
        raise ValueError("No valid years specified")
    
    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)
    
    # Prepare for data collection
    all_data = []
    failed_years = []
    
    # Get COL links from helper function
    try:
        links_df = locate_col_links()
        logger.info("Retrieved COL download links")
    except Exception as e:
        logger.error("Failed to retrieve COL download links: %s", e)
        return pd.DataFrame()
    
    # Download and process data
    logger.info("Downloading COL files for %d year(s)", len(years))
    for y in tqdm(years, desc="Processing COL Data"):
        # Check cache first
        cache_file = os.path.join(cache_dir, f"col_{y}.parquet")
        if os.path.exists(cache_file):
            try:
                df = pd.read_parquet(cache_file)
                all_data.append(df)
                continue
            except Exception as e:
                logger.warning("Failed to read cache: %s", e)
        
        # Get URL from links DataFrame
        if y not in links_df['year'].values:
            logger.warning("No download URL found for year %d", y)
            failed_years.append(y)
            continue
            
        url = links_df[links_df['year'] == y]['url'].values[0]
        df = None
        
        for attempt in range(max_retries):
            try:
                # Create temp file for download
                with tempfile.NamedTemporaryFile(suffix=".zip") as tmp_zip:
                    # Download and verify
                    download_and_verify(url, tmp_zip.name)
                    
                    # Process ZIP content
                    with zipfile.ZipFile(tmp_zip.name) as zip_ref:
                        # Find the data file (should be a .txt file)
                        txt_files = [f for f in zip_ref.namelist() if f.lower().endswith(".txt")]
                        if not txt_files:
                            logger.warning("No text file found in ZIP for %d", y)
                            continue
                        
                        with zip_ref.open(txt_files[0]) as txt_file:
                            # Read pipe-delimited file
                            df = pd.read_csv(
                                txt_file, 
                                sep="|", 
                                header=None, 
                                dtype=str,
                                encoding="latin1",
                                on_bad_lines="warn"
                            )
                
                # Check if we have the expected number of columns
                if len(df.columns) != len(COL_COLUMNS):
                    logger.warning("Column count mismatch. Expected %d, got %d for year %d", 
                                  len(COL_COLUMNS), len(df.columns), y)
                    
                    # Handle extra columns
                    if len(df.columns) > len(COL_COLUMNS):
                        df = df.iloc[:, :len(COL_COLUMNS)]
                    # Handle missing columns
                    else:
                        for i in range(len(COL_COLUMNS) - len(df.columns)):
                            df[f'extra_col_{i}'] = None
                
                # Apply column names
                df.columns = COL_COLUMNS
                
                # Add year identifier
                df["data_year"] = y
                
                # Save to cache
                df.to_parquet(cache_file)
                
                all_data.append(df)
                break
                
            except Exception as e:
                logger.warning("Attempt %d failed for %d: %s", attempt+1, y, str(e))
                if attempt == max_retries - 1:
                    failed_years.append(y)
    
    # Handle no data case
    if not all_data:
        logger.error("No COL data retrieved for specified years")
        return pd.DataFrame()
    
    # Combine all data
    logger.info("Merging COL files")
    combined = pd.concat(all_data, ignore_index=True)
    
    # Convert data types
    logger.info("Processing data types")
    
    # Numeric columns
    for col in NUMERIC_COLS:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    
    # Clean string columns
    for col in STRING_COLS:
        if col in combined.columns:
            combined[col] = combined[col].str.strip()
    
    # Report results
    logger.info("Retrieved %d COL records for %d-%d", 
               len(combined), min(years), max(years))
    
    if failed_years:
        logger.warning("Failed to retrieve data for years: %s", failed_years)
    
    return combined