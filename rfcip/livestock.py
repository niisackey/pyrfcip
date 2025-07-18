import pandas as pd
import zipfile
import os
from datetime import datetime
from typing import List, Optional, Union
import tempfile
from tqdm import tqdm
import logging
from .helpers import (
    locate_livestock_links,
    download_and_verify
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define column mappings for each program
COLUMN_MAPPING = {
    "LRP": {
        "columns": [
            "reinsurance_year", "commodity_year", "location_state_code", "location_state_abbreviation",
            "location_county_code", "location_county_name", "commodity_code", "commodity_name",
            "insurance_plan_code", "insurance_plan_name", "type_code", "type_code_name",
            "practice_code", "practice_code_name", "sales_effective_date", "endorsement_length",
            "coverage_price", "expected_end_value", "coverage_level_percent", "rate",
            "cost_per_cwt", "end_date", "endorsements_earning_premium", "endorsements_indemnified",
            "net_number_of_head", "total_weight", "subsidy_amount", "total_premium_amount",
            "producer_premium_amount", "liability_amount", "indemnity_amount", "insurance_plan_abbreviation"
        ],
        "numeric": [
            "reinsurance_year", "commodity_year", "location_state_code", "location_county_code",
            "commodity_code", "insurance_plan_code", "endorsement_length", "coverage_price",
            "expected_end_value", "coverage_level_percent", "rate", "cost_per_cwt",
            "endorsements_earning_premium", "endorsements_indemnified", "net_number_of_head",
            "total_weight", "subsidy_amount", "total_premium_amount", "producer_premium_amount",
            "liability_amount", "indemnity_amount", "type_code", "practice_code"
        ],
        "dates": ["sales_effective_date", "end_date"]
    },
    "DRP": {
        "columns": [
            "reinsurance_year", "commodity_year", "location_state_code", "location_state_abbreviation",
            "location_county_code", "location_county_name", "commodity_code", "commodity_name",
            "insurance_plan_code", "insurance_plan_name", "coverage_type_code", "coverage_type_description",
            "type_code", "type_code_name", "practice_code", "practice_code_name", "sales_effective_date",
            "coverage_level_percent", "protection_factor", "class_price_weighting_factor",
            "component_price_weighting_factor", "declared_butterfat_test", "declared_protein_test",
            "endorsements_earning_premium", "endorsements_indemnified", "net_declared_covered_milk_production",
            "subsidy_amount", "total_premium_amount", "producer_premium_amount", "liability_amount",
            "indemnity_amount", "insurance_plan_abbreviation"
        ],
        "numeric": [
            "reinsurance_year", "commodity_year", "location_state_code", "location_county_code",
            "commodity_code", "insurance_plan_code", "type_code", "practice_code", "coverage_level_percent",
            "protection_factor", "class_price_weighting_factor", "component_price_weighting_factor",
            "declared_butterfat_test", "declared_protein_test", "endorsements_earning_premium",
            "endorsements_indemnified", "net_declared_covered_milk_production", "subsidy_amount",
            "total_premium_amount", "producer_premium_amount", "liability_amount", "indemnity_amount"
        ],
        "dates": ["sales_effective_date"]
    },
    "LGM": {
        "columns": [
            "reinsurance_year", "commodity_year", "location_state_code", "location_state_abbreviation",
            "location_county_code", "location_county_name", "commodity_code", "commodity_name",
            "insurance_plan_code", "insurance_plan_name", "type_code", "type_code_name",
            "practice_code", "practice_code_name", "sales_effective_date", "target_marketings_1",
            "target_marketings_2", "target_marketings_3", "target_marketings_4", "target_marketings_5",
            "target_marketings_6", "target_marketings_7", "target_marketings_8", "target_marketings_9",
            "target_marketings_10", "target_marketings_11", "corn_equivalent_2", "corn_equivalent_3",
            "corn_equivalent_4", "corn_equivalent_5", "corn_equivalent_6", "corn_equivalent_7",
            "corn_equivalent_8", "corn_equivalent_9", "corn_equivalent_10", "corn_equivalent_11",
            "soybean_meal_equivalent_2", "soybean_meal_equivalent_3", "soybean_meal_equivalent_4",
            "soybean_meal_equivalent_5", "soybean_meal_equivalent_6", "soybean_meal_equivalent_7",
            "soybean_meal_equivalent_8", "soybean_meal_equivalent_9", "soybean_meal_equivalent_10",
            "soybean_meal_equivalent_11", "endorsements_earning_premium", "endorsements_indemnified",
            "deductible", "live_cattle_target_weight_quantity", "feeder_cattle_target_weight_quantity",
            "corn_target_weight_quantity", "liability_amount", "total_premium_amount", "subsidy_amount",
            "producer_premium_amount", "indemnity_amount", "insurance_plan_abbreviation"
        ],
        "numeric": [
            "reinsurance_year", "commodity_year", "location_state_code", "location_county_code",
            "commodity_code", "insurance_plan_code", "type_code", "practice_code", "target_marketings_1",
            "target_marketings_2", "target_marketings_3", "target_marketings_4", "target_marketings_5",
            "target_marketings_6", "target_marketings_7", "target_marketings_8", "target_marketings_9",
            "target_marketings_10", "target_marketings_11", "corn_equivalent_2", "corn_equivalent_3",
            "corn_equivalent_4", "corn_equivalent_5", "corn_equivalent_6", "corn_equivalent_7",
            "corn_equivalent_8", "corn_equivalent_9", "corn_equivalent_10", "corn_equivalent_11",
            "soybean_meal_equivalent_2", "soybean_meal_equivalent_3", "soybean_meal_equivalent_4",
            "soybean_meal_equivalent_5", "soybean_meal_equivalent_6", "soybean_meal_equivalent_7",
            "soybean_meal_equivalent_8", "soybean_meal_equivalent_9", "soybean_meal_equivalent_10",
            "soybean_meal_equivalent_11", "endorsements_earning_premium", "endorsements_indemnified",
            "deductible", "live_cattle_target_weight_quantity", "feeder_cattle_target_weight_quantity",
            "corn_target_weight_quantity", "liability_amount", "total_premium_amount", "subsidy_amount",
            "producer_premium_amount", "indemnity_amount"
        ],
        "dates": ["sales_effective_date"]
    }
}

def get_livestock_data(
    year: Union[int, List[int]] = datetime.now().year,
    program: str = "LRP",
    cache_dir: str = "livestock_cache",
    max_retries: int = 3
) -> pd.DataFrame:
    """
    Download and process USDA livestock insurance data
    
    Args:
        year: Single year or list of years to retrieve
        program: Insurance program (DRP, LGM, or LRP)
        cache_dir: Directory to cache downloaded files
        max_retries: Number of download retry attempts
        
    Returns:
        Processed DataFrame with livestock insurance data
    """
    # Validate inputs
    program = program.upper()
    valid_programs = ["DRP", "LGM", "LRP"]
    if program not in valid_programs:
        raise ValueError(f"Invalid program. Must be one of {valid_programs}")
    
    if isinstance(year, int):
        years = [year]
    else:
        years = year
    
    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)
    
    # Locate download URLs
    logger.info("Locating livestock download links")
    try:
        links_df = locate_livestock_links()
        logger.info("Retrieved livestock download links")
    except Exception as e:
        logger.error("Failed to retrieve livestock download links: %s", e)
        return pd.DataFrame()
    
    # Filter to requested program and years
    program_df = links_df[links_df["program"] == program]
    if program_df.empty:
        logger.warning("No download URLs found for program %s", program)
        return pd.DataFrame()
    
    # Prepare for data collection
    all_data = []
    failed_years = []
    
    # Download and process data
    logger.info("Downloading livestock files for %s program (%d years)", program, len(years))
    for y in tqdm(years, desc=f"Processing {program} Data"):
        # Check cache first
        cache_file = os.path.join(cache_dir, f"{program}_{y}.parquet")
        if os.path.exists(cache_file):
            try:
                df = pd.read_parquet(cache_file)
                all_data.append(df)
                continue
            except Exception as e:
                logger.warning("Failed to read cache: %s", e)
        
        # Get URL for this year and program
        program_year_df = program_df[program_df["year"] == y]
        if program_year_df.empty:
            logger.warning("No download URL found for %s %d", program, y)
            failed_years.append(y)
            continue
            
        url = program_year_df["url"].values[0]
        df = None
        
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
                        logger.warning("No text file found in ZIP for %s %d", program, y)
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
            mapping = COLUMN_MAPPING[program]
            expected_cols = len(mapping["columns"])
            if len(df.columns) != expected_cols:
                logger.warning("Column count mismatch. Expected %d, got %d for %s %d", 
                              expected_cols, len(df.columns), program, y)
                
                # Handle extra columns
                if len(df.columns) > expected_cols:
                    df = df.iloc[:, :expected_cols]
                # Handle missing columns
                else:
                    for i in range(expected_cols - len(df.columns)):
                        df[f'extra_col_{i}'] = None
            
            # Apply column names
            df.columns = mapping["columns"]
            
            # Add program and year identifiers
            df["program"] = program
            df["data_year"] = y
            
            # Save to cache
            df.to_parquet(cache_file)
            
            all_data.append(df)
            
        except Exception as e:
            logger.error("Failed to process %s %d: %s", program, y, str(e))
            failed_years.append(y)
    
    # Handle no data case
    if not all_data:
        logger.error("No livestock data retrieved for specified years")
        return pd.DataFrame()
    
    # Combine all data
    logger.info("Merging livestock files")
    combined = pd.concat(all_data, ignore_index=True)
    
    # Apply data types
    logger.info("Processing data types for %s", program)
    mapping = COLUMN_MAPPING[program]
    
    # Convert data types
    # Numeric columns
    for col in mapping["numeric"]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    
    # Date columns
    for col in mapping["dates"]:
        if col in combined.columns:
            combined[col] = pd.to_datetime(combined[col], errors="coerce")
    
    # Clean string columns
    string_cols = combined.select_dtypes(include=["object"]).columns
    for col in string_cols:
        combined[col] = combined[col].str.strip()
    
    # Report results
    retrieved_years = combined["data_year"].unique()
    logger.info("Retrieved %d records for %s program (%d-%d)", 
               len(combined), program, min(retrieved_years), max(retrieved_years))
    
    if failed_years:
        logger.warning("Failed to retrieve data for years: %s", failed_years)
    
    return combined