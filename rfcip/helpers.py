# rfcip/helpers.py
import os
import re
import zipfile
import tempfile
import requests
import pandas as pd
import warnings
import time
import logging
from bs4 import BeautifulSoup
import hashlib
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from tqdm import tqdm
from .utils import convert_state_to_fips, clean_fips, clean_column_names
from .codes import get_crop_codes, get_insurance_plan_codes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------- FILE DOWNLOAD HELPERS -------------------
def download_and_verify(
    url: str,
    dest_path: Optional[str] = None,
    expected_hash: Optional[str] = None,
    hash_type: str = "sha256",
    max_retries: int = 3,
    retry_delay: int = 5
) -> Union[bool, bytes]:
    """
    Download a file with retries and optional hash verification
    
    Args:
        url: URL to download
        dest_path: Destination path (None returns content)
        expected_hash: Expected file hash
        hash_type: 'md5' or 'sha256'
        max_retries: Number of retry attempts
        retry_delay: Delay between retry attempts in seconds
        
    Returns:
        True if successful and dest_path provided, 
        bytes content if dest_path is None and successful,
        False otherwise
    """
    for attempt in range(max_retries + 1):
        try:
            # Create a temporary directory
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = os.path.join(tmp_dir, "download.tmp")
                
                # Download file
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                with open(tmp_path, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, 
                              desc=f"Downloading {url.split('/')[-1]}") as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                
                # Verify hash if provided
                if expected_hash:
                    file_hash = calculate_file_hash(tmp_path, hash_type)
                    if file_hash != expected_hash.lower():
                        logger.warning(f"Hash mismatch: expected {expected_hash}, got {file_hash}")
                        if attempt < max_retries:
                            time.sleep(retry_delay)
                            continue
                        else:
                            return False
                
                # Move to final destination or return content
                if dest_path:
                    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
                    
                    # Windows-safe file move
                    if os.path.exists(dest_path):
                        os.unlink(dest_path)
                    os.rename(tmp_path, dest_path)
                    return True
                else:
                    with open(tmp_path, 'rb') as f:
                        return f.read()
            return True
        
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                logger.warning(f"Attempt {attempt+1} failed: {str(e)} - Retrying in {retry_delay} seconds")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to download after {max_retries} attempts: {str(e)}")
                return False
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return False
    
    return False

def extract_first_text_file_from_zip(zip_path: str) -> pd.DataFrame:
    """
    Extracts and reads the first text or csv-like file from the ZIP archive.
    Supports .txt, .dat, .csv and logs warning if no valid file found.
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            print("[DEBUG] ZIP Contents:", file_list)

            # Find the first valid text file
            for file_name in file_list:
                if file_name.lower().endswith(('.txt', '.dat', '.csv')):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as tmp_file:
                        tmp_file.write(zip_ref.read(file_name))
                        tmp_path = tmp_file.name

                    try:
                        df = pd.read_csv(tmp_path, sep='|', dtype=str)
                        os.unlink(tmp_path)
                        return df
                    except Exception as e:
                        warnings.warn(f"Failed to parse extracted file: {file_name}: {str(e)}")
                        os.unlink(tmp_path)
                        return pd.DataFrame()

            warnings.warn("No usable .txt/.csv file found in ZIP")
            return pd.DataFrame()
    except Exception as e:
        warnings.warn(f"ZIP extraction error: {str(e)}")
        return pd.DataFrame()


def get_sobtpu_data_from_zip(zip_path: str) -> pd.DataFrame:
    """
    Use this function inside get_sobtpu_data() to extract ZIP content using fallback.
    """
    df = extract_first_text_file_from_zip(zip_path)
    if df.empty:
        warnings.warn("SOBTPU fallback: no data parsed from extracted ZIP file")
    return df

def calculate_file_hash(file_path: str, hash_type: str = "sha256") -> str:
    """
    Calculate file hash
    
    Args:
        file_path: Path to file
        hash_type: 'md5' or 'sha256'
        
    Returns:
        Hexadecimal digest of file hash
    """
    hash_func = hashlib.sha256() if hash_type == "sha256" else hashlib.md5()
    
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


# ------------------- DATA LOCATION HELPERS -------------------
def locate_col_links(url: str = "https://www.rma.usda.gov/tools-reports/summary-of-business/cause-loss") -> pd.DataFrame:
    """
    Find cause of loss data download links
    
    Returns:
        DataFrame with url and year columns
    """
    return _locate_data_links(
        base_url=url,
        pattern=r"cause-loss|colsom",
        year_pattern=r"\d{4}"
    )

def locate_sobtpu_links(url: str = "https://www.rma.usda.gov/tools-reports/summary-of-business/state-county-crop-summary-business") -> pd.DataFrame:
    """
    Find SOBTPU data download links
    
    Returns:
        DataFrame with url and year columns
    """
    return _locate_data_links(
        base_url=url,
        pattern=r"sobtpu",
        exclude_patterns=[r"\.pdf", r"\.docx"],
        year_pattern=r"\d{4}"
    )

def locate_livestock_links(url: str = "https://www.rma.usda.gov/tools-reports/summary-of-business/livestock-dairy-participation") -> pd.DataFrame:
    """
    Find livestock data download links
    
    Returns:
        DataFrame with url, program, and year columns
    """
    # Get all links
    soup = _get_soup(url)
    if soup is None:
        return pd.DataFrame(columns=["url", "program", "year"])
    
    links = [a['href'] for a in soup.find_all('a', href=True)]
    
    # Filter to relevant links
    filtered = []
    for link in links:
        if any(p in link for p in ["drp_", "lgm_", "lrp_"]) and link.endswith(".zip"):
            # Extract program and year
            program = None
            if "drp_" in link: program = "DRP"
            if "lgm_" in link: program = "LGM"
            if "lrp_" in link: program = "LRP"
            
            year_match = re.search(r"_(\d{4})\.", link)
            year = int(year_match.group(1)) if year_match else None
            
            if program and year:
                filtered.append({
                    "url": link,
                    "program": program,
                    "year": year
                })
    
    return pd.DataFrame(filtered)

def _locate_data_links(
    base_url: str,
    pattern: str,
    exclude_patterns: Optional[List[str]] = None,
    year_pattern: str = r"\d{4}",
    max_retries: int = 3,
    retry_delay: int = 5
) -> pd.DataFrame:
    """Common link location logic with retry support"""
    for attempt in range(max_retries + 1):
        try:
            soup = _get_soup(base_url)
            if soup is None:
                return pd.DataFrame(columns=["url", "year"])
            
            links = [a['href'] for a in soup.find_all('a', href=True)]
            
            # Filter links
            filtered = []
            for link in links:
                if re.search(pattern, link, re.IGNORECASE):
                    # Apply exclusion patterns
                    if exclude_patterns:
                        if any(re.search(ep, link) for ep in exclude_patterns):
                            continue
                    
                    # Extract year
                    year_match = re.search(year_pattern, link)
                    year = int(year_match.group()) if year_match else None
                    
                    # Handle links that might be index pages
                    if not link.endswith(".zip"):
                        # Follow to actual download page
                        try:
                            sub_soup = _get_soup(f"https://www.rma.usda.gov{link}")
                            zip_links = [a['href'] for a in sub_soup.find_all('a', href=True) 
                                        if a['href'].endswith(".zip")]
                            if zip_links:
                                filtered.append({"url": zip_links[0], "year": year})
                        except:
                            continue
                    else:
                        filtered.append({"url": link, "year": year})
            
            return pd.DataFrame(filtered)
        
        except Exception as e:
            if attempt < max_retries:
                logger.warning(f"Attempt {attempt+1} failed: {str(e)} - Retrying in {retry_delay} seconds")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to locate links after {max_retries} attempts: {str(e)}")
                return pd.DataFrame(columns=["url", "year"])

def _get_soup(url: str, max_retries: int = 3, retry_delay: int = 5) -> Optional[BeautifulSoup]:
    """Get BeautifulSoup object for URL with retry support"""
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                logger.warning(f"Attempt {attempt+1} failed: {str(e)} - Retrying in {retry_delay} seconds")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to get page after {max_retries} attempts: {str(e)}")
                return None
    return None

# ------------------- URL CONSTRUCTION HELPERS -------------------
def include_and(url: str) -> str:
    """Add & separator to URL if needed"""
    if not url.endswith("?"):
        return url + "&"
    return url

# rfcip/helpers.py (update to get_sob_url function)
def get_sob_url(
    year: List[int],
    crop: Optional[List[str]] = None,
    delivery_type: Optional[List[str]] = None,
    insurance_plan: Optional[List[str]] = None,
    state: Optional[List[str]] = None,
    county: Optional[List[str]] = None,
    fips: Optional[str] = None,
    cov_lvl: Optional[List[float]] = None,
    comm_cat: str = "B",
    group_by: Optional[List[str]] = None,
    max_parameters: int = 5
) -> str:
    """
    Generate URL for Summary of Business data with conservative parameters
    to avoid triggering server errors.
    """
    base_url = "https://public-rma.fpac.usda.gov/apps/SummaryOfBusiness/ReportGenerator/ExportToExcel?"
    
    # Minimal set of columns that are known to work reliably
    suffix = "&VisibleColumns=CommodityYear,CommodityName,CommodityCode,LocationStateAbbreviation,LocationCountyName,InsurancePlanAbbreviation,CoverageLevelPercent,LiabilityAmount,TotalPremiumAmount,SubsidyAmount,IndemnityAmount"
    
    params = {"CC": comm_cat}
    
    # Year parameter
    if year:
        params["CY"] = ",".join(map(str, year[:max_parameters]))
    
    # Crop parameter - use only first crop to minimize complexity
    if crop:
        crop_df = get_crop_codes()
        # Use only the first crop
        c = crop[0] if isinstance(crop, list) else crop
        crop_code = None
        
        # Try to match by name
        name_match = crop_df[crop_df['CROP_NAME'].str.lower() == str(c).lower()]
        if not name_match.empty:
            crop_code = name_match['CROP_CODE'].iloc[0]
        else:
            # Try to match by code
            try:
                code_match = crop_df[crop_df['CROP_CODE'] == str(c)]
                if not code_match.empty:
                    crop_code = code_match['CROP_CODE'].iloc[0]
            except:
                pass
        
        if crop_code:
            params["CM"] = crop_code
    
    # Insurance plan - use only first plan
    if insurance_plan:
        plan_df = get_insurance_plan_codes()
        # Use only the first plan
        p = insurance_plan[0] if isinstance(insurance_plan, list) else insurance_plan
        plan_code = None
        
        # Try to match by abbreviation
        abbr_match = plan_df[plan_df['PLAN_ABBR'].str.lower() == str(p).lower()]
        if not abbr_match.empty:
            plan_code = abbr_match['insurance_plan_code'].iloc[0]
        else:
            # Try to match by name
            name_match = plan_df[plan_df['PLAN_NAME'].str.lower() == str(p).lower()]
            if not name_match.empty:
                plan_code = name_match['insurance_plan_code'].iloc[0]
        
        if plan_code:
            params["IP"] = plan_code
    
    # State parameter - use only first state
    if state:
        try:
            # Use only the first state
            s = state[0] if isinstance(state, list) else state
            state_fips = convert_state_to_fips(s)
            params["ST"] = state_fips
        except ValueError as e:
            logger.warning(f"State conversion error: {str(e)}")
    
    # County and FIPS - use only one location identifier
    if county:
        # Use only the first county
        params["CT"] = county[0] if isinstance(county, list) else county
    elif fips:
        try:
            cleaned_fips = clean_fips(fips)
            params["CT"] = cleaned_fips
        except ValueError as e:
            logger.warning(str(e))
    
    # Coverage level - use only first level
    if cov_lvl:
        # Convert to percentage (50 for 0.5, etc.)
        cl = cov_lvl[0] if isinstance(cov_lvl, list) else cov_lvl
        cov_percent = int(cl * 100)
        params["CVL"] = str(cov_percent)
    
    # Delivery type - use only first type
    if delivery_type:
        # Use only the first delivery type
        params["DT"] = delivery_type[0] if isinstance(delivery_type, list) else delivery_type
    
    # Minimal grouping parameters
    ord_params = []
    if crop:
        ord_params.append("CM")
    if state:
        ord_params.append("ST")
    
    if ord_params:
        params["ORD"] = ",".join(ord_params)
    
    # Construct full URL
    param_str = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
    full_url = f"{base_url}{param_str}{suffix}"
    logger.info(f"Generated URL: {full_url}")
    return full_url

# ------------------- VALIDATION HELPERS -------------------
def valid_crop(crop: Union[str, int]) -> bool:
    """Check if crop is valid"""
    crop_df = get_crop_codes()
    if isinstance(crop, str):
        return crop.lower() in crop_df['CROP_NAME'].str.lower().values
    elif isinstance(crop, int):
        return str(crop) in crop_df['CROP_CODE'].astype(str).values
    return False

def valid_state(state: Union[str, int]) -> bool:
    """Check if state is valid"""
    try:
        convert_state_to_fips(state)
        return True
    except ValueError:
        return False

# ------------------- SOBTPU DATA HELPER -------------------
def get_sobtpu_data(
    year: list,
    crop: list = None,
    insurance_plan: list = None,
    state: list = None,
    county: list = None,
    fips: str = None,
    cov_lvl: list = None,
    cache_dir: str = "sobtpu_cache",
    max_retries: int = 3,
    retry_delay: int = 5
) -> pd.DataFrame:
    """
    Download and parse SOBTPU data from USDA ZIP archive using fallback logic
    """
    os.makedirs(cache_dir, exist_ok=True)
    all_years_data = []

    for y in year:
        url = f"https://public-rma.fpac.usda.gov/apps/SummaryOfBusiness/ReportGenerator/SobtpuData/{y}/sobtpu_{y}.zip"
        zip_path = os.path.join(cache_dir, f"sobtpu_{y}.zip")

        if not os.path.exists(zip_path):
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                with open(zip_path, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded: {zip_path}")
            except Exception as e:
                warnings.warn(f"Failed to download ZIP for {y}: {str(e)}")
                continue

        df = get_sobtpu_data_from_zip(zip_path)
        if df.empty:
            continue

        # Normalize column names
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        df['commodity_year'] = y

        # Filter crops
        if crop:
            crop_df = get_crop_codes()
            crop_list = crop[:5]
            name_mask = crop_df['CROP_NAME'].str.lower().isin([c.lower() for c in crop_list])
            crop_codes = crop_df.loc[name_mask, 'CROP_CODE'].unique().tolist()
            if not crop_codes:
                try:
                    code_mask = crop_df['CROP_CODE'].astype(str).isin([str(c) for c in crop_list])
                    crop_codes = crop_df.loc[code_mask, 'CROP_CODE'].unique().tolist()
                except: pass
            if crop_codes:
                df = df[df['commodity_code'].astype(str).isin(map(str, crop_codes))]

        # Filter insurance plan
        if insurance_plan:
            plan_df = get_insurance_plan_codes()
            plan_list = insurance_plan[:5]
            abbr_mask = plan_df['PLAN_ABBR'].str.lower().isin([p.lower() for p in plan_list])
            plan_codes = plan_df.loc[abbr_mask, 'insurance_plan_code'].unique().tolist()
            if not plan_codes:
                name_mask = plan_df['PLAN_NAME'].str.lower().isin([p.lower() for p in plan_list])
                plan_codes = plan_df.loc[name_mask, 'insurance_plan_code'].unique().tolist()
            if plan_codes:
                df = df[df['insurance_plan_code'].astype(str).isin(map(str, plan_codes))]

        # Filter state
        if state:
            try:
                state_list = state[:5]
                state_fips = [convert_state_to_fips(s) for s in state_list]
                df = df[df['state_code'].isin(state_fips)]
            except ValueError as e:
                warnings.warn(f"State conversion error: {str(e)}")

        # Filter county
        if county:
            county_list = [c.zfill(3) for c in county[:5]]
            df = df[df['county_code'].isin(county_list)]

        # Filter fips
        if fips:
            try:
                cleaned_fips = clean_fips(fips)
                df = df[df['county_code'] == cleaned_fips[2:]]
            except ValueError as e:
                warnings.warn(str(e))

        # Filter coverage level
        if cov_lvl:
            cov_percent = [cl * 100 for cl in cov_lvl]
            df = df[df['coverage_level_percent'].astype(float).isin(cov_percent)]

        df = clean_column_names(df)
        all_years_data.append(df)

    return pd.concat(all_years_data, ignore_index=True) if all_years_data else pd.DataFrame()