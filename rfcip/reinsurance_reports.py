# reinsurance_reports.py
import pandas as pd
import requests
import zipfile
import tempfile
import os
from pathlib import Path
from bs4 import BeautifulSoup
import warnings

def download_zip_extract_csv(url: str) -> pd.DataFrame:
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = os.path.join(tmp_dir, "file.zip")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with open(zip_path, 'wb') as f:
                f.write(response.content)

            with zipfile.ZipFile(zip_path, 'r') as z:
                csv_files = [f for f in z.namelist() if f.endswith('.csv') or f.endswith('.txt')]
                if not csv_files:
                    raise ValueError("No CSV or TXT files found in ZIP archive.")
                with z.open(csv_files[0]) as f:
                    return pd.read_csv(f, dtype=str)
    except Exception as e:
        warnings.warn(f"Failed to download or extract CSV from {url}: {str(e)}")
        return pd.DataFrame()

def get_reinsurance_links(base_url: str) -> list[str]:
    try:
        response = requests.get(base_url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        links = [a['href'] for a in soup.find_all('a', href=True)]
        return [l for l in links if l.endswith('.zip') and 'reinsurance' in l.lower()]
    except Exception as e:
        warnings.warn(f"Failed to fetch reinsurance links from {base_url}: {str(e)}")
        return []

def build_reinsurance_datasets(base_url: str) -> dict:
    links = get_reinsurance_links(base_url)
    combined_data = []
    for link in links:
        if not link.startswith("http"):
            link = f"https://www.rma.usda.gov{link}"
        df = download_zip_extract_csv(link)
        if not df.empty:
            combined_data.append(df)

    if not combined_data:
        return {
            "nationalSRA": pd.DataFrame(),
            "stateSRA": pd.DataFrame(),
            "nationalLPRA": pd.DataFrame()
        }

    full_df = pd.concat(combined_data, ignore_index=True)

    national_sra = full_df[full_df['report_geography'].str.lower() == 'national']
    state_sra = full_df[full_df['report_geography'].str.lower() == 'state']
    lpra = full_df[full_df.get('report_type', '').str.contains('livestock', case=False, na=False)]

    return {
        "nationalSRA": national_sra,
        "stateSRA": state_sra,
        "nationalLPRA": lpra
    }

# Example usage (if running standalone)
if __name__ == "__main__":
    base_url = "https://www.rma.usda.gov/tools-reports/reinsurance-reports"
    datasets = build_reinsurance_datasets(base_url)
    for name, df in datasets.items():
        print(f"{name}: {len(df)} rows")
