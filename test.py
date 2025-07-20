# test_sobtpu_zip.py

import pandas as pd
import requests
import zipfile
import io
import os
from pathlib import Path


def fetch_and_parse_sobtpu(year: int = 2023, save_csv: bool = True):
    url = f"https://public-rma.fpac.usda.gov/apps/SummaryOfBusiness/ReportGenerator/SobtpuData/{year}/sobtpu_{year}.zip"
    print(f"Requesting ZIP: {url}")

    try:
        response = requests.get(url)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            excel_files = [f for f in z.namelist() if f.endswith(".xlsx")]
            if not excel_files:
                raise ValueError("No Excel file found in ZIP archive")

            df = pd.read_excel(z.open(excel_files[0]))
            print("\n✅ Columns in ZIP Excel file:")
            print(df.columns.tolist())

            if save_csv:
                out_path = Path(f"sobtpu_{year}_data.csv")
                df.to_csv(out_path, index=False)
                print(f"\n💾 Saved extracted data to: {out_path.resolve()}")

    except Exception as e:
        print(f"❌ Failed to fetch/parse ZIP: {e}")


if __name__ == "__main__":
    fetch_and_parse_sobtpu(year=2023)
