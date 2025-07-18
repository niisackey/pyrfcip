# rfcip/livestock_codes.py
import pandas as pd
import requests
import xml.etree.ElementTree as ET

def get_livestock_codes_dynamic(year: int = 2023, fallback_path="data/livestock_codes.csv") -> pd.DataFrame:
    url = f"https://public-rma.fpac.usda.gov/apps/SummaryOfBusiness/Services/SummaryDataService.svc/GetSummaryData?cropYear={year}&summaryLevel=Commodity&format=xml"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        root = ET.fromstring(response.text)

        livestock_plan_codes = {"63", "70"}  # LRP, LGM
        rows = []
        for item in root.findall(".//SummaryOfBusiness"):
            code = item.findtext("CommodityCode")
            name = item.findtext("CommodityName")
            plan_code = item.findtext("PlanCode")
            if code and name and plan_code in livestock_plan_codes:
                rows.append({"code": int(code), "commodity": name.strip()})

        df = pd.DataFrame(rows).drop_duplicates().sort_values("code").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"[Fallback] Livestock code fetch failed: {e}")
        try:
            return pd.read_csv(fallback_path)
        except Exception as e2:
            print(f"[Error] Could not load fallback livestock code CSV: {e2}")
            return pd.DataFrame()
