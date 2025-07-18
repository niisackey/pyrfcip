# tests/test_summary.py
import pytest
from rfcip.summary import get_summary_data
from rfcip.codes import get_crop_codes, get_insurance_plan_codes
from rfcip.reinsurance_reports import build_reinsurance_datasets

@pytest.mark.parametrize("valid_crop, valid_state, sample_years", [
    ("CORN", "IA", [2021, 2022])
])
def test_summary_data_valid_inputs(valid_crop, valid_state, sample_years):
    df = get_summary_data(
        crop=valid_crop,
        state=valid_state,
        year=sample_years
    )
    if df.empty:
        pytest.skip("Summary data API unavailable or returned no data")
    assert not df.empty
    assert "commodity_year" in df.columns

def test_crop_codes():
    df = get_crop_codes(year=2023)
    assert df is not None
    assert not df.empty
    assert "commodity_name" in df.columns

def test_insurance_plan_codes():
    df = get_insurance_plan_codes(year=2023)
    assert df is not None
    assert not df.empty
    assert "insurance_plan" in df.columns

def test_reinsurance_reports():
    base_url = "https://www.rma.usda.gov/tools-reports/reinsurance-reports"
    data = build_reinsurance_datasets(base_url)
    assert isinstance(data, dict)
    if not data:
        pytest.skip("Reinsurance download failed or returned empty dict")
    for key, df in data.items():
        if df.empty:
            pytest.skip(f"{key} returned empty DataFrame")
        assert "reinsurance_year" in df.columns
