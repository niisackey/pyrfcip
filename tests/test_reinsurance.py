from rfcip.reinsurance_reports import build_reinsurance_datasets
import pytest

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
