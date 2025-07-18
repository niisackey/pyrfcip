from rfcip.codes import get_crop_codes, get_insurance_plan_codes

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