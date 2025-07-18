from rfcip.livestock import get_livestock_data

def test_livestock_data():
    df = get_livestock_data(year=[2022], program="LRP")
    assert df is not None
    assert not df.empty
    assert "location_state_abbreviation" in df.columns