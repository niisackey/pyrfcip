from rfcip.col import get_col_data

def test_col_data():
    df = get_col_data(year=[2021])
    assert df is not None
    assert not df.empty
    assert "commodity_name" in df.columns