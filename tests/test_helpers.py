# tests/test_helpers.py
from rfcip.helpers import valid_state, valid_crop

def test_valid_state():
    assert valid_state("IA")
    assert not valid_state("XX")

def test_valid_crop():
    assert valid_crop("CORN")
    assert not valid_crop("FAKE")
