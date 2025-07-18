# tests/conftest.py
import pytest

@pytest.fixture
def sample_years():
    return [2021, 2022]

@pytest.fixture
def valid_crop():
    return "CORN"

@pytest.fixture
def valid_state():
    return "IA"
