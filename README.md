# USDA Crop Insurance Explorer (Python Version)

This is a Python-based port of the original R application that explores and analyzes USDA Risk Management Agency (RMA) Crop Insurance data. It provides interactive visualizations, query tools, and data exports powered by Streamlit.

---

## ✨ Features

* Query by crop, state, year, and insurance program
* Includes:

  * Summary of Business
  * County-Level Loss data
  * Livestock Insurance programs
  * Price Discovery
  * Reinsurance Reports (National and State)
* Reference tables for crop codes, insurance plans, causes of loss
* Download results as CSV

---

## 🚀 Quick Start

### 1. Clone and Install

```bash
$ git clone <https://github.com/niisackey/pyrfcip>
$ cd rfcip_python_app
$ python -m venv venv
$ venv\Scripts\activate     # On Windows
$ pip install -e .
```

### 2. Launch the App

```bash
$ streamlit run app.py
```

---

## ✅ Tests

Tests are written using `pytest`.

### Run Tests

```bash
$ pytest -v
```

Tests are designed to:

* Validate data downloads
* Check expected columns
* Gracefully skip on remote API/server failure (e.g. HTTP 500)

---

## 📊 Project Structure

```
.
rfcip/                    # Core modules for data access & cleaning
    summary.py
    col.py
    codes.py
    livestock.py
    reinsurance_reports.py
app.py                   # Streamlit UI

/tests
    test_summary.py      # All major data source tests
    test_col.py
    test_codes.py
    test_helpers.py
    ...

pyproject.toml           # Build metadata
```

---

## ⚙ Dependencies

* Python 3.9+
* Streamlit
* Pandas
* Requests
* BeautifulSoup4
* pytest

---

## 🌐 Data Sources

* USDA RMA APIs
* Reinsurance Reports: [https://www.rma.usda.gov/tools-reports/reinsurance-reports](https://www.rma.usda.gov/tools-reports/reinsurance-reports)

---

## 📊 Goal

This tool aims to improve accessibility to complex USDA crop insurance datasets for:

* Policy researchers
* Economists
* Data scientists
* Agri-business analysts

---

## 🚜 Contributions

Feel free to fork, improve, and submit a PR. Suggestions welcome.

---

## 📢 Disclaimer

This project is not affiliated with or endorsed by the USDA. Use at your own risk. Data sourced from public USDA RMA tools.
