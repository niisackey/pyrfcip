# 🇺🇸 USDA Crop Insurance Explorer (Python Version)

This project is a **Python/Streamlit port** of the original [R-based application](https://github.com/dylan-turner25/rfcip ) by **Dylan Turner**. It was cloned from his GitHub repository to improve **functionality**, **usability**, and to make the tool more accessible to Python users.

The USDA Crop Insurance Explorer provides an interactive way to access, filter, and analyze USDA agricultural data — including **Risk Management Agency (RMA)** and **National Agricultural Statistics Service (NASS)** datasets — with smart validation and CSV export features.

---

## ✨ Features

- Query by crop, state, year, and insurance program
- Combines data from **USDA RMA** and **USDA NASS**
- Modules include:
  - Summary of Business
  - County-Level Loss Data
  - Livestock Insurance Programs
  - Price Discovery Tools
  - Reinsurance Reports (National and State)
- Reference tables for:
  - Crop codes
  - Insurance plans
  - Causes of loss
- Export query results as CSV

---

## 🚀 Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/niisackey/pyrfcip
cd rfcip_python_app
python -m venv venv
venv\Scripts\activate     # On Windows
pip install -e .

2. Launch the App

streamlit run app.py


⸻

✅ Tests

Tests are written using pytest.

Run Tests

pytest -v

Test coverage includes:
	•	Data download and formatting validation
	•	Column presence and integrity
	•	Graceful handling of server/API errors (e.g., HTTP 500)

⸻

📁 Project Structure

.
rfcip/                    # Core modules for data access & processing
│   summary.py
│   col.py
│   codes.py
│   livestock.py
│   reinsurance_reports.py
│   nass_data.py          # Handles NASS data integration
app.py                   # Streamlit interface
/tests
│   test_summary.py
│   test_col.py
│   test_codes.py
│   test_nass.py          # Tests for NASS data access
│   test_helpers.py
pyproject.toml           # Build and dependency configuration


⸻

⚙ Dependencies
	•	Python 3.9+
	•	Streamlit
	•	Pandas
	•	Requests
	•	BeautifulSoup4
	•	Pytest

⸻

🌐 Data Sources
	•	USDA RMA APIs
	•	USDA NASS Quick Stats API
	•	Reinsurance Reports

⸻

🎯 Goal

The project aims to improve access to complex USDA crop and insurance datasets by combining RMA and NASS data sources into one interactive platform for:
	•	Policy researchers
	•	Economists
	•	Data scientists
	•	Agri-business professionals

⸻

🚜 Contributions

Contributions are welcome! Feel free to fork the project, submit pull requests, or open issues for suggestions and improvements.

⸻
