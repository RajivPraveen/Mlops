# Github Lab - Statistical Calculator & ML Toolkit

> **IE-7374 Machine Learning Operations** | Northeastern University
>
> An enhanced version of [Lab1](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Github_Labs/Lab1) — extending the basic calculator into a full statistical analysis and machine learning toolkit.

---

## Overview

This lab focuses on 5 core modules from the original Lab1, **plus significant enhancements**:

1. Creating a Virtual Environment
2. Creating a GitHub Repository, Cloning, and Folder Structure
3. Creating Python source files (`calculator.py` + new modules)
4. Writing tests using **Pytest** and **Unittest**
5. Implementing **GitHub Actions** for CI/CD

### What's Different from the Original Lab1

| Original Lab1 | This Version |
|---|---|
| Basic calculator: `fun1`–`fun4` (add, subtract, multiply, sum3) | **Kept all originals** + added `divide`, `power`, `square_root` |
| No statistical functions | Added `mean`, `median`, `mode`, `variance`, `std_dev`, `correlation`, `percentile` |
| No dataset | Includes a **housing price dataset** (30 records, 5 features) |
| No data processing | New **DataProcessor** module for CSV loading, normalization, filtering |
| No ML model | New **LinearRegressionModel** built from scratch (OLS) with train, predict, evaluate, save/load |
| 4 pytest + 4 unittest tests | **50+ pytest** and **20+ unittest** tests covering edge cases |
| Single workflow | **Two workflows** (pytest + unittest) running across Python 3.9–3.12 |

---

## Folder Structure

```
Github Lab/
├── data/
│   └── housing_data.csv          # Housing price dataset (30 records)
├── src/
│   ├── __init__.py               # Package exports
│   ├── calculator.py             # Enhanced calculator with statistics
│   ├── data_processor.py         # CSV data loading & processing
│   └── model.py                  # Linear regression from scratch
├── test/
│   ├── __init__.py
│   ├── test_pytest.py            # Pytest test suite (50+ tests)
│   └── test_unittest.py          # Unittest test suite (20+ tests)
├── workflows/
│   ├── pytest_action.yml         # Pytest GitHub Actions workflow
│   └── unittest_action.yml       # Unittest GitHub Actions workflow
├── README.md
└── requirements.txt
```

---

## Setup

### Step 1: Create a Virtual Environment

```bash
python -m venv github_lab_env
source github_lab_env/bin/activate      # macOS/Linux
# github_lab_env\Scripts\activate       # Windows
```

### Step 2: Clone the Repository

```bash
git clone https://github.com/RajivPraveen/Mlops.git
cd Mlops/Github\ Lab
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Calculator (Original + Enhanced)

```python
from src import calculator

# ── Original functions (same as Lab1) ──
calculator.fun1(2, 3)       # 5  (addition)
calculator.fun2(5, 3)       # 2  (subtraction)
calculator.fun3(4, 3)       # 12 (multiplication)
calculator.fun4(2, 3, 5)    # 10 (sum of three)

# ── Enhanced functions ──
calculator.divide(10, 3)    # 3.333...
calculator.power(2, 8)      # 256
calculator.square_root(144) # 12.0

# ── Statistical functions ──
data = [23, 45, 67, 12, 89, 34, 56]
calculator.mean(data)       # 46.57...
calculator.median(data)     # 45
calculator.std_dev(data)    # 23.88...
calculator.percentile(data, 75)

# ── Correlation ──
x = [1, 2, 3, 4, 5]
y = [2.1, 4.0, 5.8, 8.2, 9.9]
calculator.correlation(x, y)  # ~0.999
```

### Data Processor

```python
from src.data_processor import DataProcessor

dp = DataProcessor().load_csv("data/housing_data.csv")
print(dp.shape())                            # (30, 5)
print(dp.column_stats("price_thousands"))    # {count, mean, min, max, std}
normalized = dp.normalize_column("size_sqft")
expensive = dp.filter_rows("price_thousands", ">=", 400)
```

### Linear Regression Model

```python
from src.model import LinearRegressionModel

model = LinearRegressionModel()
model.fit([1, 2, 3, 4, 5], [3, 5, 7, 9, 11])  # y = 2x + 1

model.predict_single(10)    # 21.0
model.evaluate([1,2,3,4,5], [3,5,7,9,11])  # {mae, mse, rmse, r_squared}

model.save_model("model.json")
model.load_model("model.json")
```

---

## Testing

### Run Pytest

```bash
pytest test/test_pytest.py -v
```

### Run Unittest

```bash
python -m unittest test.test_unittest -v
```

---

## GitHub Actions

Two CI/CD workflows are configured under `.github/workflows/` at the repository root:

| Workflow | Trigger | What it Does |
|---|---|---|
| `pytest_action.yml` | Push/PR to `main` | Lints with flake8 + runs pytest across Python 3.9–3.12 |
| `unittest_action.yml` | Push/PR to `main` | Runs unittest suite on Python 3.11 |

Reference copies of these workflows are also stored in the `workflows/` folder within this lab.

---

## Dataset

**`data/housing_data.csv`** — A synthetic housing price dataset with 30 records:

| Column | Description |
|---|---|
| `size_sqft` | House size in square feet (750–2300) |
| `bedrooms` | Number of bedrooms (1–5) |
| `age_years` | Age of the house in years (1–35) |
| `distance_downtown_miles` | Distance from downtown (1.0–11.0) |
| `price_thousands` | Sale price in thousands of dollars (155–525) |
