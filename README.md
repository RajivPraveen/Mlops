<h1 align="center">MLOps Lab Assignments</h1>

<p align="center">
  <strong>IE-7374 Machine Learning Operations | Northeastern University</strong>
</p>

<p align="center">
  <a href="https://github.com/RajivPraveen/Mlops/actions/workflows/pytest_action.yml">
    <img src="https://github.com/RajivPraveen/Mlops/actions/workflows/pytest_action.yml/badge.svg" alt="Pytest">
  </a>
  <a href="https://github.com/RajivPraveen/Mlops/actions/workflows/unittest_action.yml">
    <img src="https://github.com/RajivPraveen/Mlops/actions/workflows/unittest_action.yml/badge.svg" alt="Unittest">
  </a>
  <img src="https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue" alt="Python">
</p>

---

## About

This repository contains lab assignments for the **Machine Learning Operations (MLOps)** course. Each lab folder is a self-contained project covering a different aspect of the MLOps lifecycle — from version control and CI/CD to API deployment and production ML practices.

**Course Reference:** [github.com/raminmohammadi/MLOps](https://github.com/raminmohammadi/MLOps)

---

## Labs

| Lab | Topic | Description | Status |
|-----|-------|-------------|--------|
| [**Github Lab**](./Github%20Lab/) | GitHub, Testing & CI/CD | Enhanced calculator with statistical functions, a data processor, and a linear regression model — all tested with pytest/unittest and automated via GitHub Actions | ✅ Complete |
| [**FastAPI Lab**](./FastAPI%20Lab/) | API Development | ML model deployment using FastAPI with RESTful endpoints | ✅ Complete |
| [**Streamlit labs**](./Streamlit%20labs/) | Data App Development | Interactive Streamlit applications for data exploration, backend integration, and model-driven workflows | ✅ Complete |

---

## Repository Structure

```
Mlops/
├── .github/
│   └── workflows/
│       ├── pytest_action.yml          # CI: Pytest across Python 3.9–3.12
│       └── unittest_action.yml        # CI: Unittest suite
│
├── Github Lab/
│   ├── data/                          # Housing price dataset
│   ├── src/                           # Calculator, data processor, ML model
│   ├── test/                          # Pytest & unittest test suites
│   ├── workflows/                     # Workflow reference copies
│   ├── requirements.txt
│   └── README.md
│
├── FastAPI Lab/
│   ├── src/                           # FastAPI application
│   ├── model/                         # ML model artifacts
│   ├── assets/                        # Documentation assets
│   ├── requirements.txt
│   └── README.md
│
├── Streamlit labs/
│   ├── src/                           # Streamlit app source code
│   ├── backend/                       # Backend services/utilities used by app
│   ├── data/                          # Datasets and static resources
│   ├── assets/                        # UI and documentation assets
│   ├── requirements.txt
│   └── README.md                      # Lab overview and usage details
│
├── .gitignore
└── README.md                          # ← You are here
```

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/RajivPraveen/Mlops.git
cd Mlops

# Navigate to a specific lab
cd "Github Lab"

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies & run tests
pip install -r requirements.txt
pytest test/test_pytest.py -v
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.9+** | Core language |
| **pytest** | Testing framework |
| **unittest** | Additional testing (stdlib) |
| **flake8** | Linting & code quality |
| **FastAPI** | API framework |
| **GitHub Actions** | CI/CD pipelines |

---

<p align="center">
  <sub>Rajiv Praveen | Northeastern University | Spring 2026</sub>
</p>
