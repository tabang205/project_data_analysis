# Financial Accounts Payable Data Science Project

An end-to-end predictive machine learning system for Accounts Payable (AP) cashflow management and vendor payment risk analysis, built on real Metro Nashville Vendor Payments data (~1.3M records).

## Project Overview

- **Dual-Track Objective**:
  - **Classification (`is_late`)**: Predict whether an incoming vendor invoice will be paid late (`days_to_pay > 30` days) to prioritize approvals and mitigate supplier friction.
  - **Regression (`days_to_pay`)**: Predict expected settlement days (modeled via `log1p(days_to_pay)`) to forecast accurate cashflow outflow schedules.
- **Data Engine**: DuckDB SQL for invoice grain resolution and multi-line item aggregation; Pandera for rigorous schema validation.
- **Machine Learning**: Point-in-time out-of-time temporal feature engineering, class-weighted baselines, and calibrated LightGBM/XGBoost models with SHAP explanations.
- **Interactive UI**: Streamlit application for risk scoring and cashflow calendar visualization.

## Architecture & Directory Layout

```
├── artifacts/              # Serialized models, metrics, and SHAP artifacts (gitignored)
├── data/
│   ├── raw/                # Raw ingested Parquet files from ArcGIS FeatureServer
│   ├── interim/            # Grain-resolved invoice payments & audit tables
│   └── processed/          # Point-in-time train/val/test feature matrices
├── docs/
│   ├── learning/           # Concept deep-dives and troubleshooting logs
│   └── superpowers/        # Architectural specs and implementation plans
├── src/
│   ├── app/                # Streamlit dashboard and UI components
│   ├── data/               # Ingestion, schema validation, and grain resolution
│   ├── features/           # Temporal feature engineering & out-of-time splits
│   ├── models/             # Baselines, tree models, calibration, and SHAP
│   └── config.py           # Centralized immutable configuration
├── tests/                  # Unit and integration test suite
├── pyproject.toml          # Project configuration & pytest settings
└── requirements.txt        # Pinned project dependencies
```

## Getting Started

### 1. Environment Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run Tests
```powershell
pytest -v
```
