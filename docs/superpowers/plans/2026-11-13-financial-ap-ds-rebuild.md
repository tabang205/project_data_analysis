# Financial AP Data Science Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `project_data_analysis` from scratch into a rigorous, end-to-end Data Science portfolio project predicting invoice late-payment risk (`is_late`) and payment timing/cashflow (`days_to_pay`) using real Metro Nashville Vendor Payments data, paired with Vietnamese learning notes and a troubleshooting log.

**Architecture:** Legacy v1 files are preserved under Git tag `v1-legacy` and cleared from `main`. Phase 1 builds reproducible raw ingestion (ArcGIS REST API), DuckDB SQL grain resolution (`raw_distributions` -> `invoice_payment` grain), Pandera data contracts, statistical EDA, strict point-in-time feature engineering (`train`/`val`/`test` out-of-time splits), dual-track evaluation, and local linear baselines. Phase 2 is strictly gated behind user confirmation that their current Kaggle job has finished, then pushes processed datasets to Kaggle to train/tune/calibrate LightGBM & XGBoost models with SHAP and feeds artifacts into a lightweight Streamlit dashboard.

**Tech Stack:** Python 3.12 (`.venv`), DuckDB, Pandas, PyArrow, Pandera, Scikit-learn, LightGBM, XGBoost, SHAP, Matplotlib/Seaborn, Pytest, Streamlit, Kaggle API/MCP.

**Spec:** [docs/superpowers/specs/2026-11-13-financial-ap-ds-rebuild-design.md](file:///D:/Antigravity/projects/project_data_analysis/docs/superpowers/specs/2026-11-13-financial-ap-ds-rebuild-design.md)

## Global Constraints
- Storage: All project files stay inside `D:\Antigravity\projects\project_data_analysis\`; raw/interim/processed parquet files live under `data/` (gitignored).
- Data source: Official Metro Nashville Open Data ArcGIS FeatureServer (`1,307,617` raw rows); no synthetic replacements or third-party Kaggle CSVs.
- Anti-leakage: Features at `Invoice_Date = t` MUST NEVER inspect `Payment_Date >= t`, `Payment_Number`, or `Check_Amount` of the current or future payments.
- Splits: `Train` (`2022-07-01`..`2024-06-30`), `Val` (`2024-07-01`..`2024-12-31`), `Test` (`2025-01-01`..`2025-10-31`, 61-day right-censoring buffer before `2025-12-31`).
- Late threshold: `is_late = 1` if `days_to_pay > 30`, else `0`; secondary regression target `days_to_pay` (trained via `log1p(days_to_pay)`).
- Kaggle Gate: Do NOT push datasets or run training kernels on Kaggle until the user explicitly confirms their existing Kaggle project has finished.
- Lean code hygiene: Zero debug `print()` clutter (use Python `logging`), minimal WHY-focused comments, type hints on all functions.

## Review Focus
1. **Multi-line & Multi-department Invoice Aggregation:** An invoice split across multiple `Distribution_Amount` rows and multiple `Department_Number`s must aggregate to a single row with exact `SUM(Distribution_Amount)`, `line_item_count`, `dept_count`, and deterministic `MIN(Department_Name)`. -> Tested in Task 3 (`test_grain_aggregates_multiline_multidept_invoice`).
2. **Corrupted Dates & Credit Memos (`<= 0`):** Raw rows with out-of-range `Invoice_Date` (e.g., 1991, 2083), `Invoice_Date > Payment_Date`, or `invoice_amount <= 0` must be routed to audit/credit outputs (recording `vendor_has_credit_history`) instead of silently entering model training. -> Tested in Task 3 (`test_audit_splits_bad_dates_and_credit_memos`).
3. **Temporal Feature Leakage (`Payment_Date >= current.Invoice_Date`):** Historical vendor/department aggregates for an invoice at `t` must ignore prior invoices whose `Payment_Date >= t` even when their `Invoice_Date < t`. -> Tested in Task 5 (`test_features_ignore_future_payments_after_invoice_date`).
4. **Right-Censoring Cutoff & Cashflow Aggregation Consistency:** Rows with `Invoice_Date > 2025-10-31` must be excluded from `train/val/test`, and predicted cashflow outflows on `predicted_payment_date` must match summed invoice amounts by horizon bucket. -> Tested in Task 5 (`test_time_split_enforces_censoring_cutoff`) and Task 6 (`test_cashflow_aggregation_matches_predicted_dates`).
5. **Cold-Start Vendors & Zero-Day Payments (`days_to_pay == 0`):** First-time vendors (`prior_count == 0`) and same-day settlements (`log1p(0) == 0`) must yield finite NaN-safe features and non-negative inverse-transformed day predictions. -> Tested in Task 5 (`test_cold_start_vendor_finite_features`) and Task 6 (`test_regression_inverse_transform_non_negative`).

---

### Task 1: Preserve Legacy Tag, Clean `main` & Scaffold Project

**Files:**
- Create: `pyproject.toml`, `requirements.txt`, `.env.example`, `src/__init__.py`, `src/config.py`, `src/data/__init__.py`, `src/features/__init__.py`, `src/models/__init__.py`, `src/app/__init__.py`, `data/raw/.gitkeep`, `data/interim/.gitkeep`, `data/processed/.gitkeep`, `artifacts/.gitkeep`, `docs/learning/00_roadmap_and_concepts.md`, `docs/learning/troubleshooting_log.md`
- Modify: `.gitignore`
- Remove from `main` (after tagging `v1-legacy` at `8ea7b7a`): `financial-rag/`, `invoices_scan/`, `ap_analysis.py`, `invoice_validation.py`, `paddleocr_qwen_extraction.py`, `streamlit_app.py`, `data-analysis.ipynb`, `colab_learning.ipynb`, `kaggle_production.ipynb`, `README.md` (replaced by new `README.md` stub)

**Interfaces:**
- Produces: `src.config.Settings` dataclass and `SETTINGS: Settings` singleton (`raw_dir`, `interim_dir`, `processed_dir`, `artifacts_dir`, `nashville_feature_server_url`, `late_threshold_days=30`, `train_end="2024-06-30"`, `val_end="2024-12-31"`, `test_end="2025-10-31"`, `min_valid_invoice_date="2015-01-01"`, `max_valid_payment_date="2025-12-31"`).

- [ ] **Step 1: Tag `v1-legacy` at `8ea7b7a` and clean legacy v1 files from working tree**

Run:
```powershell
git -C "D:\Antigravity\projects\project_data_analysis" tag -a v1-legacy 8ea7b7a -m "Archive v1 RAG + OCR + Scam notebooks before Data Science rebuild"
git -C "D:\Antigravity\projects\project_data_analysis" rm -rf financial-rag invoices_scan ap_analysis.py invoice_validation.py paddleocr_qwen_extraction.py streamlit_app.py data-analysis.ipynb colab_learning.ipynb kaggle_production.ipynb
```
Expected: Tag `v1-legacy` exists (`git tag -l` shows `v1-legacy`), legacy v1 files removed from index.

- [ ] **Step 2: Install missing DS dependencies into `.venv` and write `requirements.txt` / `pyproject.toml` / `.env.example` / `.gitignore`**

Run:
```powershell
D:\Antigravity\projects\project_data_analysis\.venv\Scripts\python.exe -m pip install duckdb lightgbm xgboost shap matplotlib seaborn pytest plotly
```
Pin installed versions cleanly in `requirements.txt` and configure `pytest` (`testpaths = ["tests"]`, `pythonpath = ["."]`) in `pyproject.toml`. Update `.gitignore` to ignore `data/raw/*`, `data/interim/*`, `data/processed/*`, `artifacts/*` (except `.gitkeep`) and `.env`.

- [ ] **Step 3: Write failing config smoke test `tests/test_config.py`**

```python
from src.config import SETTINGS

def test_settings_splits_are_strictly_ordered_and_buffered():
    assert SETTINGS.train_end < SETTINGS.val_end < SETTINGS.test_end
    assert SETTINGS.late_threshold_days == 30
    assert SETTINGS.test_end == "2025-10-31"
```

- [ ] **Step 4: Run test to verify it fails**

Run: `D:\Antigravity\projects\project_data_analysis\.venv\Scripts\pytest.exe tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src'`.

- [ ] **Step 5: Implement `src/config.py`, package `__init__.py` files, minimal `README.md`, `docs/learning/00_roadmap_and_concepts.md`, and `docs/learning/troubleshooting_log.md`**

Create `Settings` frozen dataclass in `src/config.py` using `pathlib.Path` rooted at project base directory and export `SETTINGS = Settings()`. Write roadmap + concept overview in `00_roadmap_and_concepts.md` and initialize `troubleshooting_log.md`.

- [ ] **Step 6: Run test to verify it passes and commit**

Run: `D:\Antigravity\projects\project_data_analysis\.venv\Scripts\pytest.exe tests/test_config.py -v`
Expected: PASS (1 passed).

```powershell
git add -A
git commit -m "chore: archive v1-legacy tag and scaffold DS rebuild"
```

---

### Task 2: Reproducible Raw Ingestion & Pandera Validation (`fetch_nashville.py`, `validate_raw.py`)

**Files:**
- Create: `src/data/fetch_nashville.py`, `src/data/validate_raw.py`
- Test: `tests/test_ingestion.py`

**Interfaces:**
- Consumes: `src.config.SETTINGS`
- Produces:
  - `fetch_nashville_data(output_path: Path, page_size: int = 2000, max_rows: int | None = None, session: requests.Session | None = None) -> Path`
  - `validate_raw_schema(df: pd.DataFrame) -> pd.DataFrame` (coerces epoch-ms or ISO timestamps to UTC-naive `datetime64[ns]`, validates column types & non-null `FID` via Pandera `DataFrameSchema`)

- [ ] **Step 1: Write failing tests in `tests/test_ingestion.py`**

```python
import pandas as pd
from unittest.mock import MagicMock
from src.data.fetch_nashville import fetch_nashville_data
from src.data.validate_raw import validate_raw_schema

def test_fetch_paginates_via_exceeded_transfer_limit(tmp_path):
    mock_session = MagicMock()
    page1 = {"features": [{"attributes": {"FID": 1, "Operating_Unit": "A", "Supplier_Number": "S1", "Invoice_Number": "I1", "Distribution_Amount": 100.0, "Invoice_Date": 1665446400000, "Payment_Date": 1665446400000, "Payment_Number": 10, "Check_Amount": 100.0, "Department_Name": "D1", "Department_Number": 1}}], "exceededTransferLimit": True}
    page2 = {"features": [{"attributes": {"FID": 2, "Operating_Unit": "A", "Supplier_Number": "S2", "Invoice_Number": "I2", "Distribution_Amount": 200.0, "Invoice_Date": 1665532800000, "Payment_Date": 1665619200000, "Payment_Number": 11, "Check_Amount": 200.0, "Department_Name": "D1", "Department_Number": 1}}], "exceededTransferLimit": False}
    resp1, resp2 = MagicMock(), MagicMock()
    resp1.json.return_value, resp2.json.return_value = page1, page2
    resp1.raise_for_status = resp2.raise_for_status = lambda: None
    mock_session.get.side_effect = [resp1, resp2]

    out = fetch_nashville_data(tmp_path / "raw.parquet", page_size=1, session=mock_session)
    df = pd.read_parquet(out)
    assert len(df) == 2
    assert list(df["FID"]) == [1, 2]

def test_validate_raw_schema_parses_epoch_ms_dates():
    df = pd.DataFrame([{"FID": 1, "Operating_Unit": "METRO", "Supplier_Number": "797", "Invoice_Number": "INV1", "Distribution_Amount": 50.0, "Invoice_Date": 1665446400000, "Payment_Date": 1665532800000, "Payment_Number": 10, "Check_Amount": 50.0, "Department_Name": "MNPS", "Department_Number": 80}])
    cleaned = validate_raw_schema(df)
    assert pd.api.types.is_datetime64_any_dtype(cleaned["Invoice_Date"])
    assert cleaned.loc[0, "Invoice_Date"] == pd.Timestamp("2022-10-11")
```

- [ ] **Step 2: Run tests to verify failure**

Run: `D:\Antigravity\projects\project_data_analysis\.venv\Scripts\pytest.exe tests/test_ingestion.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `fetch_nashville.py` and `validate_raw.py`**

Implement paginated `resultOffset` fetching ordered by `FID ASC` with HTTP retries (`urllib3.util.retry.Retry`), incremental parquet chunking or clean memory-safe assembly, and `validate_raw_schema()` using `pandera.pandas` converting epoch-ms to `datetime64[ns]`. Add CLI entrypoint (`python -m src.data.fetch_nashville`).

- [ ] **Step 4: Run tests to verify pass, then run full live download to `data/raw/nashville_raw.parquet` and commit**

Run:
```powershell
D:\Antigravity\projects\project_data_analysis\.venv\Scripts\pytest.exe tests/test_ingestion.py -v
D:\Antigravity\projects\project_data_analysis\.venv\Scripts\python.exe -m src.data.fetch_nashville
```
Expected: Tests PASS; `data/raw/nashville_raw.parquet` created with ~1.3M rows (document exact fetched count and any HTTP/API quirks in `docs/learning/troubleshooting_log.md`).

```powershell
git add src/data/fetch_nashville.py src/data/validate_raw.py tests/test_ingestion.py docs/learning/troubleshooting_log.md
git commit -m "feat(data): add Nashville ArcGIS ingestion and raw schema validation"
```

---

### Task 3: DuckDB Grain Resolution & Data Quality Audit (`build_invoice_grain.sql`, `build_dataset.py`)

**Files:**
- Create: `sql/build_invoice_grain.sql`, `src/data/build_dataset.py`, `docs/learning/01_data_audit_and_sql.md`
- Test: `tests/test_grain_and_cleaning.py`

**Interfaces:**
- Consumes: `data/raw/nashville_raw.parquet`, `src.config.SETTINGS`
- Produces:
  - `build_invoice_dataset(raw_parquet: Path, out_clean_parquet: Path, out_audit_json: Path, out_credit_parquet: Path) -> dict[str, int]`
  - Columns of `invoices_clean.parquet`: `operating_unit`, `supplier_number`, `invoice_number`, `payment_number`, `invoice_date`, `payment_date`, `primary_department`, `primary_dept_number`, `dept_count`, `line_item_count`, `invoice_amount`, `check_amount`, `days_to_pay`, `is_late`

- [ ] **Step 1: Write failing tests covering Review Focus #1 and #2 in `tests/test_grain_and_cleaning.py`**

```python
import pandas as pd
from src.data.build_dataset import build_invoice_dataset

def test_grain_aggregates_multiline_multidept_invoice(tmp_path):
    raw = pd.DataFrame([
        {"FID": 1, "Operating_Unit": "METRO", "Supplier_Number": "S1", "Invoice_Number": "INV-A", "Payment_Number": 100, "Invoice_Date": pd.Timestamp("2024-01-01"), "Payment_Date": pd.Timestamp("2024-02-05"), "Department_Name": "Health", "Department_Number": 20, "Distribution_Amount": 300.0, "Check_Amount": 500.0},
        {"FID": 2, "Operating_Unit": "METRO", "Supplier_Number": "S1", "Invoice_Number": "INV-A", "Payment_Number": 100, "Invoice_Date": pd.Timestamp("2024-01-01"), "Payment_Date": pd.Timestamp("2024-02-05"), "Department_Name": "Admin", "Department_Number": 10, "Distribution_Amount": 200.0, "Check_Amount": 500.0},
    ])
    raw_path = tmp_path / "raw.parquet"
    raw.to_parquet(raw_path)
    build_invoice_dataset(raw_path, tmp_path / "clean.parquet", tmp_path / "audit.json", tmp_path / "credit.parquet")
    clean = pd.read_parquet(tmp_path / "clean.parquet")
    assert len(clean) == 1
    assert clean.loc[0, "invoice_amount"] == 500.0
    assert clean.loc[0, "line_item_count"] == 2
    assert clean.loc[0, "dept_count"] == 2
    assert clean.loc[0, "primary_department"] == "Admin"
    assert clean.loc[0, "days_to_pay"] == 35
    assert clean.loc[0, "is_late"] == 1

def test_audit_splits_bad_dates_and_credit_memos(tmp_path):
    raw = pd.DataFrame([
        {"FID": 1, "Operating_Unit": "M", "Supplier_Number": "S1", "Invoice_Number": "OLD", "Payment_Number": 1, "Invoice_Date": pd.Timestamp("1991-01-01"), "Payment_Date": pd.Timestamp("2024-01-10"), "Department_Name": "D", "Department_Number": 1, "Distribution_Amount": 100.0, "Check_Amount": 100.0},
        {"FID": 2, "Operating_Unit": "M", "Supplier_Number": "S1", "Invoice_Number": "FUT", "Payment_Number": 2, "Invoice_Date": pd.Timestamp("2083-01-01"), "Payment_Date": pd.Timestamp("2024-01-10"), "Department_Name": "D", "Department_Number": 1, "Distribution_Amount": 100.0, "Check_Amount": 100.0},
        {"FID": 3, "Operating_Unit": "M", "Supplier_Number": "S1", "Invoice_Number": "REV", "Payment_Number": 3, "Invoice_Date": pd.Timestamp("2024-02-10"), "Payment_Date": pd.Timestamp("2024-02-01"), "Department_Name": "D", "Department_Number": 1, "Distribution_Amount": 100.0, "Check_Amount": 100.0},
        {"FID": 4, "Operating_Unit": "M", "Supplier_Number": "S1", "Invoice_Number": "CRED", "Payment_Number": 4, "Invoice_Date": pd.Timestamp("2024-01-01"), "Payment_Date": pd.Timestamp("2024-01-05"), "Department_Name": "D", "Department_Number": 1, "Distribution_Amount": -50.0, "Check_Amount": 50.0},
        {"FID": 5, "Operating_Unit": "M", "Supplier_Number": "S1", "Invoice_Number": "OK", "Payment_Number": 5, "Invoice_Date": pd.Timestamp("2024-01-01"), "Payment_Date": pd.Timestamp("2024-01-11"), "Department_Name": "D", "Department_Number": 1, "Distribution_Amount": 100.0, "Check_Amount": 100.0},
    ])
    raw_path = tmp_path / "raw.parquet"
    raw.to_parquet(raw_path)
    stats = build_invoice_dataset(raw_path, tmp_path / "clean.parquet", tmp_path / "audit.json", tmp_path / "credit.parquet")
    clean = pd.read_parquet(tmp_path / "clean.parquet")
    credit = pd.read_parquet(tmp_path / "credit.parquet")
    assert len(clean) == 1 and clean.loc[0, "invoice_number"] == "OK"
    assert len(credit) == 1 and credit.loc[0, "invoice_number"] == "CRED"
    assert stats["dropped_bad_dates"] == 3
```

- [ ] **Step 2: Run tests to verify failure**

Run: `D:\Antigravity\projects\project_data_analysis\.venv\Scripts\pytest.exe tests/test_grain_and_cleaning.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `sql/build_invoice_grain.sql` and `src/data/build_dataset.py`**

Use DuckDB SQL over Parquet:
- Group by `(Operating_Unit, Supplier_Number, Invoice_Number, Payment_Number, Invoice_Date, Payment_Date)`.
- Filter non-null identifiers, split `invoice_amount <= 0` into `credit.parquet`, and separate rows violating `Invoice_Date BETWEEN min_valid_invoice_date AND max_valid_payment_date` or `Invoice_Date > Payment_Date` (log counts to `audit.json`).
- Compute `days_to_pay = DATEDIFF('day', invoice_date, payment_date)` and `is_late = CASE WHEN days_to_pay > 30 THEN 1 ELSE 0 END`.

- [ ] **Step 4: Run tests, execute on real `nashville_raw.parquet`, write `docs/learning/01_data_audit_and_sql.md`, and commit**

Run:
```powershell
D:\Antigravity\projects\project_data_analysis\.venv\Scripts\pytest.exe tests/test_grain_and_cleaning.py -v
D:\Antigravity\projects\project_data_analysis\.venv\Scripts\python.exe -m src.data.build_dataset
```
Record real row counts, anomaly breakdown (1991/2083 typos, credit memos, multi-line consolidation ratio) in `01_data_audit_and_sql.md` and `troubleshooting_log.md`.

```powershell
git add sql/ src/data/build_dataset.py tests/test_grain_and_cleaning.py docs/learning/
git commit -m "feat(data): resolve invoice grain via DuckDB and audit raw anomalies"
```

---

### Task 4: Statistical EDA & Hypothesis Testing (`notebooks/01_eda_statistical_audit.ipynb`)

**Files:**
- Create: `notebooks/01_eda_statistical_audit.ipynb`, `docs/learning/02_eda_and_statistics.md`

**Interfaces:**
- Consumes: `data/interim/invoices_clean.parquet`, `data/interim/invoices_audit_summary.json`
- Produces: Executed EDA notebook + summary figures (`artifacts/eda/*.png`) + `docs/learning/02_eda_and_statistics.md` documenting verified distributions, right-censoring proof (`> 60` day tail share), 30-day vs 45-day threshold sensitivity, fiscal June seasonality, and statistical hypothesis tests (Mann-Whitney U, Chi-square).

- [ ] **Step 1: Author `notebooks/01_eda_statistical_audit.ipynb`**

Cover: (1) Data Audit Funnel, (2) `days_to_pay` skew & percentiles (p50, p90, p95, p99), (3) Right-censoring verification before `2025-12-31`, (4) Late rate by threshold (30d vs 45d) & fiscal month, (5) Hypothesis tests: e.g., high-value invoices / multi-dept invoices vs payment delay, (6) Vendor concentration & Pareto.

- [ ] **Step 2: Execute notebook end-to-end on real `invoices_clean.parquet` and inspect outputs**

Run: `D:\Antigravity\projects\project_data_analysis\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda_statistical_audit.ipynb` (or install `nbconvert`/`ipykernel` if missing, or run via `papermill` / `python` script exporting PNGs). Verify all charts and exact p-values/quantiles print cleanly.

- [ ] **Step 3: Write `docs/learning/02_eda_and_statistics.md` and update `troubleshooting_log.md`, then commit**

```powershell
git add notebooks/01_eda_statistical_audit.ipynb docs/learning/
git commit -m "feat(eda): add statistical EDA notebook and hypothesis testing notes"
```

---

### Task 5: Point-in-Time Feature Engineering & Time Splits (`src/features/build_features.py`)

**Files:**
- Create: `src/features/build_features.py`, `docs/learning/03_features_and_leakage.md`
- Test: `tests/test_no_leakage.py`

**Interfaces:**
- Consumes: `data/interim/invoices_clean.parquet`, `data/interim/invoices_credit.parquet`, `src.config.SETTINGS`
- Produces:
  - `build_point_in_time_features(invoices_df: pd.DataFrame, credit_df: pd.DataFrame) -> pd.DataFrame`
  - `split_out_of_time(featured_df: pd.DataFrame, train_end: str, val_end: str, test_end: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`
  - Feature columns (strictly knowable on `invoice_date`): `log_invoice_amount`, `line_item_count`, `dept_count`, `amount_vs_vendor_median_ratio`, `vendor_prior_paid_count`, `vendor_prior_late_rate_smoothed`, `vendor_prior_avg_days_90d`, `vendor_prior_avg_days_365d`, `vendor_prior_std_days_365d`, `vendor_has_prior_credit`, `dept_inv_count_7d`, `dept_inv_count_30d`, `dept_prior_late_rate`, `same_day_vendor_inv_count`, `inv_month`, `inv_quarter`, `inv_dayofweek`, `days_to_fiscal_year_end`, plus categoricals (`operating_unit`, `primary_department`).

- [ ] **Step 1: Write failing leakage, censoring & cold-start tests in `tests/test_no_leakage.py` (Review Focus #3, #4, #5)**

```python
import numpy as np
import pandas as pd
from src.features.build_features import build_point_in_time_features, split_out_of_time

def test_features_ignore_future_payments_after_invoice_date():
    inv_a = {"operating_unit": "M", "supplier_number": "S1", "invoice_number": "A", "payment_number": 1, "invoice_date": pd.Timestamp("2024-01-01"), "payment_date": pd.Timestamp("2024-03-01"), "primary_department": "D1", "primary_dept_number": 1, "dept_count": 1, "line_item_count": 1, "invoice_amount": 100.0, "check_amount": 100.0, "days_to_pay": 60, "is_late": 1}
    inv_b = {"operating_unit": "M", "supplier_number": "S1", "invoice_number": "B", "payment_number": 2, "invoice_date": pd.Timestamp("2024-02-01"), "payment_date": pd.Timestamp("2024-02-10"), "primary_department": "D1", "primary_dept_number": 1, "dept_count": 1, "line_item_count": 1, "invoice_amount": 100.0, "check_amount": 100.0, "days_to_pay": 9, "is_late": 0}
    df_feat = build_point_in_time_features(pd.DataFrame([inv_a, inv_b]), pd.DataFrame(columns=["supplier_number", "invoice_date"]))
    row_b = df_feat.loc[df_feat["invoice_number"] == "B"].iloc[0]
    # On 2024-02-01, inv_a has NOT been paid yet (payment_date = 2024-03-01 > 2024-02-01)
    assert row_b["vendor_prior_paid_count"] == 0

def test_cold_start_vendor_finite_features():
    inv = pd.DataFrame([{"operating_unit": "M", "supplier_number": "NEW", "invoice_number": "X", "payment_number": 1, "invoice_date": pd.Timestamp("2024-05-01"), "payment_date": pd.Timestamp("2024-05-01"), "primary_department": "D1", "primary_dept_number": 1, "dept_count": 1, "line_item_count": 1, "invoice_amount": 250.0, "check_amount": 250.0, "days_to_pay": 0, "is_late": 0}])
    df_feat = build_point_in_time_features(inv, pd.DataFrame(columns=["supplier_number", "invoice_date"]))
    assert df_feat.loc[0, "vendor_prior_paid_count"] == 0
    assert np.isfinite(df_feat.loc[0, "vendor_prior_late_rate_smoothed"])
    assert np.isfinite(df_feat.loc[0, "amount_vs_vendor_median_ratio"])

def test_time_split_enforces_censoring_cutoff():
    df = pd.DataFrame({"invoice_date": pd.to_datetime(["2023-01-01", "2024-08-01", "2025-05-01", "2025-11-15"])})
    tr, va, te = split_out_of_time(df, "2024-06-30", "2024-12-31", "2025-10-31")
    assert len(tr) == 1 and len(va) == 1 and len(te) == 1
    assert te["invoice_date"].max() <= pd.Timestamp("2025-10-31")
```

- [ ] **Step 2: Run tests to verify failure**

Run: `D:\Antigravity\projects\project_data_analysis\.venv\Scripts\pytest.exe tests/test_no_leakage.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/features/build_features.py`**

Implement using DuckDB SQL window/range joins (`prior.payment_date < curr.invoice_date`) so computation over ~500k–1M invoices runs in seconds while enforcing strict point-in-time boundaries. Add CLI entrypoint saving `train.parquet`, `val.parquet`, `test.parquet` to `data/processed/`.

- [ ] **Step 4: Run tests, generate `data/processed/*.parquet`, write `docs/learning/03_features_and_leakage.md`, and commit**

Run:
```powershell
D:\Antigravity\projects\project_data_analysis\.venv\Scripts\pytest.exe tests/test_no_leakage.py -v
D:\Antigravity\projects\project_data_analysis\.venv\Scripts\python.exe -m src.features.build_features
```
Expected: All leakage/cold-start/split tests PASS; `data/processed/{train,val,test}.parquet` created.

```powershell
git add src/features/build_features.py tests/test_no_leakage.py docs/learning/03_features_and_leakage.md
git commit -m "feat(features): add leak-free point-in-time features and time splits"
```

---

### Task 6: Dual-Track Evaluation Suite & Local Baselines (`evaluate.py`, `train_baseline.py`)

**Files:**
- Create: `src/models/evaluate.py`, `src/models/train_baseline.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `data/processed/{train,val,test}.parquet`, `src.config.SETTINGS`
- Produces:
  - `evaluate_classification(y_true: np.ndarray, y_prob: np.ndarray, invoice_amounts: np.ndarray, top_k_fractions: tuple[float, ...] = (0.05, 0.10, 0.20)) -> dict[str, Any]` (returns `pr_auc`, `roc_auc`, `brier_score`, `lift_at_k`, `precision_at_k`, `late_value_captured_at_k`)
  - `inverse_transform_days(log_preds: np.ndarray) -> np.ndarray` (applies `np.expm1` clipped at `>= 0.0`)
  - `evaluate_regression_and_cashflow(df_eval: pd.DataFrame, pred_days: np.ndarray, horizons: tuple[int, ...] = (7, 14, 30)) -> dict[str, Any]` (returns `mae`, `rmse`, `median_ae`, and weekly/horizon cash outflow comparison `actual_outflow` vs `predicted_outflow` & `mape_pct`)
  - `train_and_eval_baselines(...) -> dict[str, Any]` saving `artifacts/baseline_metrics.json`

- [ ] **Step 1: Write failing metric & cashflow tests in `tests/test_metrics.py` (Review Focus #4 & #5)**

```python
import numpy as np
import pandas as pd
from src.models.evaluate import (
    evaluate_classification,
    evaluate_regression_and_cashflow,
    inverse_transform_days,
)

def test_regression_inverse_transform_non_negative():
    preds = inverse_transform_days(np.array([-0.5, 0.0, np.log1p(15.0)]))
    assert np.all(preds >= 0.0)
    assert np.isclose(preds[1], 0.0)
    assert np.isclose(preds[2], 15.0)

def test_cashflow_aggregation_matches_predicted_dates():
    df = pd.DataFrame({
        "invoice_date": pd.to_datetime(["2025-01-01", "2025-01-01"]),
        "payment_date": pd.to_datetime(["2025-01-06", "2025-01-20"]),
        "days_to_pay": [5.0, 19.0],
        "invoice_amount": [1000.0, 2000.0],
    })
    res = evaluate_regression_and_cashflow(df, np.array([5.0, 10.0]), horizons=(7, 14, 30))
    assert np.isclose(res["mae"], 4.5)
    assert res["cashflow_horizons"]["7d"]["actual_outflow"] == 1000.0
    assert res["cashflow_horizons"]["14d"]["predicted_outflow"] == 3000.0

def test_classification_lift_and_value_captured():
    y_true = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.2, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    amts = np.array([500.0] + [100.0] * 9)
    res = evaluate_classification(y_true, y_prob, amts, top_k_fractions=(0.20,))
    assert res["top_k"]["0.2"]["precision"] == 1.0
    assert np.isclose(res["top_k"]["0.2"]["lift"], 5.0)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `D:\Antigravity\projects\project_data_analysis\.venv\Scripts\pytest.exe tests/test_metrics.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/models/evaluate.py` and `src/models/train_baseline.py`**

`train_baseline.py` trains a scikit-learn `Pipeline` (`ColumnTransformer` with `StandardScaler` + `OneHotEncoder(handle_unknown="ignore", min_frequency=50)`) for both `LogisticRegression` (`is_late`) and `Ridge` (`log1p(days_to_pay)`), evaluates on `val` & `test`, and saves `artifacts/baseline_metrics.json`.

- [ ] **Step 4: Run tests, run local baselines, record baseline numbers, and commit**

Run:
```powershell
D:\Antigravity\projects\project_data_analysis\.venv\Scripts\pytest.exe tests/test_metrics.py -v
D:\Antigravity\projects\project_data_analysis\.venv\Scripts\python.exe -m src.models.train_baseline
```
Expected: All tests PASS; `artifacts/baseline_metrics.json` populated.

```powershell
git add src/models/evaluate.py src/models/train_baseline.py tests/test_metrics.py
git commit -m "feat(models): add dual-track evaluation suite and local linear baselines"
```

---

### Task 7: GBDT Training Engine & Kaggle Notebook Prep (`train_gbdt.py`, `02_kaggle_training.ipynb`)

**Files:**
- Create: `src/models/train_gbdt.py`, `notebooks/02_kaggle_training.ipynb`
- Test: `tests/test_gbdt_smoke.py`

**Interfaces:**
- Consumes: `src.features.build_features` output schema, `src.models.evaluate`
- Produces:
  - `train_dual_gbdt_models(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: Path, n_estimators: int = 500) -> dict[str, Any]` (trains LightGBM + XGBoost Classifier with `CalibratedClassifierCV` on `val` + LightGBM + XGBoost Regressor on `log1p(days_to_pay)`, computes SHAP values on sample of `test_df`, saves models + predictions + metrics JSON + SHAP PNGs to `out_dir`).

- [ ] **Step 1: Write failing smoke test in `tests/test_gbdt_smoke.py`**

Construct a tiny 60-row synthetic fixture matching feature columns, call `train_dual_gbdt_models(..., n_estimators=10)`, and assert `gbdt_metrics.json`, `test_predictions.parquet`, and both classifier/regressor artifacts are saved and finite.

- [ ] **Step 2: Run test to verify failure**

Run: `D:\Antigravity\projects\project_data_analysis\.venv\Scripts\pytest.exe tests/test_gbdt_smoke.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/models/train_gbdt.py` and `notebooks/02_kaggle_training.ipynb`**

Keep `02_kaggle_training.ipynb` clean and self-contained (loads `/kaggle/input/.../{train,val,test}.parquet`, runs LightGBM + XGBoost tuning/calibration for both tracks, exports SHAP plots, calibration curves, and `test_predictions.parquet` to `/kaggle/working/`).

- [ ] **Step 4: Run local smoke test (without touching Kaggle!) and commit**

Run: `D:\Antigravity\projects\project_data_analysis\.venv\Scripts\pytest.exe tests/ -v`
Expected: Full local test suite PASSES.

```powershell
git add src/models/train_gbdt.py notebooks/02_kaggle_training.ipynb tests/test_gbdt_smoke.py
git commit -m "feat(models): add dual-track GBDT training engine and Kaggle notebook"
```

---

### Task 8: [GATE — CHỜ XÁC NHẬN KAGGLE] Kaggle Dataset Import, Cloud Training & Artifact Retrieval

**Files:**
- Create: `artifacts/kaggle_results/` (gitignored outputs pulled from Kaggle), `docs/learning/04_modeling_and_eval.md`
- Modify: `docs/learning/troubleshooting_log.md`

- [ ] **Step 1: Confirm with user that their active Kaggle project is complete**

HARD STOP before executing Step 2: Explicitly ask and receive confirmation from user before calling any Kaggle upload/kernel push tool.

- [ ] **Step 2: Create/upload `data/processed/{train,val,test}.parquet` as a private/public Kaggle Dataset (`dataset-metadata.json`)**
- [ ] **Step 3: Push `notebooks/02_kaggle_training.ipynb` via Kaggle API/MCP and wait for `COMPLETE` status**
- [ ] **Step 4: Download kernel outputs (`gbdt_metrics.json`, `test_predictions.parquet`, SHAP & calibration figures) into `artifacts/kaggle_results/`**
- [ ] **Step 5: Write `docs/learning/04_modeling_and_eval.md` comparing Baseline vs LightGBM vs XGBoost across both tracks + error analysis, then commit docs**

---

### Task 9: Streamlit Portfolio Dashboard & Final `README.md` (`src/app/dashboard.py`, `README.md`)

**Files:**
- Create: `src/app/dashboard.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `data/interim/invoices_audit_summary.json`, `artifacts/baseline_metrics.json`, `artifacts/kaggle_results/{gbdt_metrics.json,test_predictions.parquet,*.png}` (with graceful fallback to baseline metrics if Kaggle artifacts aren't downloaded yet).

- [ ] **Step 1: Implement `src/app/dashboard.py`**

Build a 3-tab Streamlit dashboard: (1) Data Audit & Statistical EDA, (2) Late Payment Risk Scoring (`PR-AUC`, Calibration, Lift@K, SHAP explanations, top high-risk invoices), (3) Cashflow Timing Forecast (`MAE`/`RMSE`, Actual vs Predicted 7/14/30-day Cash Outflow).

- [ ] **Step 2: Smoke-test dashboard import/headless startup and update `README.md`**

Run:
```powershell
D:\Antigravity\projects\project_data_analysis\.venv\Scripts\pytest.exe tests/ -v
```
Verify all tests pass and `README.md` links every verified metric, architecture diagram, learning note (`docs/learning/00..04`), and run command accurately.

- [ ] **Step 3: Final commit**

```powershell
git add src/app/dashboard.py README.md
git commit -m "feat(app): add dual-track AP risk & cashflow dashboard and final README"
```
