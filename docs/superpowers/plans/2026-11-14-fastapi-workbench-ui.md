# 5-Zone Enterprise AP Workbench UI (FastAPI + Tailwind/JS) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pixel-accurate 5-zone Enterprise Analytics Workbench (`left rail`, `sidebar`, `tabs + main content`, `right side bar`, `bottom bar`) powered by FastAPI + Tailwind CSS + Vanilla ES6 JS + Chart.js, tested with `pytest`, and showcased at the top of `README.md`.

**Architecture:** FastAPI (`src/api/`) wraps existing analytical functions in `src/app/loaders.py` via cached dependency injection and exposes typed REST JSON endpoints under `/api/v1/*` plus static mounts for `/artifacts` and `src/web/static/`. The single-page frontend (`src/web/static/index.html`, `app.js`, `styles.css`) implements the 5-zone `100vh` workbench wireframe and renders interactive Chart.js visuals, verified PNG diagnostics, department/invoice triage, and an in-app `docs/learning/` reader.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, Uvicorn, `httpx` (`TestClient`), Tailwind CSS (CDN + custom CSS tokens), Chart.js, Vanilla ES6 Modules.

**Spec:** `docs/superpowers/specs/2026-11-14-fastapi-workbench-ui-design.md`

## Global Constraints

- Target Python 3.11+ with 100% type hints (`PEP 484`); never use raw `Any` in Pydantic response models.
- All Pydantic v2 schemas in `src/api/schemas.py` must inherit `BaseModel` with `model_config = ConfigDict(from_attributes=True, populate_by_name=True)`.
- Every FastAPI route must declare an explicit `response_model`.
- Zero `print()` statements in `src/` or `tests/` and zero `console.log()` debug spam in `src/web/static/app.js`.
- Reuse `src/app/loaders.py` functions — never duplicate metric math or hardcode metric numbers in HTML/JS.
- Preserve `src/app/dashboard.py` and all 32 existing pytest tests passing.

## Review Focus

1. **Missing Kaggle artifacts fallback (`AP_ARTIFACTS_DIR` without `kaggle_results/`):** `/api/v1/health`, `/api/v1/overview`, `/api/v1/departments`, `/api/v1/invoices/triage`, and `/api/v1/cashflow/series` return HTTP 200 with `kaggle_ready=False` and empty collections instead of HTTP 500.
2. **Learning doc slug allowlist (`/api/v1/docs/{slug}`):** Slugs outside existing `docs/learning/*.md` filenames (or traversal attempts) return HTTP 404.
3. **Query validation on `/api/v1/invoices/triage`:** Invalid `prob_threshold` outside `[0.0, 1.0]`, `limit` outside `[1, 200]`, or `model` not in `("xgb", "lgb")` returns HTTP 422.
4. **Department filter with spaces & high threshold:** Filtering `/api/v1/invoices/triage` by a department name containing spaces (e.g. `"Water Services"`) or a threshold yielding 0 matches returns valid `InvoiceTriageResponse` with `matched_count=0` and `items=[]`.
5. **5-Zone DOM & static asset contract (`GET /` and `/static/app.js`):** Root HTML renders all 5 required structural regions (`data-zone="left-rail"`, `data-zone="sidebar"`, `data-zone="main-content"`, `data-zone="right-sidebar"`, `data-zone="bottom-bar"`) and `app.js` contains zero `console.log(` calls.

---

### Task 1: FastAPI Workbench Backend, Pydantic v2 Schemas & API Tests

**Files:**
- Modify: `pyproject.toml`, `requirements.txt`
- Create: `src/api/__init__.py`
- Create: `src/api/schemas.py`
- Create: `src/api/deps.py`
- Create: `src/api/main.py`
- Test: `tests/test_api_workbench.py`

**Interfaces:**
- Consumes from `src/app/loaders.py`:
  - `resolve_data_dirs(artifacts_dir: Path | None = None, interim_dir: Path | None = None) -> tuple[Path, Path]`
  - `load_audit_summary(interim_dir: Path) -> dict[str, Any] | None`
  - `load_baseline_metrics(artifacts_dir: Path) -> dict[str, Any] | None`
  - `load_gbdt_metrics(artifacts_dir: Path) -> dict[str, Any] | None`
  - `load_test_predictions(artifacts_dir: Path) -> pd.DataFrame | None`
  - `build_audit_funnel_df(audit_summary: dict[str, Any]) -> pd.DataFrame`
  - `build_splits_summary_df() -> pd.DataFrame`
  - `build_classification_comparison_df(baseline_metrics, gbdt_metrics) -> pd.DataFrame`
  - `build_regression_comparison_df(baseline_metrics, gbdt_metrics) -> pd.DataFrame`
  - `build_shap_top_features_df(gbdt_metrics) -> pd.DataFrame`
  - `filter_high_risk_invoices(preds_df, department, min_amount, prob_threshold, model_col, top_n) -> tuple[pd.DataFrame, dict[str, float | int]]`
  - `compute_cashflow_horizon_comparison(preds_df, horizons) -> pd.DataFrame`
  - `compute_weekly_cashflow_series(preds_df, departments) -> pd.DataFrame`
  - `compute_department_error_breakdown(preds_df, min_invoices) -> pd.DataFrame`
- Produces in `src/api/main.py`:
  - `app: FastAPI`
  - `GET /api/v1/health` -> `HealthResponse`
  - `GET /api/v1/overview` -> `OverviewResponse`
  - `GET /api/v1/departments` -> `DepartmentListResponse`
  - `GET /api/v1/invoices/triage` -> `InvoiceTriageResponse`
  - `GET /api/v1/cashflow/series` -> `CashflowSeriesResponse`
  - `GET /api/v1/docs` -> `LearningDocsListResponse`
  - `GET /api/v1/docs/{slug}` -> `LearningDocDetailResponse`

- [ ] **Step 1: Install `fastapi` and `httpx` in `.venv` and record in `pyproject.toml` / `requirements.txt`**

Run:
```powershell
D:\Antigravity\projects\project_data_analysis\.venv\Scripts\pip.exe install fastapi httpx
```

- [ ] **Step 2: Write failing API tests in `tests/test_api_workbench.py`**

Cover all 7 endpoints, real artifact grounding (`clean_invoice_rows == 811914`, `XGBoost (Isotonic)` test `PR-AUC` `0.4092`, `14d` MAPE `0.65`), and Review Focus items 1–4 (`AP_ARTIFACTS_DIR` fallback without `kaggle_results/`, `/api/v1/docs/{slug}` 404 guard, `/api/v1/invoices/triage` 422 validation, and empty/spaced department filtering).

- [ ] **Step 3: Run `pytest tests/test_api_workbench.py -v` to confirm failure**

Run: `D:\Antigravity\projects\project_data_analysis\.venv\Scripts\pytest.exe tests/test_api_workbench.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.api'`)

- [ ] **Step 4: Implement `src/api/schemas.py`, `src/api/deps.py`, and `src/api/main.py`**

Implement strict Pydantic v2 models (`ConfigDict(from_attributes=True, populate_by_name=True)`), `functools.lru_cache` artifact store in `src/api/deps.py` keyed by `(str(artifacts_dir), str(interim_dir))`, and FastAPI routes in `src/api/main.py` including static mounts for `/artifacts` and `/static` (with a placeholder `src/web/static/index.html` if not yet created).

- [ ] **Step 5: Run `pytest tests/test_api_workbench.py -v` to verify all backend tests pass**

Run: `D:\Antigravity\projects\project_data_analysis\.venv\Scripts\pytest.exe tests/test_api_workbench.py -v`
Expected: PASS

- [ ] **Step 6: Commit Task 1**

```powershell
git add pyproject.toml requirements.txt src/api/ tests/test_api_workbench.py
git commit -m "feat(api): add FastAPI workbench endpoints and Pydantic v2 schemas"
```

---

### Task 2: 5-Zone Enterprise Workbench Frontend, Screenshot & `README.md`

**Files:**
- Create: `src/web/static/index.html`
- Create: `src/web/static/styles.css`
- Create: `src/web/static/app.js`
- Create: `assets/workbench_preview.png`
- Modify: `tests/test_api_workbench.py`
- Modify: `README.md`

**Interfaces:**
- Consumes REST JSON from Task 1 (`/api/v1/overview`, `/api/v1/departments`, `/api/v1/invoices/triage`, `/api/v1/cashflow/series`, `/api/v1/docs`, `/api/v1/docs/{slug}`, `/artifacts/*`).
- Exposes `GET /` returning `src/web/static/index.html` with 5 semantic zones (`data-zone="left-rail"`, `data-zone="sidebar"`, `data-zone="main-content"`, `data-zone="right-sidebar"`, `data-zone="bottom-bar"`).

- [ ] **Step 1: Add frontend contract tests (Review Focus #5) to `tests/test_api_workbench.py`**

Assert `GET /` returns HTTP 200 HTML containing all 5 `data-zone="..."` attributes, all 3 tab triggers, and `/static/app.js` + `/static/styles.css` return HTTP 200 with zero `console.log(` calls.

- [ ] **Step 2: Run `pytest tests/test_api_workbench.py -v` to confirm the new frontend tests fail**

Run: `D:\Antigravity\projects\project_data_analysis\.venv\Scripts\pytest.exe tests/test_api_workbench.py -v`
Expected: FAIL on missing 5-zone markup / static JS

- [ ] **Step 3: Implement `src/web/static/index.html`, `src/web/static/styles.css`, and `src/web/static/app.js`**

Build the 5-zone Enterprise Workbench matching the user's wireframe:
1. **`data-zone="left-rail"`:** Icon rail (`Workbench`, `Triage Search`, `Risk Alerts`, `Learning Docs`, `Settings` + bottom profile avatar).
2. **`data-zone="sidebar"`:** Department search + interactive department list with late-rate color dots, invoice count, and spend; clicking filters the workbench.
3. **`data-zone="main-content"`:**
   - Top 3-tab bar (`1. Data Audit & Statistical EDA`, `2. Late Payment Risk (Track 1)`, `3. Cashflow Forecast (Track 2)`).
   - Row of 3 visual cards (interactive Chart.js canvas + verified PNG thumbnails with click-to-zoom modal, title, and metric subtitle beneath each card).
   - Analytical summary & benchmark comparison table.
   - Bottom horizontal action/triage card with interactive high-risk invoice table (clicking any invoice inspects it in the right sidebar) + CSV export button.
4. **`data-zone="right-sidebar"`:**
   - System verification checklist card (`DuckDB Grain 811,914`, `Strict Point-in-Time < t`, `61d Censoring Buffer`, `Isotonic Calibration`).
   - Mini chart preview box (SHAP Top 5 horizontal bar chart / selected invoice risk gauge).
   - Selected Department or Invoice Inspector list (`XGB Late Prob`, `Pred vs Actual Days`, `Dollar Exposure`, `Department MAE`).
5. **`data-zone="bottom-bar"`:** Status badge, split counts (`Clean: 811,914 | Test: 195,889`), active model toggle (`XGBoost Isotonic` / `LightGBM Isotonic`), and ROI assumption pill.

- [ ] **Step 4: Run full `pytest tests/ -v`, capture live workbench screenshot to `assets/workbench_preview.png`, and update `README.md`**

Run `pytest tests/ -v` to confirm all tests pass. Launch `uvicorn src.api.main:app --port 8011`, render in headless Chromium (`playwright` or Edge/Chrome CLI `--headless --screenshot`) at `1600x960` to save `assets/workbench_preview.png`, and update `README.md` with the embedded screenshot and `uvicorn src.api.main:app --reload --port 8000` quickstart.

- [ ] **Step 5: Commit Task 2**

```powershell
git add src/web/ assets/workbench_preview.png tests/test_api_workbench.py README.md
git commit -m "feat(web): build 5-zone FastAPI + Tailwind enterprise workbench UI"
```
