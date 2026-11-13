# 5-Zone Enterprise AP Risk & Cashflow Workbench UI Design (FastAPI + Tailwind/JS)

- **Date:** 2026-11-14
- **Status:** Draft — Pending User Approval
- **Scope:** Focused UI/API upgrade (`src/api/`, `src/web/`, `tests/test_api_workbench.py`, `README.md` + `assets/` demo screenshot)

---

## 1. Intent & Why
Replace the generic vertical-scroll dashboard impression with a pixel-accurate **5-zone Enterprise Analytics Workbench** matching the user's wireframe (`left rail`, `sidebar`, `tabs + main content`, `right side bar`, `bottom bar`). This makes the GitHub portfolio demo look and behave like a real financial SaaS/IDE product while keeping single-command Python execution for Data Science recruiters.

## 2. Goals, Non-Goals & Anti-Goals
- **Goals:**
  - Implement the exact 5-zone viewport shell (`100vh` locked desktop layout with independent scrollable panes, responsive fallback on narrow screens).
  - Serve both REST JSON endpoints and static assets (`HTML5 + Tailwind CSS + ES6 Modules + Chart.js`) from a single FastAPI app (`uvicorn src.api.main:app --reload --port 8000`).
  - Reuse `src/app/loaders.py` as the single source of truth so zero metric calculations are duplicated.
  - Capture a high-resolution screenshot of the live workbench (`assets/workbench_preview.png`) and embed it at the top of `README.md` so visitors on GitHub see the new UI immediately.
  - Preserve `src/app/dashboard.py` and all 32 existing pytest tests intact.
- **Non-Goals:**
  - No Node.js/Webpack/Vite build step required to run the project.
  - No database migration (reads existing Parquet & JSON artifacts via in-memory LRU cache).
- **Anti-Goals:**
  - Never hardcode or fabricate metrics in JS/HTML — every number must come from the FastAPI endpoints backed by `invoices_audit_summary.json`, `baseline_metrics.json`, `gbdt_metrics.json`, `test_predictions.parquet`, and `docs/learning/*.md`.

## 3. Constraints & Standards
- **Python & FastAPI (`d:\.agents\rules\python-fastapi.md`):** Python 3.11+, 100% type hints, Pydantic v2 `BaseModel` (`ConfigDict(from_attributes=True, populate_by_name=True)`), explicit `response_model` on all API routes, FastAPI `Depends()` for artifact repository injection.
- **Clean Code (`d:\.agents\rules\clean-code.md`):** Zero `print()` statements in Python or `console.log()` spam in JS; functions <= 30–40 lines.
- **Graceful Fallback:** If `artifacts/kaggle_results/` is missing, `/api/v1/overview` and `/api/v1/departments` gracefully fall back to `baseline_metrics.json` and `invoices_audit_summary.json` without HTTP 500 errors.

## 4. Approaches Considered
1. **Approach 1 (Selected): FastAPI REST Backend + Modular Vanilla ES6 JS + Tailwind CSS + Chart.js (Single-Process)**
   - *Trade-off:* Zero Node build tooling; starts with one `uvicorn` command; fast `TestClient` integration tests; complete pixel control over the 5-zone wireframe.
2. **Approach 2: FastAPI + Next.js (TypeScript) Separate App**
   - *Trade-off:* Adds Node.js toolchain, dual servers, and heavy `node_modules` overhead to a Python DS repository.

## 5. Architecture & 5-Zone Wireframe Specification

```
+------+------------------+-----------------------------------------+--------------------+
| Left | Sidebar          | [Tab 1: Audit/EDA] [Tab 2: Risk] [Tab 3]| Right Sidebar      |
| Rail | (Dept/Vendor     |-----------------------------------------| (Inspector)        |
|      |  Navigator)      | Header & Context Title                  | - System Checklist |
| [🏠] | [Search input  ] | +-----------+ +-----------+ +---------+ |   (o) Grain Audit  |
| [🔍] |                  | | Card 1    | | Card 2    | | Card 3  | |   (o) 61d Buffer   |
| [🔔] | (o) All Depts    | | Visual/KPI| | Visual/KPI| | Visual  | |   (o) Isotonic Cal |
| [📄] | (o) MNPS         | +-----------+ +-----------+ +---------+ | - Mini Chart/SHAP  |
| [⚙️] | (o) Water Svcs   | Analytical Narrative & Benchmark Table  |   [Preview Box]    |
|      | (o) NDOT         |-----------------------------------------| - Selected Entity  |
|      | (o) General Svcs | [Action Card / High-Risk Invoice Triage |   (o) Prob / MAE   |
| [👤] | (o) Sheriff      |  Row List + "Export / Inspect" CTA]     |   (o) Dollar Exp.  |
+------+------------------+-----------------------------------------+--------------------+
| Bottom Bar: Status: READY | Clean: 811,914 | Test: 195,889 | Active Model: XGBoost     |
+----------------------------------------------------------------------------------------+
```

### Zone Breakdown (Builder Calls)
1. **Left Rail (`w-14`):**
   - Icons switching active mode: `Workbench` (Home), `Invoice Triage Search`, `Risk Alerts` (high-value late exposure), `Learning Docs Drawer/Modal` (renders `docs/learning/00..04` & `troubleshooting_log.md`), and `Model/ROI Settings` + bottom profile badge.
2. **Left Sidebar (`w-64`):**
   - Search filter + scrollable list of **Departments** (from `compute_department_error_breakdown`, plus `"All Departments (Citywide)"`) showing invoice count and late-rate indicator dot. Clicking a department filters the center workspace and updates the Right Inspector.
3. **Center Workspace (`flex-1`):**
   - **Top Tabs Bar:** 3 equal-width tabs matching the wireframe (`1. Data Audit & Statistical EDA`, `2. Late Payment Risk (Track 1)`, `3. Cashflow Forecast (Track 2)`).
   - **Row of 3 Visual Cards:** Each card has a thumbnail/interactive chart area (backed by verified `/artifacts/...` PNGs with lightbox zoom + Chart.js interactive canvas) and concise title + metric subtitle underneath, matching the 3-card row in the wireframe.
   - **Middle Narrative & Benchmark Section:** Clean typography summarizing statistical findings + interactive model comparison table (`Dummy`, `Logistic/Ridge`, `LightGBM`, `XGBoost`).
   - **Bottom Action / Triage Card:** Matches the wireframe's bottom horizontal card (avatar/badge on left, summary of filtered high-risk invoices or cashflow horizon delta in center, action button & interactive invoice table selector on right).
4. **Right Sidebar (`w-80`):**
   - **Top Checklist Box:** Verification status items (`DuckDB Grain 811,914`, `Point-in-Time Strict < t`, `61-Day Censoring Buffer`, `Isotonic Calibration`).
   - **Middle Preview Box:** SHAP Top Drivers horizontal bar chart or Selected Invoice/Department risk gauge.
   - **Bottom Inspector List:** Live metrics for the currently selected Department or Invoice (`XGB Late Prob`, `Predicted vs Actual Days`, `Dollar Exposure`, `Department MAE`).
5. **Bottom Status Bar (`h-8`):**
   - Left: API & artifact mode badge (`KAGGLE_GBDT_ACTIVE` or `BASELINE_FALLBACK`), dataset split counts.
   - Right: Active model selector (`XGBoost` / `LightGBM` / `Baseline`) and test execution timestamp.

## 6. FastAPI Endpoints (`src/api/`)
- `GET /api/v1/health` -> `HealthResponse` (status, kaggle_ready, split counts)
- `GET /api/v1/overview` -> `OverviewResponse` (audit funnel, splits summary, classification & regression benchmark tables, SHAP top features, available plot URLs)
- `GET /api/v1/departments?min_invoices=50` -> `DepartmentListResponse` (per-department volume, late rate, XGB/LGB MAE & RMSE for left sidebar + right inspector)
- `GET /api/v1/invoices/triage?department=...&min_amount=...&prob_threshold=...&model=xgb&limit=50` -> `InvoiceTriageResponse` (filtered high-risk invoices + summary stats)
- `GET /api/v1/cashflow/series?department=...` -> `CashflowSeriesResponse` (weekly actual vs LGB/XGB outflow series + 7d/14d/30d/60d horizon comparison)
- `GET /api/v1/docs` -> `LearningDocsListResponse` & `GET /api/v1/docs/{slug}` -> `LearningDocDetailResponse` (serves `docs/learning/*.md` content for in-app study)
- Static mounts: `/artifacts` -> `artifacts/`, `/` -> `src/web/static/index.html`.
