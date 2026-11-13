"""Unit and headless Streamlit AppTest tests for the AP portfolio dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

DASHBOARD_PATH = Path(__file__).resolve().parent.parent / "src" / "app" / "dashboard.py"

from src.app.loaders import (
    build_audit_funnel_df,
    build_classification_comparison_df,
    build_regression_comparison_df,
    build_shap_top_features_df,
    compute_cashflow_horizon_comparison,
    compute_department_error_breakdown,
    compute_weekly_cashflow_series,
    filter_high_risk_invoices,
    load_audit_summary,
    load_baseline_metrics,
    load_gbdt_metrics,
    load_test_predictions,
)


def _sample_baseline_metrics() -> dict:
    cls_block = {
        "pr_auc": 0.30,
        "roc_auc": 0.60,
        "brier_score": 0.16,
        "lift_at_k": {"0.05": 2.1, "0.1": 1.8, "0.2": 1.5},
        "late_value_captured_at_k": {"0.2": 0.40},
    }
    reg_block = {
        "mae": 11.0,
        "rmse": 20.0,
        "median_ae": 6.5,
        "cashflow_horizons": {
            "7d": {"mape_pct": 45.0},
            "14d": {"mape_pct": 24.0},
            "30d": {"mape_pct": 6.7},
        },
    }
    return {
        "val": {"n_samples": 100, "classification": cls_block, "regression": reg_block},
        "test": {"n_samples": 100, "classification": cls_block, "regression": reg_block},
    }


def _sample_predictions_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "invoice_number": [f"INV-{i:03d}" for i in range(6)],
            "supplier_number": ["SUP-1", "SUP-1", "SUP-2", "SUP-2", "SUP-3", "SUP-3"],
            "operating_unit": ["METRO_GG"] * 4 + ["METRO_MNPS"] * 2,
            "primary_department": ["Fire", "Fire", "MNPS", "MNPS", "Water", "Water"],
            "invoice_date": pd.to_datetime(
                ["2025-01-05", "2025-01-06", "2025-01-10", "2025-01-12", "2025-01-15", "2025-01-20"]
            ),
            "payment_date": pd.to_datetime(
                ["2025-01-12", "2025-02-20", "2025-01-24", "2025-02-11", "2025-01-25", "2025-03-01"]
            ),
            "invoice_amount": [5000.0, 25000.0, 1200.0, 8000.0, 300.0, 15000.0],
            "days_to_pay": [7, 45, 14, 30, 10, 40],
            "is_late": [0, 1, 0, 0, 0, 1],
            "lgb_prob_late": [0.12, 0.58, 0.22, 0.31, 0.09, 0.49],
            "xgb_prob_late": [0.10, 0.62, 0.19, 0.28, 0.08, 0.54],
            "lgb_pred_days": [8.0, 40.0, 14.0, 31.0, 12.0, 38.0],
            "xgb_pred_days": [7.0, 42.0, 14.0, 31.0, 11.0, 39.0],
        }
    )


def test_loaders_graceful_fallback_when_kaggle_missing(tmp_path: Path) -> None:
    interim_dir = tmp_path / "interim"
    artifacts_dir = tmp_path / "artifacts"
    interim_dir.mkdir()
    artifacts_dir.mkdir()

    baseline_path = artifacts_dir / "baseline_metrics.json"
    baseline_path.write_text(json.dumps(_sample_baseline_metrics()), encoding="utf-8")

    assert load_audit_summary(interim_dir) is None
    assert load_baseline_metrics(artifacts_dir) is not None
    assert load_gbdt_metrics(artifacts_dir) is None
    assert load_test_predictions(artifacts_dir) is None

    cls_df = build_classification_comparison_df(load_baseline_metrics(artifacts_dir), None)
    reg_df = build_regression_comparison_df(load_baseline_metrics(artifacts_dir), None)
    assert set(cls_df["Model"]) == {"Dummy (Train Prior 27.53%)", "LogisticRegression"}
    assert set(reg_df["Model"]) == {"Median Baseline (30.0d)", "Ridge (alpha=10.0)"}
    assert build_shap_top_features_df(None).empty


def test_loaders_and_transforms_with_synthetic_kaggle(tmp_path: Path) -> None:
    interim_dir = tmp_path / "interim"
    kaggle_dir = tmp_path / "artifacts" / "kaggle_results"
    interim_dir.mkdir(parents=True)
    kaggle_dir.mkdir(parents=True)

    audit = {
        "raw_rows": 1000,
        "grouped_rows": 700,
        "dropped_missing_keys": 0,
        "dropped_bad_dates": 50,
        "credit_memo_rows": 20,
        "clean_invoice_rows": 680,
    }
    (interim_dir / "invoices_audit_summary.json").write_text(json.dumps(audit), encoding="utf-8")
    loaded_audit = load_audit_summary(interim_dir)
    assert loaded_audit is not None
    funnel_df = build_audit_funnel_df(loaded_audit)
    assert funnel_df["Row Count"].iloc[0] == 1000
    assert funnel_df["Row Count"].iloc[-1] == 680

    preds_df = _sample_predictions_df()
    preds_df.to_parquet(kaggle_dir / "test_predictions.parquet", index=False)
    loaded_preds = load_test_predictions(tmp_path / "artifacts")
    assert loaded_preds is not None
    assert len(loaded_preds) == 6

    top_df, summary = filter_high_risk_invoices(
        preds_df=loaded_preds,
        model_col="xgb_prob_late",
        departments=["Fire", "Water"],
        min_amount=1000.0,
        prob_threshold=0.50,
        top_n=10,
    )
    assert summary["matched_count"] == 2
    assert summary["actual_late_count"] == 2
    assert summary["actual_late_rate"] == pytest.approx(1.0)
    assert top_df["invoice_number"].tolist() == ["INV-001", "INV-005"]

    empty_df, empty_summary = filter_high_risk_invoices(
        preds_df=loaded_preds,
        model_col="xgb_prob_late",
        departments=[],
        min_amount=999999.0,
        prob_threshold=0.95,
    )
    assert empty_df.empty
    assert empty_summary["matched_count"] == 0

    horizon_df = compute_cashflow_horizon_comparison(loaded_preds, horizons=(7, 14, 30))
    assert set(horizon_df["Horizon"]) == {"7d", "14d", "30d"}
    weekly_df = compute_weekly_cashflow_series(loaded_preds)
    assert not weekly_df.empty
    dept_err_df = compute_department_error_breakdown(loaded_preds, min_invoices=2)
    assert len(dept_err_df) == 3


def test_dashboard_apptest_smoke_real_artifacts() -> None:
    at = AppTest.from_file(DASHBOARD_PATH, default_timeout=30).run()
    assert not at.exception
    assert not at.error
    assert len(at.metric) >= 4
    assert len(at.dataframe) >= 4

    at.slider(key="triage_prob_threshold").set_value(0.5).run()
    assert not at.exception

    at.segmented_control(key="triage_model").set_value("LightGBM").run()
    assert not at.exception


def test_dashboard_apptest_graceful_fallback_without_kaggle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts_dir = tmp_path / "artifacts"
    interim_dir = tmp_path / "interim"
    artifacts_dir.mkdir()
    interim_dir.mkdir()
    (artifacts_dir / "baseline_metrics.json").write_text(
        json.dumps(_sample_baseline_metrics()),
        encoding="utf-8",
    )

    monkeypatch.setenv("AP_ARTIFACTS_DIR", str(artifacts_dir))
    monkeypatch.setenv("AP_INTERIM_DIR", str(interim_dir))

    at = AppTest.from_file(DASHBOARD_PATH, default_timeout=20).run()
    assert not at.exception
    assert not at.error
    assert any("Kaggle GBDT artifacts" in info.value for info in at.info)
