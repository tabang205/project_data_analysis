"""Integration and schema tests for the FastAPI 5-zone AP workbench backend."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.deps import clear_artifact_cache
from src.api.main import app


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    clear_artifact_cache()


def test_health_endpoint_real_artifacts() -> None:
    client = TestClient(app)
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["kaggle_ready"] is True
    assert body["clean_invoices"] == 811914
    assert body["test_invoices"] == 195889


def test_overview_endpoint_real_grounding() -> None:
    client = TestClient(app)
    res = client.get("/api/v1/overview")
    assert res.status_code == 200
    body = res.json()
    assert body["kaggle_ready"] is True
    assert body["audit_summary"]["raw_rows"] == 1307617
    assert body["audit_summary"]["clean_invoice_rows"] == 811914
    assert len(body["splits"]) == 5

    cls_rows = [
        r
        for r in body["classification_benchmark"]
        if r["model"] == "XGBoost (Isotonic)" and r["split"] == "Test"
    ]
    assert len(cls_rows) == 1
    assert pytest.approx(cls_rows[0]["pr_auc"], abs=1e-4) == 0.4092
    assert pytest.approx(cls_rows[0]["lift_5"], abs=1e-2) == 2.83

    reg_rows = [
        r
        for r in body["regression_benchmark"]
        if r["model"] == "XGBoost Regressor" and r["split"] == "Test"
    ]
    assert len(reg_rows) == 1
    assert pytest.approx(reg_rows[0]["mae"], abs=1e-2) == 8.73
    assert pytest.approx(reg_rows[0]["mape_14d"], abs=1e-2) == 0.65

    assert len(body["shap_top_features"]) == 20
    assert body["shap_top_features"][0]["feature"] == "vendor_prior_late_rate_smoothed"
    assert len(body["eda_plots"]) == 6
    assert len(body["cls_plots"]) == 3
    assert len(body["reg_plots"]) == 1


def test_departments_and_cashflow_endpoints() -> None:
    client = TestClient(app)
    dept_res = client.get("/api/v1/departments", params={"min_invoices": 500})
    assert dept_res.status_code == 200
    dept_body = dept_res.json()
    assert dept_body["total_departments"] >= 10
    first_dept = dept_body["departments"][0]["department"]

    cf_res = client.get("/api/v1/cashflow/series", params={"department": first_dept})
    assert cf_res.status_code == 200
    cf_body = cf_res.json()
    assert cf_body["department"] == first_dept
    assert len(cf_body["weekly_series"]) > 0
    assert len(cf_body["horizons"]) == 12


def test_invoice_triage_filtering_and_validation() -> None:
    client = TestClient(app)
    res = client.get(
        "/api/v1/invoices/triage",
        params={
            "department": "Water Services",
            "min_amount": 1000.0,
            "prob_threshold": 0.45,
            "model": "xgb",
            "limit": 15,
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["summary"]["matched_count"] >= len(body["items"])
    assert len(body["items"]) <= 15
    for item in body["items"]:
        assert item["primary_department"] == "Water Services"
        assert item["invoice_amount"] >= 1000.0
        assert item["pred_prob_late"] >= 0.45
        assert item["xgb_pred_days"] is not None

    empty_res = client.get(
        "/api/v1/invoices/triage",
        params={"department": "Water Services", "prob_threshold": 0.9999},
    )
    assert empty_res.status_code == 200
    empty_body = empty_res.json()
    assert empty_body["summary"]["matched_count"] == 0
    assert empty_body["items"] == []

    bad_prob = client.get("/api/v1/invoices/triage", params={"prob_threshold": 1.5})
    assert bad_prob.status_code == 422

    bad_model = client.get("/api/v1/invoices/triage", params={"model": "random_forest"})
    assert bad_model.status_code == 422


def test_learning_docs_endpoints_and_traversal_guard() -> None:
    client = TestClient(app)
    list_res = client.get("/api/v1/docs")
    assert list_res.status_code == 200
    docs = list_res.json()["docs"]
    slugs = [d["slug"] for d in docs]
    assert "00_roadmap_and_concepts" in slugs
    assert "04_modeling_and_eval" in slugs
    assert "troubleshooting_log" in slugs

    detail_res = client.get("/api/v1/docs/04_modeling_and_eval")
    assert detail_res.status_code == 200
    assert "XGBoost" in detail_res.json()["content"]

    missing_res = client.get("/api/v1/docs/nonexistent_doc_slug")
    assert missing_res.status_code == 404


def test_fallback_mode_when_kaggle_results_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts_dir = tmp_path / "artifacts"
    interim_dir = tmp_path / "interim"
    artifacts_dir.mkdir()
    interim_dir.mkdir()

    baseline_payload = {
        "val": {
            "classification": {
                "pr_auc": 0.25,
                "roc_auc": 0.56,
                "brier_score": 0.16,
                "lift_at_k": {"0.05": 1.5, "0.10": 1.4, "0.20": 1.2},
                "value_captured_at_k": {"0.20": 0.30},
            },
            "regression": {
                "mae": 11.2,
                "rmse": 28.5,
                "median_ae": 6.1,
                "cashflow_horizons": {
                    "7d": {"mape_pct": 40.0},
                    "14d": {"mape_pct": 20.0},
                    "30d": {"mape_pct": 8.0},
                },
            },
        },
        "test": {
            "classification": {
                "pr_auc": 0.2984,
                "roc_auc": 0.5853,
                "brier_score": 0.1701,
                "lift_at_k": {"0.05": 2.01, "0.10": 1.81, "0.20": 1.50},
                "value_captured_at_k": {"0.20": 0.4038},
            },
            "regression": {
                "mae": 10.97,
                "rmse": 20.14,
                "median_ae": 6.64,
                "cashflow_horizons": {
                    "7d": {"mape_pct": 46.37},
                    "14d": {"mape_pct": 23.67},
                    "30d": {"mape_pct": 6.65},
                },
            },
        },
    }
    audit_payload = {
        "raw_rows": 1307617,
        "grouped_invoice_rows": 823841,
        "dropped_missing_keys": 0,
        "dropped_bad_dates": 57152,
        "credit_memo_rows": 11927,
        "clean_invoice_rows": 811914,
    }
    (artifacts_dir / "baseline_metrics.json").write_text(
        json.dumps(baseline_payload),
        encoding="utf-8",
    )
    (interim_dir / "invoices_audit_summary.json").write_text(
        json.dumps(audit_payload),
        encoding="utf-8",
    )
    monkeypatch.setenv("AP_ARTIFACTS_DIR", str(artifacts_dir))
    monkeypatch.setenv("AP_INTERIM_DIR", str(interim_dir))
    clear_artifact_cache()

    client = TestClient(app)
    for path in (
        "/api/v1/health",
        "/api/v1/overview",
        "/api/v1/departments",
        "/api/v1/invoices/triage",
        "/api/v1/cashflow/series",
    ):
        res = client.get(path)
        assert res.status_code == 200
    assert client.get("/api/v1/health").json()["kaggle_ready"] is False


def test_frontend_five_zone_dom_and_static_assets() -> None:
    client = TestClient(app)
    html_res = client.get("/")
    assert html_res.status_code == 200
    html = html_res.text
    for zone in ("left-rail", "sidebar", "main-content", "right-sidebar", "bottom-bar"):
        assert f'data-zone="{zone}"' in html
    for tab_id in ("tab-eda", "tab-risk", "tab-cashflow"):
        assert f'data-tab="{tab_id}"' in html

    js_res = client.get("/static/app.js")
    assert js_res.status_code == 200
    assert "console.log(" not in js_res.text

    css_res = client.get("/static/styles.css")
    assert css_res.status_code == 200

