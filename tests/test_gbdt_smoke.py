import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_CLASSIFICATION,
    TARGET_REGRESSION,
)
from src.models.train_gbdt import train_dual_gbdt_models


def _make_synthetic_split(
    seed: int,
    n_samples: int = 60,
    start_date: str = "2024-01-01",
    include_unseen_category: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    inv_dates = pd.date_range(start_date, periods=n_samples, freq="D")

    risk_signal = rng.normal(0.0, 1.0, size=n_samples)
    days_to_pay = np.clip(np.round(28.0 + 12.0 * risk_signal + rng.normal(0.0, 4.0, n_samples)), 1, 90)
    is_late = (days_to_pay > 30).astype(int)
    pay_dates = inv_dates + pd.to_timedelta(days_to_pay.astype(int), unit="D")

    data: dict[str, object] = {
        "invoice_number": [f"INV-{seed}-{i:04d}" for i in range(n_samples)],
        "supplier_number": [f"SUP-{i % 12:03d}" for i in range(n_samples)],
        "invoice_date": inv_dates,
        "payment_date": pay_dates,
        "invoice_amount": rng.uniform(250.0, 15000.0, size=n_samples),
        TARGET_REGRESSION: days_to_pay.astype(float),
        TARGET_CLASSIFICATION: is_late,
    }
    for idx, num_col in enumerate(NUMERIC_FEATURES):
        base = risk_signal * (1.5 if idx % 3 == 0 else 0.4)
        data[num_col] = base + rng.normal(5.0, 1.0, size=n_samples)

    cats = ["DEPT_FINANCE", "DEPT_PUBLIC_WORKS", "DEPT_POLICE"]
    for cat_col in CATEGORICAL_FEATURES:
        vals = rng.choice(cats, size=n_samples).tolist()
        if include_unseen_category:
            vals[0] = "UNSEEN_UNIT_999"
        data[cat_col] = vals

    return pd.DataFrame(data)


def test_train_dual_gbdt_models_smoke(tmp_path: Path) -> None:
    train_df = _make_synthetic_split(seed=10, n_samples=60, start_date="2023-07-01")
    val_df = _make_synthetic_split(seed=20, n_samples=60, start_date="2024-07-01")
    test_df = _make_synthetic_split(
        seed=30,
        n_samples=60,
        start_date="2025-01-01",
        include_unseen_category=True,
    )

    out_dir = tmp_path / "gbdt_artifacts"
    metrics = train_dual_gbdt_models(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        out_dir=out_dir,
        n_estimators=10,
    )

    metrics_path = out_dir / "gbdt_metrics.json"
    assert metrics_path.exists()
    loaded_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    meta = loaded_metrics["meta"]
    assert meta["val_calibrated_in_sample"] is True
    for model_tag in ("lgb_cls", "xgb_cls", "lgb_reg", "xgb_reg"):
        assert model_tag in meta["best_params"]
        assert isinstance(meta["best_iteration"][model_tag], int)

    for model_key in ("lgbm", "xgb"):
        assert model_key in metrics
        assert model_key in loaded_metrics
        for split_key in ("val", "test"):
            cls_metrics = loaded_metrics[model_key][split_key]["classification"]
            reg_metrics = loaded_metrics[model_key][split_key]["regression"]

            assert np.isfinite(cls_metrics["pr_auc"])
            assert np.isfinite(cls_metrics["roc_auc"])
            assert np.isfinite(cls_metrics["brier_score"])
            assert np.isfinite(cls_metrics["uncalibrated_brier_score"])
            assert np.isfinite(reg_metrics["mae"])
            assert np.isfinite(reg_metrics["rmse"])
            assert np.isfinite(reg_metrics["median_ae"])
            assert "7d" in reg_metrics["cashflow_horizons"]
            assert "14d" in reg_metrics["cashflow_horizons"]
            assert "30d" in reg_metrics["cashflow_horizons"]

    assert "shap_top_features" in loaded_metrics
    assert len(loaded_metrics["shap_top_features"]) > 0
    assert all(np.isfinite(item["mean_abs_shap"]) for item in loaded_metrics["shap_top_features"])

    preds_path = out_dir / "test_predictions.parquet"
    assert preds_path.exists()
    preds_df = pd.read_parquet(preds_path)
    assert len(preds_df) == len(test_df)

    expected_cols = [
        "invoice_number",
        "supplier_number",
        "operating_unit",
        "primary_department",
        "invoice_date",
        "payment_date",
        "invoice_amount",
        "days_to_pay",
        "is_late",
        "lgb_prob_late",
        "xgb_prob_late",
        "lgb_pred_days",
        "xgb_pred_days",
    ]
    for col in expected_cols:
        assert col in preds_df.columns

    for prob_col in ("lgb_prob_late", "xgb_prob_late"):
        vals = preds_df[prob_col].to_numpy(dtype=float)
        assert np.all(np.isfinite(vals))
        assert np.all((vals >= 0.0) & (vals <= 1.0))

    for day_col in ("lgb_pred_days", "xgb_pred_days"):
        vals = preds_df[day_col].to_numpy(dtype=float)
        assert np.all(np.isfinite(vals))
        assert np.all(vals >= 0.0)

    artifact_names = [
        "ordinal_encoder.joblib",
        "lgb_classifier.joblib",
        "lgb_calibrator.joblib",
        "xgb_classifier.joblib",
        "xgb_calibrator.joblib",
        "lgb_regressor.joblib",
        "xgb_regressor.joblib",
    ]
    for artifact in artifact_names:
        artifact_path = out_dir / artifact
        assert artifact_path.exists(), f"Missing artifact: {artifact}"
        loaded_obj = joblib.load(artifact_path)
        assert loaded_obj is not None

    figure_names = [
        "calibration_curve.png",
        "pr_curve.png",
        "shap_summary.png",
        "cashflow_forecast.png",
    ]
    for fig_name in figure_names:
        fig_path = out_dir / fig_name
        assert fig_path.exists(), f"Missing figure: {fig_name}"
        assert fig_path.stat().st_size > 0


def test_train_dual_gbdt_models_rejects_empty_split(tmp_path: Path) -> None:
    train_df = _make_synthetic_split(seed=10, n_samples=20, start_date="2023-07-01")
    empty_df = train_df.iloc[0:0].copy()
    with pytest.raises(ValueError, match="non-empty"):
        train_dual_gbdt_models(
            train_df=train_df,
            val_df=empty_df,
            test_df=train_df,
            out_dir=tmp_path / "empty_out",
            n_estimators=5,
        )
