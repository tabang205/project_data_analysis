import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLS,
    NUMERIC_FEATURES,
    TARGET_CLASSIFICATION,
    TARGET_REGRESSION,
)
from src.models.evaluate import (
    evaluate_classification,
    evaluate_regression_and_cashflow,
    inverse_transform_days,
)
from src.models.train_baseline import train_and_eval_baselines


def test_regression_inverse_transform_non_negative() -> None:
    preds = inverse_transform_days(np.array([-0.5, 0.0, np.log1p(15.0)]))
    assert np.all(preds >= 0.0)
    assert np.isclose(preds[1], 0.0)
    assert np.isclose(preds[2], 15.0)


def test_cashflow_aggregation_matches_predicted_dates() -> None:
    df = pd.DataFrame(
        {
            "invoice_date": pd.to_datetime(["2025-01-01", "2025-01-01"]),
            "payment_date": pd.to_datetime(["2025-01-06", "2025-01-20"]),
            "days_to_pay": [5.0, 19.0],
            "invoice_amount": [1000.0, 2000.0],
        }
    )
    res = evaluate_regression_and_cashflow(df, np.array([5.0, 10.0]), horizons=(7, 14, 30))
    assert np.isclose(res["mae"], 4.5)
    assert res["cashflow_horizons"]["7d"]["actual_outflow"] == 1000.0
    assert res["cashflow_horizons"]["14d"]["predicted_outflow"] == 3000.0
    assert "rmse" in res
    assert "median_ae" in res
    assert "abs_error" in res["cashflow_horizons"]["7d"]
    assert "mape_pct" in res["cashflow_horizons"]["7d"]


def test_classification_lift_and_value_captured() -> None:
    y_true = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.2, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    amts = np.array([500.0] + [100.0] * 9)
    res = evaluate_classification(y_true, y_prob, amts, top_k_fractions=(0.20,))
    assert res["top_k"]["0.2"]["precision"] == 1.0
    assert np.isclose(res["top_k"]["0.2"]["lift"], 5.0)
    assert np.isclose(res["top_k"]["0.2"]["late_value_captured"], 1.0)
    assert "pr_auc" in res
    assert "roc_auc" in res
    assert "brier_score" in res


def test_classification_single_class_edge_case() -> None:
    y_true = np.zeros(10, dtype=int)
    y_prob = np.linspace(0.1, 0.9, 10)
    amts = np.full(10, 100.0)
    res = evaluate_classification(y_true, y_prob, amts, top_k_fractions=(0.10, 0.20))
    assert np.isnan(res["roc_auc"])
    assert np.isnan(res["pr_auc"])
    assert "brier_score" in res
    assert res["top_k"]["0.1"]["precision"] == 0.0
    assert res["top_k"]["0.1"]["lift"] == 0.0


def test_classification_business_impact_sensitivity() -> None:
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    y_prob = np.array([0.95, 0.85, 0.4, 0.3, 0.2, 0.1, 0.1, 0.05, 0.05, 0.01])
    amts = np.array([1000.0, 500.0] + [100.0] * 8)
    res = evaluate_classification(
        y_true,
        y_prob,
        amts,
        top_k_fractions=(0.10, 0.20),
        cost_ratios=(1.0, 5.0, 10.0),
    )
    sensitivity = res["business_impact_sensitivity"]
    assert len(sensitivity) == 6  # 2 fractions * 3 cost ratios
    first_entry = sensitivity[0]
    assert "review_fraction" in first_entry
    assert "cost_ratio" in first_entry
    assert "net_benefit_score" in first_entry


def test_train_and_eval_baselines_synthetic(tmp_path) -> None:
    rng = np.random.default_rng(42)
    n_samples = 60

    def make_synthetic_df() -> pd.DataFrame:
        data = {
            "invoice_date": pd.date_range("2024-01-01", periods=n_samples, freq="D"),
            "payment_date": pd.date_range("2024-01-20", periods=n_samples, freq="D"),
            "invoice_amount": rng.uniform(100.0, 5000.0, size=n_samples),
            "days_to_pay": rng.integers(1, 60, size=n_samples).astype(float),
        }
        for num_col in NUMERIC_FEATURES:
            data[num_col] = rng.normal(10.0, 2.0, size=n_samples)
        for cat_col in CATEGORICAL_FEATURES:
            data[cat_col] = rng.choice(["DEPT_A", "DEPT_B", "DEPT_C"], size=n_samples)
        df = pd.DataFrame(data)
        df[TARGET_CLASSIFICATION] = (df["days_to_pay"] > 30).astype(int)
        df[TARGET_REGRESSION] = df["days_to_pay"]
        return df

    train_df = make_synthetic_df()
    val_df = make_synthetic_df()
    test_df = make_synthetic_df()

    artifacts_dir = tmp_path / "artifacts"
    metrics = train_and_eval_baselines(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        artifacts_dir=artifacts_dir,
    )

    assert (artifacts_dir / "baseline_metrics.json").exists()
    assert (artifacts_dir / "baseline_classifier.joblib").exists()
    assert (artifacts_dir / "baseline_regressor.joblib").exists()
    assert "val" in metrics
    assert "test" in metrics
    assert "classification" in metrics["val"]
    assert "regression" in metrics["val"]
