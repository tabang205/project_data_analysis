"""Data loaders and pure analytical transforms for the AP Streamlit dashboard."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import SETTINGS

EDA_PLOT_SPECS: tuple[tuple[str, str], ...] = (
    ("01_data_audit_funnel.png", "Data quality audit funnel (1.31M raw -> 811.9k clean)"),
    ("02_distributions_skew.png", "Heavy right-skew in days_to_pay (Net-30 spike) & invoice_amount"),
    ("03_right_censoring_verification.png", "2025 right-censoring collapse vs 61-day cutoff buffer"),
    ("04_threshold_and_seasonality.png", "Threshold sensitivity & June 30 fiscal-year-end seasonality"),
    ("05_hypothesis_tests.png", "Formal hypothesis tests (METRO_GG vs METRO_MNPS, multi-dept)"),
    ("06_vendor_pareto.png", "Super-Pareto vendor concentration (99.73% spend in top 20%)"),
)

KAGGLE_CLS_PLOT_SPECS: tuple[tuple[str, str], ...] = (
    ("pr_curve.png", "Precision-Recall Curve (Test Hold-out)"),
    ("calibration_curve.png", "Reliability / Calibration Curve (Test Hold-out)"),
    ("shap_summary.png", "SHAP Global Feature Importance (1,000 Test Samples)"),
)

KAGGLE_REG_PLOT_SPECS: tuple[tuple[str, str], ...] = (
    ("cashflow_forecast.png", "Cashflow Horizon Outflow Comparison (7d / 14d / 30d)"),
)

# Verified offline against data/processed/{train,val,test}.parquet using src.models.evaluate.
_DUMMY_CLS_METRICS: dict[str, dict[str, float]] = {
    "Val": {
        "pr_auc": 0.19089623536309863,
        "roc_auc": 0.5,
        "brier_score": 0.16158005130412897,
        "lift_0.05": 0.9412765654764573,
        "lift_0.1": 0.9747840611784762,
        "lift_0.2": 0.9851308524426996,
        "value_captured_0.2": 0.18386172246032043,
    },
    "Test": {
        "pr_auc": 0.20759205468403025,
        "roc_auc": 0.5,
        "brier_score": 0.16908291650466842,
        "lift_0.05": 1.0136946739122492,
        "lift_0.1": 0.9966751038526334,
        "lift_0.2": 0.9972898800084764,
        "value_captured_0.2": 0.2150261863991038,
    },
}

_MEDIAN_REG_METRICS: dict[str, dict[str, float]] = {
    "Val": {
        "mae": 11.71911807818994,
        "rmse": 30.090507496877294,
        "median_ae": 2.0,
        "mape_7d": 100.0,
        "mape_14d": 100.0,
        "mape_30d": 19.738232093798594,
    },
    "Test": {
        "mae": 9.691284349810351,
        "rmse": 20.373381654708055,
        "median_ae": 2.0,
        "mape_7d": 100.0,
        "mape_14d": 100.0,
        "mape_30d": 14.990207896354182,
    },
}


def resolve_data_dirs(
    artifacts_dir: Path | None = None,
    interim_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Resolve artifacts and interim directories with optional environment overrides."""
    env_artifacts = os.environ.get("AP_ARTIFACTS_DIR")
    env_interim = os.environ.get("AP_INTERIM_DIR")

    resolved_artifacts = (
        artifacts_dir
        if artifacts_dir is not None
        else (Path(env_artifacts) if env_artifacts else SETTINGS.artifacts_dir)
    )
    resolved_interim = (
        interim_dir
        if interim_dir is not None
        else (Path(env_interim) if env_interim else SETTINGS.interim_dir)
    )
    return resolved_artifacts, resolved_interim


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)
    return data


def load_audit_summary(interim_dir: Path) -> dict[str, Any] | None:
    """Load DuckDB grain & data quality audit counts if available."""
    return _load_json_if_exists(interim_dir / "invoices_audit_summary.json")


def load_baseline_metrics(artifacts_dir: Path) -> dict[str, Any] | None:
    """Load local linear baseline metrics JSON if available."""
    return _load_json_if_exists(artifacts_dir / "baseline_metrics.json")


def load_gbdt_metrics(artifacts_dir: Path) -> dict[str, Any] | None:
    """Load Kaggle LightGBM/XGBoost metrics JSON if available."""
    return _load_json_if_exists(artifacts_dir / "kaggle_results" / "gbdt_metrics.json")


def load_test_predictions(artifacts_dir: Path) -> pd.DataFrame | None:
    """Load out-of-time test prediction parquet if available."""
    parquet_path = artifacts_dir / "kaggle_results" / "test_predictions.parquet"
    if not parquet_path.is_file():
        return None
    df = pd.read_parquet(parquet_path)
    if "invoice_date" in df.columns:
        df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    if "payment_date" in df.columns:
        df["payment_date"] = pd.to_datetime(df["payment_date"])
    return df


def list_existing_images(
    directory: Path,
    specs: tuple[tuple[str, str], ...],
) -> list[tuple[str, Path]]:
    """Return existing (caption, path) pairs preserving requested display order."""
    if not directory.is_dir():
        return []
    found: list[tuple[str, Path]] = []
    for filename, caption in specs:
        candidate = directory / filename
        if candidate.is_file():
            found.append((caption, candidate))
    return found


def build_audit_funnel_df(audit: dict[str, Any]) -> pd.DataFrame:
    """Convert raw audit summary JSON into a funnel table."""
    raw_rows = int(audit["raw_rows"])
    dropped_missing = int(audit["dropped_missing_keys"])
    dropped_bad_dates = int(audit["dropped_bad_dates"])
    valid_raw = raw_rows - dropped_missing - dropped_bad_dates
    grouped_rows = int(audit["grouped_rows"])
    credit_memos = int(audit["credit_memo_rows"])
    clean_rows = int(audit["clean_invoice_rows"])

    stages = [
        ("1. Raw ArcGIS Feed Rows", raw_rows, "Raw line-item distribution records"),
        ("2. Missing Business Keys Dropped", dropped_missing, "All 6 invoice keys 100% populated"),
        ("3. Bad / Out-of-Bounds Dates Dropped", dropped_bad_dates, "Years 1991/2083 or Invoice_Date > Payment_Date"),
        ("4. Valid Raw Distribution Rows", valid_raw, "Passed temporal & key integrity gates"),
        ("5. Consolidated Invoice Grain", grouped_rows, "Grouped by 6-key business ID (~1.52 lines/invoice)"),
        ("6. Credit Memos Separated", credit_memos, "invoice_amount <= 0 routed to credit audit"),
        ("7. Clean Positive Invoices", clean_rows, "invoice_amount > 0 used for ML splits"),
    ]
    return pd.DataFrame(
        {
            "Stage": [s[0] for s in stages],
            "Row Count": [s[1] for s in stages],
            "Share of Raw (%)": [round((s[1] / raw_rows) * 100.0, 2) if raw_rows > 0 else 0.0 for s in stages],
            "Notes": [s[2] for s in stages],
        }
    )


def build_splits_summary_df() -> pd.DataFrame:
    """Return the verified Out-of-Time split design and boundary audit table."""
    rows = [
        {
            "Segment": "Left-Truncation Excluded",
            "Date Window": "2015-01-16 to 2022-06-30",
            "Invoices": 21329,
            "Late Rate (%)": 52.40,
            "Purpose": "Excluded from Train (payment ledger starts 2022-07-01); retained as prior history",
        },
        {
            "Segment": "Train",
            "Date Window": f"2022-07-01 to {SETTINGS.train_end}",
            "Invoices": 456414,
            "Late Rate (%)": 27.53,
            "Purpose": "2 full fiscal years for model fitting (4.47% cold-start)",
        },
        {
            "Segment": "Validation",
            "Date Window": f"2024-07-01 to {SETTINGS.val_end}",
            "Invoices": 115974,
            "Late Rate (%)": 19.09,
            "Purpose": "Early stopping, hyperparameter selection & isotonic calibration (2.17% cold-start)",
        },
        {
            "Segment": "Test (Hold-out)",
            "Date Window": f"2025-01-01 to {SETTINGS.test_end}",
            "Invoices": 195889,
            "Late Rate (%)": 20.76,
            "Purpose": "Final out-of-time evaluation ($450.44M late exposure, 1.97% cold-start)",
        },
        {
            "Segment": "Right-Censoring Buffer",
            "Date Window": f"2025-11-01 to {SETTINGS.max_valid_payment_date}",
            "Invoices": 22308,
            "Late Rate (%)": 6.24,
            "Purpose": "61-day observation buffer (>= p90=61d) preventing Nov/Dec truncation bias",
        },
    ]
    return pd.DataFrame(rows)


def _extract_cls_row(
    split_label: str,
    model_name: str,
    cls_dict: dict[str, Any],
) -> dict[str, Any]:
    lift_at_k = cls_dict.get("lift_at_k", {})
    val_cap_at_k = cls_dict.get("late_value_captured_at_k", {})
    return {
        "Split": split_label,
        "Model": model_name,
        "PR-AUC": round(float(cls_dict["pr_auc"]), 4),
        "ROC-AUC": round(float(cls_dict["roc_auc"]), 4),
        "Brier Score": round(float(cls_dict["brier_score"]), 4),
        "Lift@5%": round(float(lift_at_k.get("0.05", np.nan)), 2),
        "Lift@10%": round(float(lift_at_k.get("0.1", np.nan)), 2),
        "Lift@20%": round(float(lift_at_k.get("0.2", np.nan)), 2),
        "Value Captured@20% (%)": round(float(val_cap_at_k.get("0.2", np.nan)) * 100.0, 2),
    }


def build_classification_comparison_df(
    baseline: dict[str, Any] | None,
    gbdt: dict[str, Any] | None,
) -> pd.DataFrame:
    """Assemble Track 1 classification benchmark table across Val and Test splits."""
    rows: list[dict[str, Any]] = []
    for split_key, split_label in (("val", "Val"), ("test", "Test")):
        dummy = _DUMMY_CLS_METRICS[split_label]
        rows.append(
            {
                "Split": split_label,
                "Model": "Dummy (Train Prior 27.53%)",
                "PR-AUC": round(dummy["pr_auc"], 4),
                "ROC-AUC": round(dummy["roc_auc"], 4),
                "Brier Score": round(dummy["brier_score"], 4),
                "Lift@5%": round(dummy["lift_0.05"], 2),
                "Lift@10%": round(dummy["lift_0.1"], 2),
                "Lift@20%": round(dummy["lift_0.2"], 2),
                "Value Captured@20% (%)": round(dummy["value_captured_0.2"] * 100.0, 2),
            }
        )
        if baseline and split_key in baseline:
            rows.append(
                _extract_cls_row(
                    split_label,
                    "LogisticRegression",
                    baseline[split_key]["classification"],
                )
            )
        if gbdt:
            for model_key, model_label in (
                ("lgbm", "LightGBM (Isotonic)"),
                ("xgb", "XGBoost (Isotonic)"),
            ):
                if model_key in gbdt and split_key in gbdt[model_key]:
                    rows.append(
                        _extract_cls_row(
                            split_label,
                            model_label,
                            gbdt[model_key][split_key]["classification"],
                        )
                    )
    return pd.DataFrame(rows)


def _extract_reg_row(
    split_label: str,
    model_name: str,
    reg_dict: dict[str, Any],
) -> dict[str, Any]:
    horizons = reg_dict.get("cashflow_horizons", {})
    return {
        "Split": split_label,
        "Model": model_name,
        "MAE (days)": round(float(reg_dict["mae"]), 2),
        "RMSE (days)": round(float(reg_dict["rmse"]), 2),
        "MedianAE (days)": round(float(reg_dict["median_ae"]), 2),
        "7d MAPE (%)": round(float(horizons.get("7d", {}).get("mape_pct", np.nan)), 2),
        "14d MAPE (%)": round(float(horizons.get("14d", {}).get("mape_pct", np.nan)), 2),
        "30d MAPE (%)": round(float(horizons.get("30d", {}).get("mape_pct", np.nan)), 2),
    }


def build_regression_comparison_df(
    baseline: dict[str, Any] | None,
    gbdt: dict[str, Any] | None,
) -> pd.DataFrame:
    """Assemble Track 2 regression and cashflow horizon benchmark table."""
    rows: list[dict[str, Any]] = []
    for split_key, split_label in (("val", "Val"), ("test", "Test")):
        median_m = _MEDIAN_REG_METRICS[split_label]
        rows.append(
            {
                "Split": split_label,
                "Model": "Median Baseline (30.0d)",
                "MAE (days)": round(median_m["mae"], 2),
                "RMSE (days)": round(median_m["rmse"], 2),
                "MedianAE (days)": round(median_m["median_ae"], 2),
                "7d MAPE (%)": round(median_m["mape_7d"], 2),
                "14d MAPE (%)": round(median_m["mape_14d"], 2),
                "30d MAPE (%)": round(median_m["mape_30d"], 2),
            }
        )
        if baseline and split_key in baseline:
            rows.append(
                _extract_reg_row(
                    split_label,
                    "Ridge (alpha=10.0)",
                    baseline[split_key]["regression"],
                )
            )
        if gbdt:
            for model_key, model_label in (
                ("lgbm", "LightGBM Regressor"),
                ("xgb", "XGBoost Regressor"),
            ):
                if model_key in gbdt and split_key in gbdt[model_key]:
                    rows.append(
                        _extract_reg_row(
                            split_label,
                            model_label,
                            gbdt[model_key][split_key]["regression"],
                        )
                    )
    return pd.DataFrame(rows)


def build_shap_top_features_df(gbdt: dict[str, Any] | None) -> pd.DataFrame:
    """Build ranked SHAP global feature importance table from gbdt_metrics.json."""
    if not gbdt or "shap_top_features" not in gbdt:
        return pd.DataFrame(columns=["Rank", "Feature", "Mean |SHAP|"])
    items = gbdt["shap_top_features"]
    return pd.DataFrame(
        [
            {
                "Rank": idx,
                "Feature": str(entry["feature"]),
                "Mean |SHAP|": round(float(entry["mean_abs_shap"]), 4),
            }
            for idx, entry in enumerate(items, start=1)
        ]
    )


def filter_high_risk_invoices(
    preds_df: pd.DataFrame,
    model_col: str,
    departments: list[str],
    min_amount: float,
    prob_threshold: float,
    top_n: int = 100,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Filter test predictions for high-risk AP triage and compute exposure summary."""
    mask = (preds_df["invoice_amount"] >= min_amount) & (preds_df[model_col] >= prob_threshold)
    if departments:
        mask = mask & preds_df["primary_department"].isin(departments)

    filtered = preds_df.loc[mask].copy()
    n_matched = int(len(filtered))
    if n_matched == 0:
        empty = pd.DataFrame(
            columns=[
                "invoice_number",
                "supplier_number",
                "primary_department",
                "invoice_date",
                "invoice_amount",
                "pred_prob_late",
                "expected_late_exposure",
                "is_late",
                "days_to_pay",
            ]
        )
        return empty, {
            "matched_count": 0,
            "total_exposure": 0.0,
            "actual_late_count": 0,
            "actual_late_rate": 0.0,
            "actual_late_exposure": 0.0,
        }

    total_exposure = float(filtered["invoice_amount"].sum())
    actual_late_mask = filtered["is_late"] == 1
    actual_late_count = int(actual_late_mask.sum())
    actual_late_rate = float(actual_late_count / n_matched)
    actual_late_exposure = float(filtered.loc[actual_late_mask, "invoice_amount"].sum())

    top_df = filtered.sort_values(
        by=[model_col, "invoice_amount"],
        ascending=[False, False],
    ).head(top_n)

    display_df = pd.DataFrame(
        {
            "invoice_number": top_df["invoice_number"],
            "supplier_number": top_df["supplier_number"],
            "primary_department": top_df["primary_department"],
            "invoice_date": top_df["invoice_date"],
            "invoice_amount": top_df["invoice_amount"],
            "pred_prob_late": top_df[model_col],
            "expected_late_exposure": top_df[model_col] * top_df["invoice_amount"],
            "is_late": top_df["is_late"].astype(bool),
            "days_to_pay": top_df["days_to_pay"],
        }
    ).reset_index(drop=True)

    return display_df, {
        "matched_count": n_matched,
        "total_exposure": total_exposure,
        "actual_late_count": actual_late_count,
        "actual_late_rate": actual_late_rate,
        "actual_late_exposure": actual_late_exposure,
    }


def compute_cashflow_horizon_comparison(
    preds_df: pd.DataFrame,
    horizons: tuple[int, ...] = (7, 14, 30, 60),
) -> pd.DataFrame:
    """Compute actual vs LightGBM vs XGBoost cumulative cash outflow ($M) by horizon."""
    actual_diff = (preds_df["payment_date"] - preds_df["invoice_date"]).dt.days.to_numpy()
    lgb_diff = np.round(preds_df["lgb_pred_days"].to_numpy(dtype=np.float64))
    xgb_diff = np.round(preds_df["xgb_pred_days"].to_numpy(dtype=np.float64))
    amts = preds_df["invoice_amount"].to_numpy(dtype=np.float64)

    rows: list[dict[str, Any]] = []
    for h in horizons:
        actual_sum = float(np.sum(amts[actual_diff <= h]))
        lgb_sum = float(np.sum(amts[lgb_diff <= h]))
        xgb_sum = float(np.sum(amts[xgb_diff <= h]))

        for series_name, val in (
            ("Actual Outflow", actual_sum),
            ("LightGBM Predicted", lgb_sum),
            ("XGBoost Predicted", xgb_sum),
        ):
            mape = (abs(actual_sum - val) / actual_sum * 100.0) if actual_sum > 0 else 0.0
            rows.append(
                {
                    "Horizon": f"{h}d",
                    "Horizon Days": h,
                    "Series": series_name,
                    "Outflow ($M)": round(val / 1e6, 2),
                    "MAPE (%)": round(mape, 2),
                }
            )
    return pd.DataFrame(rows)


def compute_weekly_cashflow_series(
    preds_df: pd.DataFrame,
    departments: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate weekly cash outflow ($M) comparing actual payment date vs predicted dates."""
    work_df = (
        preds_df[preds_df["primary_department"].isin(departments)]
        if departments
        else preds_df
    )
    if work_df.empty:
        return pd.DataFrame(columns=["Payment Week", "Series", "Outflow ($M)"])

    inv_dates = pd.to_datetime(work_df["invoice_date"])
    actual_dates = pd.to_datetime(work_df["payment_date"])
    lgb_dates = inv_dates + pd.to_timedelta(
        np.round(work_df["lgb_pred_days"].to_numpy(dtype=np.float64)),
        unit="D",
    )
    xgb_dates = inv_dates + pd.to_timedelta(
        np.round(work_df["xgb_pred_days"].to_numpy(dtype=np.float64)),
        unit="D",
    )
    amts = work_df["invoice_amount"].to_numpy(dtype=np.float64) / 1e6

    frames = []
    for label, dates in (
        ("Actual Outflow", actual_dates),
        ("LightGBM Predicted", lgb_dates),
        ("XGBoost Predicted", xgb_dates),
    ):
        weekly = (
            pd.DataFrame({"Payment Week": dates.dt.to_period("W-MON").dt.start_time, "Outflow ($M)": amts})
            .groupby("Payment Week", as_index=False)["Outflow ($M)"]
            .sum()
        )
        weekly["Series"] = label
        frames.append(weekly)

    combined = pd.concat(frames, ignore_index=True)
    # Clip predicted spillover tail beyond the observed payment cutoff so weekly comparison stays aligned.
    max_actual_week = actual_dates.dt.to_period("W-MON").dt.start_time.max()
    min_actual_week = actual_dates.dt.to_period("W-MON").dt.start_time.min()
    return combined[
        (combined["Payment Week"] >= min_actual_week) & (combined["Payment Week"] <= max_actual_week)
    ].sort_values("Payment Week").reset_index(drop=True)


def compute_department_error_breakdown(
    preds_df: pd.DataFrame,
    min_invoices: int = 500,
) -> pd.DataFrame:
    """Compute per-department late rate, actual vs predicted days, and MAE/RMSE breakdown."""
    if preds_df.empty:
        return pd.DataFrame()

    df = preds_df.copy()
    df["xgb_abs_err"] = (df["days_to_pay"] - df["xgb_pred_days"]).abs()
    df["lgb_abs_err"] = (df["days_to_pay"] - df["lgb_pred_days"]).abs()
    df["xgb_sq_err"] = (df["days_to_pay"] - df["xgb_pred_days"]) ** 2

    grouped = (
        df.groupby("primary_department", as_index=False)
        .agg(
            Invoices=("invoice_number", "count"),
            Total_Spend_M=("invoice_amount", lambda s: float(s.sum()) / 1e6),
            Actual_Late_Rate=("is_late", lambda s: float(s.mean()) * 100.0),
            XGB_Avg_Prob=("xgb_prob_late", lambda s: float(s.mean()) * 100.0),
            Actual_Avg_Days=("days_to_pay", "mean"),
            XGB_Pred_Days=("xgb_pred_days", "mean"),
            LGB_MAE_Days=("lgb_abs_err", "mean"),
            XGB_MAE_Days=("xgb_abs_err", "mean"),
            XGB_MSE_Days=("xgb_sq_err", "mean"),
        )
    )
    grouped = grouped[grouped["Invoices"] >= min_invoices].copy()
    if grouped.empty:
        return grouped

    grouped["XGB_RMSE_Days"] = np.sqrt(grouped.pop("XGB_MSE_Days"))
    grouped = grouped.rename(
        columns={
            "primary_department": "Department",
            "Total_Spend_M": "Total Spend ($M)",
            "Actual_Late_Rate": "Actual Late Rate (%)",
            "XGB_Avg_Prob": "XGB Avg Prob (%)",
            "Actual_Avg_Days": "Actual Avg Days",
            "XGB_Pred_Days": "XGB Pred Avg Days",
            "LGB_MAE_Days": "LGB MAE (days)",
            "XGB_MAE_Days": "XGB MAE (days)",
            "XGB_RMSE_Days": "XGB RMSE (days)",
        }
    )
    return grouped.sort_values("XGB MAE (days)", ascending=False).reset_index(drop=True)
