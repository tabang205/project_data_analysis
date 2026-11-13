import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def inverse_transform_days(log_preds: np.ndarray) -> np.ndarray:
    """Convert log1p-transformed day predictions back to raw days, clipped at >= 0.0."""
    arr = np.asarray(log_preds, dtype=np.float64)
    return np.clip(np.expm1(arr), a_min=0.0, a_max=None)


def evaluate_classification(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    invoice_amounts: np.ndarray | pd.Series,
    top_k_fractions: tuple[float, ...] = (0.05, 0.10, 0.20),
    cost_ratios: tuple[float, ...] = (1.0, 5.0, 10.0),
) -> dict[str, Any]:
    """Evaluate binary classification metrics with top-k review lift and business impact sensitivity.

    Guards single-class edge cases on ROC-AUC and PR-AUC, and avoids hardcoded single-point
    assumptions by sweeping over review capacity fractions and late penalty-to-review cost ratios.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)
    amts_arr = np.asarray(invoice_amounts, dtype=np.float64)

    n_samples = len(y_true_arr)
    unique_classes = np.unique(y_true_arr)

    if len(unique_classes) < 2:
        pr_auc = float("nan")
        roc_auc = float("nan")
    else:
        pr_auc = float(average_precision_score(y_true_arr, y_prob_arr))
        roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))

    brier_score = float(brier_score_loss(y_true_arr, y_prob_arr))

    n_pos = int(np.sum(y_true_arr))
    base_rate = float(n_pos / n_samples) if n_samples > 0 else 0.0
    total_late_amount = float(np.sum(amts_arr[y_true_arr == 1])) if n_pos > 0 else 0.0

    sort_order = np.argsort(-y_prob_arr, kind="mergesort")
    sorted_y_true = y_true_arr[sort_order]
    sorted_amts = amts_arr[sort_order]

    top_k: dict[str, dict[str, Any]] = {}
    for frac in top_k_fractions:
        k = max(1, int(round(n_samples * frac)))
        k = min(k, n_samples)
        top_y = sorted_y_true[:k]
        top_amt = sorted_amts[:k]

        tp = int(np.sum(top_y))
        precision = float(tp / k) if k > 0 else 0.0
        recall = float(tp / n_pos) if n_pos > 0 else 0.0
        lift = float(precision / base_rate) if base_rate > 0.0 else 0.0
        captured_late_amt = float(np.sum(top_amt[top_y == 1]))
        late_value_captured = (
            float(captured_late_amt / total_late_amount) if total_late_amount > 0.0 else 0.0
        )

        key = f"{frac:g}"
        top_k[key] = {
            "k": k,
            "review_fraction": float(frac),
            "precision": precision,
            "recall": recall,
            "lift": lift,
            "late_value_captured": late_value_captured,
            "captured_late_amount": captured_late_amt,
            "total_late_amount": total_late_amount,
        }

    lift_at_k = {f"{frac:g}": top_k[f"{frac:g}"]["lift"] for frac in top_k_fractions}
    precision_at_k = {f"{frac:g}": top_k[f"{frac:g}"]["precision"] for frac in top_k_fractions}
    late_value_captured_at_k = {
        f"{frac:g}": top_k[f"{frac:g}"]["late_value_captured"] for frac in top_k_fractions
    }

    business_impact_sensitivity: list[dict[str, Any]] = []
    for frac in top_k_fractions:
        key = f"{frac:g}"
        k = top_k[key]["k"]
        tp = int(np.sum(sorted_y_true[:k]))
        fp = k - tp
        late_val_pct = float(top_k[key]["late_value_captured"] * 100.0)

        for ratio in cost_ratios:
            net_benefit = float((tp * ratio) - k)
            roi = float((tp * ratio) / k) if k > 0 else 0.0
            business_impact_sensitivity.append(
                {
                    "review_fraction": float(frac),
                    "cost_ratio": float(ratio),
                    "k": k,
                    "true_positives": tp,
                    "false_positives": fp,
                    "late_value_captured_pct": late_val_pct,
                    "net_benefit_score": net_benefit,
                    "roi": roi,
                }
            )

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier_score": brier_score,
        "base_rate": base_rate,
        "top_k": top_k,
        "lift_at_k": lift_at_k,
        "precision_at_k": precision_at_k,
        "late_value_captured_at_k": late_value_captured_at_k,
        "business_impact_sensitivity": business_impact_sensitivity,
    }


def evaluate_regression_and_cashflow(
    df_eval: pd.DataFrame,
    pred_days: np.ndarray | pd.Series,
    horizons: tuple[int, ...] = (7, 14, 30),
) -> dict[str, Any]:
    """Evaluate pointwise regression metrics and aggregated cash outflow forecast error by horizon."""
    actual_days = np.asarray(df_eval["days_to_pay"], dtype=np.float64)
    pred_days_arr = np.asarray(pred_days, dtype=np.float64)

    mae = float(mean_absolute_error(actual_days, pred_days_arr))
    rmse = float(np.sqrt(mean_squared_error(actual_days, pred_days_arr)))
    median_ae = float(median_absolute_error(actual_days, pred_days_arr))

    inv_date = pd.to_datetime(df_eval["invoice_date"])
    pay_date = pd.to_datetime(df_eval["payment_date"])
    actual_diff = (pay_date - inv_date).dt.days

    pred_pay_date = inv_date + pd.to_timedelta(np.round(pred_days_arr), unit="D")
    pred_diff = (pred_pay_date - inv_date).dt.days
    amts = np.asarray(df_eval["invoice_amount"], dtype=np.float64)

    cashflow_horizons: dict[str, dict[str, Any]] = {}
    for h in horizons:
        actual_mask = actual_diff <= h
        pred_mask = pred_diff <= h
        actual_outflow = float(np.sum(amts[actual_mask]))
        predicted_outflow = float(np.sum(amts[pred_mask]))
        abs_error = abs(actual_outflow - predicted_outflow)
        mape_pct = (
            float((abs_error / actual_outflow) * 100.0)
            if actual_outflow > 0.0
            else (0.0 if predicted_outflow == 0.0 else 100.0)
        )

        cashflow_horizons[f"{h}d"] = {
            "horizon_days": int(h),
            "actual_outflow": actual_outflow,
            "predicted_outflow": predicted_outflow,
            "abs_error": abs_error,
            "mape_pct": mape_pct,
        }

    return {
        "mae": mae,
        "rmse": rmse,
        "median_ae": median_ae,
        "cashflow_horizons": cashflow_horizons,
    }
