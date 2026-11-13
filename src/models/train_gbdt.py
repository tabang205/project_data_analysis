import argparse
import inspect
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve
from sklearn.preprocessing import OrdinalEncoder

try:
    from sklearn.frozen import FrozenEstimator
except ImportError:  # pragma: no cover - fallback for sklearn < 1.6 on older Kaggle images
    FrozenEstimator = None  # type: ignore[assignment,misc]

from src.config import SETTINGS
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

logger = logging.getLogger(__name__)

_PREDICTION_PRESERVED_COLS: tuple[str, ...] = (
    "invoice_number",
    "supplier_number",
    "operating_unit",
    "primary_department",
    "invoice_date",
    "payment_date",
    "invoice_amount",
    "days_to_pay",
    "is_late",
)


def _encode_split_features(
    df: pd.DataFrame,
    encoder: OrdinalEncoder,
) -> pd.DataFrame:
    """Transform numeric and categorical columns into a pure float64 feature frame."""
    num_df = df[list(NUMERIC_FEATURES)].astype(np.float64).copy()
    cat_raw = df[list(CATEGORICAL_FEATURES)].astype(object).fillna("__MISSING__")
    cat_encoded = encoder.transform(cat_raw)
    cat_df = pd.DataFrame(
        cat_encoded,
        columns=list(CATEGORICAL_FEATURES),
        index=df.index,
        dtype=np.float64,
    )
    return pd.concat([num_df, cat_df], axis=1)[list(FEATURE_COLS)]


def _fit_ordinal_encoder(train_df: pd.DataFrame) -> OrdinalEncoder:
    """Fit OrdinalEncoder strictly on training categoricals to avoid leakage."""
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
        dtype=np.float64,
    )
    cat_train = train_df[list(CATEGORICAL_FEATURES)].astype(object).fillna("__MISSING__")
    encoder.fit(cat_train)
    return encoder


def _fit_lgb_with_early_stopping(
    model: lgb.LGBMModel,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    stopping_rounds: int,
) -> None:
    """Fit LightGBM using eval_X/eval_y on >=4.7 or eval_set on older versions."""
    callbacks = [lgb.early_stopping(stopping_rounds, verbose=False)]
    if "eval_X" in inspect.signature(model.fit).parameters:
        model.fit(X_train, y_train, eval_X=X_val, eval_y=y_val, callbacks=callbacks)
    else:  # pragma: no cover
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)


def _fit_posthoc_calibrator(
    model: Any,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> CalibratedClassifierCV:
    """Calibrate pre-fitted classifier probabilities on validation data using isotonic regression."""
    if FrozenEstimator is not None:
        calibrator = CalibratedClassifierCV(
            estimator=FrozenEstimator(model),
            method="isotonic",
        )
    else:  # pragma: no cover
        calibrator = CalibratedClassifierCV(
            estimator=model,
            cv="prefit",
            method="isotonic",
        )
    calibrator.fit(X_val, y_val)
    return calibrator


def _normalize_shap_values(raw_values: Any) -> np.ndarray:
    """Extract 2D positive-class SHAP matrix across differing SHAP/LightGBM return shapes."""
    if isinstance(raw_values, list):
        arr = np.asarray(raw_values[1] if len(raw_values) > 1 else raw_values[0], dtype=np.float64)
    else:
        arr = np.asarray(raw_values, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[:, :, 1] if arr.shape[2] > 1 else arr[:, :, 0]
    return arr


def _evaluate_cls_model(
    raw_model: Any,
    calibrator: Any,
    X_split: pd.DataFrame,
    split_df: pd.DataFrame,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Evaluate raw vs calibrated classification probabilities on a single split."""
    y_true = split_df[TARGET_CLASSIFICATION].to_numpy().astype(int)
    amts = split_df["invoice_amount"].to_numpy(dtype=np.float64)

    raw_probs = np.asarray(raw_model.predict_proba(X_split)[:, 1], dtype=np.float64)
    cal_probs = np.asarray(calibrator.predict_proba(X_split)[:, 1], dtype=np.float64)

    uncal_brier = float(brier_score_loss(y_true, raw_probs))
    cls_metrics = evaluate_classification(
        y_true=y_true,
        y_prob=cal_probs,
        invoice_amounts=amts,
    )
    cls_metrics["uncalibrated_brier_score"] = uncal_brier
    cls_metrics["brier_improvement"] = float(uncal_brier - cls_metrics["brier_score"])
    return cls_metrics, raw_probs, cal_probs


def _evaluate_reg_model(
    reg_model: Any,
    X_split: pd.DataFrame,
    split_df: pd.DataFrame,
) -> tuple[dict[str, Any], np.ndarray]:
    """Invert log1p predictions and evaluate pointwise regression and cashflow horizons."""
    log_preds = reg_model.predict(X_split)
    pred_days = inverse_transform_days(log_preds)
    reg_metrics = evaluate_regression_and_cashflow(df_eval=split_df, pred_days=pred_days)
    return reg_metrics, pred_days


def _save_calibration_plot(
    y_test: np.ndarray,
    prob_curves: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: Path,
) -> None:
    """Plot reliability diagram comparing uncalibrated vs isotonic-calibrated models."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")

    colors = {"lgbm": "#1f77b4", "xgb": "#d62728"}
    for name, (raw_p, cal_p) in prob_curves.items():
        color = colors.get(name, "#2ca02c")
        frac_pos_raw, mean_pred_raw = calibration_curve(y_test, raw_p, n_bins=10, strategy="quantile")
        frac_pos_cal, mean_pred_cal = calibration_curve(y_test, cal_p, n_bins=10, strategy="quantile")
        brier_raw = brier_score_loss(y_test, raw_p)
        brier_cal = brier_score_loss(y_test, cal_p)
        ax.plot(
            mean_pred_raw,
            frac_pos_raw,
            marker="o",
            linestyle=":",
            alpha=0.6,
            color=color,
            label=f"{name.upper()} raw (Brier={brier_raw:.4f})",
        )
        ax.plot(
            mean_pred_cal,
            frac_pos_cal,
            marker="s",
            linewidth=2.0,
            color=color,
            label=f"{name.upper()} calibrated (Brier={brier_cal:.4f})",
        )

    ax.set_title("Test Calibration Curve (Raw vs Isotonic)")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Fraction of Late Invoices")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_pr_curve_plot(
    y_test: np.ndarray,
    calibrated_probs: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """Plot Precision-Recall curves on the out-of-time test set."""
    fig, ax = plt.subplots(figsize=(7, 5))
    base_rate = float(np.mean(y_test))
    ax.axhline(base_rate, linestyle="--", color="gray", label=f"Baseline late rate ({base_rate:.3f})")

    colors = {"lgbm": "#1f77b4", "xgb": "#d62728"}
    for name, probs in calibrated_probs.items():
        precision, recall, _ = precision_recall_curve(y_test, probs)
        ap = average_precision_score(y_test, probs)
        ax.plot(
            recall,
            precision,
            linewidth=2.0,
            color=colors.get(name, "#2ca02c"),
            label=f"{name.upper()} (PR-AUC={ap:.4f})",
        )

    ax.set_title("Test Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_cashflow_forecast_plot(
    lgb_cashflow: dict[str, dict[str, Any]],
    xgb_cashflow: dict[str, dict[str, Any]],
    out_path: Path,
) -> None:
    """Plot actual vs predicted cumulative cash outflow across 7d, 14d, and 30d horizons."""
    horizons = list(lgb_cashflow.keys())
    x = np.arange(len(horizons))
    width = 0.25

    actual_vals = [lgb_cashflow[h]["actual_outflow"] for h in horizons]
    lgb_vals = [lgb_cashflow[h]["predicted_outflow"] for h in horizons]
    xgb_vals = [xgb_cashflow[h]["predicted_outflow"] for h in horizons]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, actual_vals, width, label="Actual Outflow", color="#2f4b7c")
    ax.bar(x, lgb_vals, width, label="LightGBM Predicted", color="#1f77b4")
    ax.bar(x + width, xgb_vals, width, label="XGBoost Predicted", color="#d62728")

    for idx, h in enumerate(horizons):
        lgb_mape = lgb_cashflow[h]["mape_pct"]
        xgb_mape = xgb_cashflow[h]["mape_pct"]
        y_top = max(actual_vals[idx], lgb_vals[idx], xgb_vals[idx])
        ax.text(
            x[idx],
            y_top * 1.02 if y_top > 0 else 0.5,
            f"MAPE: L={lgb_mape:.1f}%, X={xgb_mape:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_title("Test Cash Outflow Forecast by Horizon")
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.set_ylabel("Cumulative Cash Outflow ($)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _compute_and_save_shap(
    lgb_clf: lgb.LGBMClassifier,
    X_test: pd.DataFrame,
    out_path: Path,
    max_samples: int = 1000,
) -> list[dict[str, Any]]:
    """Compute TreeExplainer SHAP values on a test sample and save summary beeswarm plot."""
    sample_n = min(max_samples, len(X_test))
    X_shap = X_test.sample(n=sample_n, random_state=42) if len(X_test) > sample_n else X_test

    explainer = shap.TreeExplainer(lgb_clf)
    explanation = explainer(X_shap)
    shap_matrix = _normalize_shap_values(explanation.values)

    mean_abs = np.mean(np.abs(shap_matrix), axis=0)
    ranking = sorted(
        (
            {"feature": col, "mean_abs_shap": float(val)}
            for col, val in zip(X_shap.columns, mean_abs, strict=True)
        ),
        key=lambda item: item["mean_abs_shap"],
        reverse=True,
    )

    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_matrix, X_shap, show=False, max_display=20)
    plt.title("LightGBM Late-Payment Risk SHAP Summary")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    return ranking


def train_dual_gbdt_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: Path,
    n_estimators: int = 500,
) -> dict[str, Any]:
    """Train calibrated LightGBM & XGBoost dual-track models and export artifacts and SHAP plots.

    Track 1 predicts late-payment probability (`is_late`) with post-hoc isotonic calibration on `val_df`.
    Track 2 predicts `log1p(days_to_pay)` and evaluates inverted day error and horizon cash outflow.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stopping_rounds = min(50, max(5, n_estimators // 5))
    min_child_samples = min(20, max(2, len(train_df) // 10))

    encoder = _fit_ordinal_encoder(train_df)
    X_train = _encode_split_features(train_df, encoder)
    X_val = _encode_split_features(val_df, encoder)
    X_test = _encode_split_features(test_df, encoder)

    y_train_cls = train_df[TARGET_CLASSIFICATION].to_numpy().astype(int)
    y_val_cls = val_df[TARGET_CLASSIFICATION].to_numpy().astype(int)
    y_test_cls = test_df[TARGET_CLASSIFICATION].to_numpy().astype(int)

    y_train_reg = np.log1p(np.clip(train_df[TARGET_REGRESSION].to_numpy(dtype=np.float64), 0.0, None))
    y_val_reg = np.log1p(np.clip(val_df[TARGET_REGRESSION].to_numpy(dtype=np.float64), 0.0, None))

    logger.info(
        "Training dual-track GBDT models (n_estimators=%d, train=%d, val=%d, test=%d)",
        n_estimators,
        len(train_df),
        len(val_df),
        len(test_df),
    )

    lgb_clf = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=min_child_samples,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    _fit_lgb_with_early_stopping(
        lgb_clf, X_train, y_train_cls, X_val, y_val_cls, stopping_rounds
    )
    lgb_cal = _fit_posthoc_calibrator(lgb_clf, X_val, y_val_cls)

    xgb_clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        early_stopping_rounds=stopping_rounds,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    xgb_clf.fit(X_train, y_train_cls, eval_set=[(X_val, y_val_cls)], verbose=False)
    xgb_cal = _fit_posthoc_calibrator(xgb_clf, X_val, y_val_cls)

    lgb_reg = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=min_child_samples,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    _fit_lgb_with_early_stopping(
        lgb_reg, X_train, y_train_reg, X_val, y_val_reg, stopping_rounds
    )

    xgb_reg = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="rmse",
        early_stopping_rounds=stopping_rounds,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    xgb_reg.fit(X_train, y_train_reg, eval_set=[(X_val, y_val_reg)], verbose=False)

    models_bundle = {
        "lgbm": {"clf": lgb_clf, "cal": lgb_cal, "reg": lgb_reg},
        "xgb": {"clf": xgb_clf, "cal": xgb_cal, "reg": xgb_reg},
    }
    splits = {"val": (val_df, X_val), "test": (test_df, X_test)}

    metrics: dict[str, Any] = {
        "meta": {
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
            "n_estimators": int(n_estimators),
            "features": list(FEATURE_COLS),
            "calibration_method": "isotonic",
        }
    }

    test_raw_probs: dict[str, np.ndarray] = {}
    test_cal_probs: dict[str, np.ndarray] = {}
    test_pred_days: dict[str, np.ndarray] = {}

    for model_name, bundle in models_bundle.items():
        metrics[model_name] = {}
        for split_name, (split_df, X_split) in splits.items():
            cls_eval, raw_p, cal_p = _evaluate_cls_model(
                bundle["clf"], bundle["cal"], X_split, split_df
            )
            reg_eval, pred_d = _evaluate_reg_model(bundle["reg"], X_split, split_df)
            metrics[model_name][split_name] = {
                "n_samples": len(split_df),
                "classification": cls_eval,
                "regression": reg_eval,
            }
            if split_name == "test":
                test_raw_probs[model_name] = raw_p
                test_cal_probs[model_name] = cal_p
                test_pred_days[model_name] = pred_d

    shap_ranking = _compute_and_save_shap(
        lgb_clf=lgb_clf,
        X_test=X_test,
        out_path=out_dir / "shap_summary.png",
    )
    metrics["shap_top_features"] = shap_ranking

    _save_calibration_plot(
        y_test=y_test_cls,
        prob_curves={
            name: (test_raw_probs[name], test_cal_probs[name])
            for name in ("lgbm", "xgb")
        },
        out_path=out_dir / "calibration_curve.png",
    )
    _save_pr_curve_plot(
        y_test=y_test_cls,
        calibrated_probs=test_cal_probs,
        out_path=out_dir / "pr_curve.png",
    )
    _save_cashflow_forecast_plot(
        lgb_cashflow=metrics["lgbm"]["test"]["regression"]["cashflow_horizons"],
        xgb_cashflow=metrics["xgb"]["test"]["regression"]["cashflow_horizons"],
        out_path=out_dir / "cashflow_forecast.png",
    )

    preserved_cols = [c for c in _PREDICTION_PRESERVED_COLS if c in test_df.columns]
    preds_df = test_df[preserved_cols].copy()
    preds_df["lgb_prob_late"] = test_cal_probs["lgbm"]
    preds_df["xgb_prob_late"] = test_cal_probs["xgb"]
    preds_df["lgb_pred_days"] = test_pred_days["lgbm"]
    preds_df["xgb_pred_days"] = test_pred_days["xgb"]
    preds_df.to_parquet(out_dir / "test_predictions.parquet", index=False, compression="snappy")

    artifacts = {
        "ordinal_encoder.joblib": encoder,
        "lgb_classifier.joblib": lgb_clf,
        "lgb_calibrator.joblib": lgb_cal,
        "xgb_classifier.joblib": xgb_clf,
        "xgb_calibrator.joblib": xgb_cal,
        "lgb_regressor.joblib": lgb_reg,
        "xgb_regressor.joblib": xgb_reg,
    }
    for filename, obj in artifacts.items():
        joblib.dump(obj, out_dir / filename)

    with open(out_dir / "gbdt_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(
        "GBDT training complete -> %s | Test PR-AUC: LGB=%.4f, XGB=%.4f | Test MAE: LGB=%.2f, XGB=%.2f",
        out_dir,
        metrics["lgbm"]["test"]["classification"]["pr_auc"],
        metrics["xgb"]["test"]["classification"]["pr_auc"],
        metrics["lgbm"]["test"]["regression"]["mae"],
        metrics["xgb"]["test"]["regression"]["mae"],
    )
    return metrics


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Train calibrated dual-track LightGBM & XGBoost models."
    )
    parser.add_argument("--processed-dir", type=Path, default=SETTINGS.processed_dir)
    parser.add_argument("--out-dir", type=Path, default=SETTINGS.artifacts_dir)
    parser.add_argument("--n-estimators", type=int, default=500)
    args = parser.parse_args()

    train_df = pd.read_parquet(args.processed_dir / "train.parquet")
    val_df = pd.read_parquet(args.processed_dir / "val.parquet")
    test_df = pd.read_parquet(args.processed_dir / "test.parquet")
    train_dual_gbdt_models(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        out_dir=args.out_dir,
        n_estimators=args.n_estimators,
    )


if __name__ == "__main__":
    main()
