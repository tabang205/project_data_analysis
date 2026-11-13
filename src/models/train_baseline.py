import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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


def build_baseline_preprocessor() -> ColumnTransformer:
    """Standardize numeric features and one-hot encode high-frequency categoricals."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(NUMERIC_FEATURES)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=50),
                list(CATEGORICAL_FEATURES),
            ),
        ]
    )


def train_and_eval_baselines(
    train_df: pd.DataFrame | None = None,
    val_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
    artifacts_dir: Path | None = None,
) -> dict[str, Any]:
    """Train linear baselines (LogisticRegression & Ridge) and evaluate dual tracks.

    Strictly excludes post-payment columns from X by selecting only FEATURE_COLS.
    Trains LogisticRegression on is_late and Ridge on log1p(days_to_pay).
    Evaluates on validation and test splits, saving models and baseline_metrics.json.
    """
    proc_dir = SETTINGS.processed_dir
    if train_df is None:
        train_df = pd.read_parquet(proc_dir / "train.parquet")
    if val_df is None:
        val_df = pd.read_parquet(proc_dir / "val.parquet")
    if test_df is None:
        test_df = pd.read_parquet(proc_dir / "test.parquet")

    target_artifacts_dir = artifacts_dir or SETTINGS.artifacts_dir
    target_artifacts_dir.mkdir(parents=True, exist_ok=True)

    feature_col_list = list(FEATURE_COLS)
    X_train = train_df[feature_col_list]
    y_train_cls = train_df[TARGET_CLASSIFICATION].to_numpy().astype(int)
    y_train_reg_log = np.log1p(train_df[TARGET_REGRESSION].to_numpy().astype(float))

    logger.info(
        "Training baselines on %d rows with %d features",
        len(train_df),
        len(feature_col_list),
    )

    # Track 1: Logistic Regression for is_late risk
    clf_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_baseline_preprocessor()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    clf_pipeline.fit(X_train, y_train_cls)

    # Track 2: Ridge Regression for log1p(days_to_pay)
    reg_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_baseline_preprocessor()),
            ("regressor", Ridge(alpha=10.0, random_state=42)),
        ]
    )
    reg_pipeline.fit(X_train, y_train_reg_log)

    eval_splits = {
        "val": val_df,
        "test": test_df,
    }

    metrics: dict[str, Any] = {
        "meta": {
            "n_train": len(train_df),
            "n_features": len(feature_col_list),
            "numeric_features": list(NUMERIC_FEATURES),
            "categorical_features": list(CATEGORICAL_FEATURES),
            "models": {
                "classification": "LogisticRegression(max_iter=1000, random_state=42)",
                "regression": "Ridge(alpha=10.0, random_state=42)",
            },
        }
    }

    for split_name, split_df in eval_splits.items():
        X_split = split_df[feature_col_list]
        y_split_cls = split_df[TARGET_CLASSIFICATION].to_numpy().astype(int)
        amts = split_df["invoice_amount"].to_numpy().astype(float)

        y_prob_cls = clf_pipeline.predict_proba(X_split)[:, 1]
        cls_eval = evaluate_classification(
            y_true=y_split_cls,
            y_prob=y_prob_cls,
            invoice_amounts=amts,
        )

        reg_log_preds = reg_pipeline.predict(X_split)
        pred_days = inverse_transform_days(reg_log_preds)
        reg_eval = evaluate_regression_and_cashflow(
            df_eval=split_df,
            pred_days=pred_days,
        )

        metrics[split_name] = {
            "n_samples": len(split_df),
            "classification": cls_eval,
            "regression": reg_eval,
        }

        logger.info(
            "%s evaluation: PR-AUC=%.4f, ROC-AUC=%.4f, Brier=%.4f | MAE=%.2f, RMSE=%.2f, MedianAE=%.2f",
            split_name.upper(),
            cls_eval["pr_auc"],
            cls_eval["roc_auc"],
            cls_eval["brier_score"],
            reg_eval["mae"],
            reg_eval["rmse"],
            reg_eval["median_ae"],
        )

    metrics_path = target_artifacts_dir / "baseline_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    clf_path = target_artifacts_dir / "baseline_classifier.joblib"
    reg_path = target_artifacts_dir / "baseline_regressor.joblib"
    joblib.dump(clf_pipeline, clf_path)
    joblib.dump(reg_pipeline, reg_path)

    logger.info("Saved baseline artifacts to %s", target_artifacts_dir)
    return metrics


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Train and evaluate dual-track linear baseline models."
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=SETTINGS.artifacts_dir,
        help="Directory to save baseline metrics and model joblibs",
    )
    args = parser.parse_args()
    train_and_eval_baselines(artifacts_dir=args.artifacts_dir)


if __name__ == "__main__":
    main()
