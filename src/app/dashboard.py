"""Dual-track Metro Nashville AP late payment risk & cashflow forecast dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from src.app.loaders import (
    EDA_PLOT_SPECS,
    KAGGLE_CLS_PLOT_SPECS,
    KAGGLE_REG_PLOT_SPECS,
    build_audit_funnel_df,
    build_classification_comparison_df,
    build_regression_comparison_df,
    build_shap_top_features_df,
    build_splits_summary_df,
    compute_cashflow_horizon_comparison,
    compute_department_error_breakdown,
    compute_weekly_cashflow_series,
    filter_high_risk_invoices,
    list_existing_images,
    load_audit_summary,
    load_baseline_metrics,
    load_gbdt_metrics,
    load_test_predictions,
    resolve_data_dirs,
)


@st.cache_data(ttl="15m", max_entries=10)
def cached_audit_summary(interim_dir_str: str) -> dict[str, Any] | None:
    """Cache audit summary JSON keyed by interim directory path."""
    return load_audit_summary(Path(interim_dir_str))


@st.cache_data(ttl="15m", max_entries=10)
def cached_baseline_metrics(artifacts_dir_str: str) -> dict[str, Any] | None:
    """Cache local baseline metrics JSON keyed by artifacts directory path."""
    return load_baseline_metrics(Path(artifacts_dir_str))


@st.cache_data(ttl="15m", max_entries=10)
def cached_gbdt_metrics(artifacts_dir_str: str) -> dict[str, Any] | None:
    """Cache Kaggle GBDT metrics JSON keyed by artifacts directory path."""
    return load_gbdt_metrics(Path(artifacts_dir_str))


@st.cache_data(ttl="15m", max_entries=5)
def cached_test_predictions(artifacts_dir_str: str) -> pd.DataFrame | None:
    """Cache out-of-time test prediction parquet keyed by artifacts directory path."""
    return load_test_predictions(Path(artifacts_dir_str))


@st.cache_data(ttl="15m", max_entries=5)
def cached_department_breakdown(
    artifacts_dir_str: str,
    min_invoices: int,
) -> pd.DataFrame:
    """Cache per-department regression & risk breakdown for Tab 3."""
    preds_df = load_test_predictions(Path(artifacts_dir_str))
    if preds_df is None:
        return pd.DataFrame()
    return compute_department_error_breakdown(preds_df, min_invoices=min_invoices)


@st.cache_data(ttl="15m", max_entries=5)
def cached_horizon_comparison(artifacts_dir_str: str) -> pd.DataFrame:
    """Cache 7d/14d/30d/60d actual vs GBDT cash outflow summary."""
    preds_df = load_test_predictions(Path(artifacts_dir_str))
    if preds_df is None:
        return pd.DataFrame()
    return compute_cashflow_horizon_comparison(preds_df)


def render_image_grid(images: list[tuple[str, Path]], columns_per_row: int = 2) -> None:
    """Render verified artifact PNGs in a clean responsive two-column grid."""
    for idx in range(0, len(images), columns_per_row):
        batch = images[idx : idx + columns_per_row]
        cols = st.columns(len(batch))
        for col, (caption, img_path) in zip(cols, batch):
            with col:
                with st.container(border=True):
                    st.image(str(img_path), caption=caption, width="stretch")


def render_high_risk_triage_section(preds_df: pd.DataFrame) -> None:
    """Render interactive department/amount/threshold triage over test predictions."""
    st.subheader("High-risk invoice triage explorer", anchor=False)
    st.caption(
        "Filter 195,889 out-of-time test invoices (2025-01 to 2025-10) by predicted "
        "late probability, dollar exposure, and department."
    )

    departments = sorted(preds_df["primary_department"].dropna().unique().tolist())
    max_amt = float(preds_df["invoice_amount"].max())

    with st.container(border=True):
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.2, 1.5, 1.3])
        with ctrl_col1:
            selected_model = st.segmented_control(
                "Scoring model",
                options=["XGBoost", "LightGBM"],
                default="XGBoost",
                key="triage_model",
            )
            active_model = selected_model or "XGBoost"
            prob_col = "xgb_prob_late" if active_model == "XGBoost" else "lgb_prob_late"
        with ctrl_col2:
            selected_depts = st.multiselect(
                "Departments (leave empty for all)",
                options=departments,
                default=[],
                key="triage_departments",
            )
        with ctrl_col3:
            prob_threshold = st.slider(
                "Minimum predicted late probability",
                min_value=0.0,
                max_value=0.70,
                value=0.35,
                step=0.05,
                key="triage_prob_threshold",
            )

        slider_col1, slider_col2 = st.columns(2)
        with slider_col1:
            min_amount = st.number_input(
                "Minimum invoice amount ($)",
                min_value=0.0,
                max_value=max_amt,
                value=1000.0,
                step=500.0,
                key="triage_min_amount",
            )
        with slider_col2:
            top_n = st.slider(
                "Top rows to display",
                min_value=10,
                max_value=200,
                value=50,
                step=10,
                key="triage_top_n",
            )

    top_df, summary = filter_high_risk_invoices(
        preds_df=preds_df,
        model_col=prob_col,
        departments=selected_depts,
        min_amount=float(min_amount),
        prob_threshold=float(prob_threshold),
        top_n=int(top_n),
    )

    with st.container(horizontal=True):
        st.metric("Matched invoices", f"{summary['matched_count']:,}", border=True)
        st.metric(
            "Filtered dollar exposure",
            f"${summary['total_exposure'] / 1e6:,.2f}M",
            border=True,
        )
        st.metric(
            "Observed late rate in slice",
            f"{summary['actual_late_rate'] * 100:.1f}%",
            delta=f"{summary['actual_late_count']:,} actual late",
            delta_color="off",
            border=True,
        )
        st.metric(
            "Actual late dollar exposure",
            f"${summary['actual_late_exposure'] / 1e6:,.2f}M",
            border=True,
        )

    if top_df.empty:
        st.warning("No invoices match the current filter thresholds.", icon=":material/filter_alt_off:")
        return

    st.dataframe(
        top_df,
        column_config={
            "invoice_number": st.column_config.TextColumn("Invoice #", pinned=True),
            "supplier_number": st.column_config.TextColumn("Supplier #"),
            "primary_department": st.column_config.TextColumn("Department"),
            "invoice_date": st.column_config.DateColumn("Invoice date", format="YYYY-MM-DD"),
            "invoice_amount": st.column_config.NumberColumn("Amount ($)", format="$%,.2f"),
            "pred_prob_late": st.column_config.ProgressColumn(
                f"{active_model} P(late)",
                min_value=0.0,
                max_value=1.0,
                format="%.3f",
            ),
            "expected_late_exposure": st.column_config.NumberColumn(
                "Expected late ($)", format="$%,.2f"
            ),
            "is_late": st.column_config.CheckboxColumn("Actual late (>30d)"),
            "days_to_pay": st.column_config.NumberColumn("Actual days", format="%d d"),
        },
        hide_index=True,
        key="triage_table",
    )


def render_cashflow_interactive_section(
    preds_df: pd.DataFrame,
    artifacts_dir_str: str,
) -> None:
    """Render interactive weekly cash outflow comparison and department error table."""
    st.subheader("Interactive cash outflow & department error analysis", anchor=False)
    departments = sorted(preds_df["primary_department"].dropna().unique().tolist())

    selected_depts = st.multiselect(
        "Filter weekly cash outflow by department (empty = citywide total)",
        options=departments,
        default=[],
        key="cashflow_departments",
    )

    weekly_df = compute_weekly_cashflow_series(preds_df, departments=selected_depts or None)
    horizon_df = cached_horizon_comparison(artifacts_dir_str)

    chart_col1, chart_col2 = st.columns([1.6, 1.0])
    with chart_col1:
        with st.container(border=True):
            st.markdown("**Weekly cash outflow: actual vs predicted ($M)**")
            if weekly_df.empty:
                st.info("No weekly cashflow rows available for the selected filter.")
            else:
                line_chart = (
                    alt.Chart(weekly_df)
                    .mark_line(strokeWidth=2.2)
                    .encode(
                        x=alt.X("Payment Week:T", title="Payment week"),
                        y=alt.Y("Outflow ($M):Q", title="Weekly outflow ($M)"),
                        color=alt.Color("Series:N", title="Series"),
                        tooltip=[
                            alt.Tooltip("Payment Week:T", title="Week"),
                            alt.Tooltip("Series:N", title="Series"),
                            alt.Tooltip("Outflow ($M):Q", title="Outflow ($M)", format=",.2f"),
                        ],
                    )
                    .properties(height=340)
                )
                st.altair_chart(line_chart, width="stretch")

    with chart_col2:
        with st.container(border=True):
            st.markdown("**Cumulative cash outflow by horizon ($M)**")
            if not horizon_df.empty:
                bar_chart = (
                    alt.Chart(horizon_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("Horizon:N", sort=["7d", "14d", "30d", "60d"], title="Horizon"),
                        y=alt.Y("Outflow ($M):Q", title="Cumulative outflow ($M)"),
                        color=alt.Color("Series:N", title="Series"),
                        xOffset="Series:N",
                        tooltip=[
                            alt.Tooltip("Horizon:N"),
                            alt.Tooltip("Series:N"),
                            alt.Tooltip("Outflow ($M):Q", format=",.2f"),
                            alt.Tooltip("MAPE (%):Q", format=".2f"),
                        ],
                    )
                    .properties(height=340)
                )
                st.altair_chart(bar_chart, width="stretch")

    min_dept_invoices = st.slider(
        "Minimum department invoice count (Test split)",
        min_value=100,
        max_value=5000,
        value=500,
        step=100,
        key="dept_min_invoices",
    )
    dept_err_df = cached_department_breakdown(artifacts_dir_str, min_invoices=int(min_dept_invoices))
    with st.container(border=True):
        st.markdown("**Department error breakdown (sorted by XGBoost MAE)**")
        if dept_err_df.empty:
            st.info("No departments meet the minimum invoice threshold.")
        else:
            st.dataframe(
                dept_err_df,
                column_config={
                    "Department": st.column_config.TextColumn("Department", pinned=True),
                    "Invoices": st.column_config.NumberColumn("Invoices", format="%,d"),
                    "Total Spend ($M)": st.column_config.NumberColumn("Spend ($M)", format="$%,.2fM"),
                    "Actual Late Rate (%)": st.column_config.NumberColumn("Actual late %", format="%.2f%%"),
                    "XGB Avg Prob (%)": st.column_config.NumberColumn("XGB prob %", format="%.2f%%"),
                    "Actual Avg Days": st.column_config.NumberColumn("Actual avg days", format="%.2f"),
                    "XGB Pred Avg Days": st.column_config.NumberColumn("XGB pred days", format="%.2f"),
                    "LGB MAE (days)": st.column_config.NumberColumn("LGB MAE", format="%.2f d"),
                    "XGB MAE (days)": st.column_config.NumberColumn("XGB MAE", format="%.2f d"),
                    "XGB RMSE (days)": st.column_config.NumberColumn("XGB RMSE", format="%.2f d"),
                },
                hide_index=True,
                key="dept_error_table",
            )


st.set_page_config(
    page_title="Metro Nashville AP Risk & Cashflow",
    page_icon=":material/account_balance:",
    layout="wide",
)

artifacts_dir, interim_dir = resolve_data_dirs()
artifacts_dir_str = str(artifacts_dir)
interim_dir_str = str(interim_dir)

audit_summary = cached_audit_summary(interim_dir_str)
baseline_metrics = cached_baseline_metrics(artifacts_dir_str)
gbdt_metrics = cached_gbdt_metrics(artifacts_dir_str)
test_predictions = cached_test_predictions(artifacts_dir_str)
kaggle_ready = gbdt_metrics is not None

with st.sidebar:
    st.markdown("### :material/account_balance: Metro Nashville AP")
    st.caption("Dual-track Accounts Payable Risk Scoring & Cashflow Timing Forecast")
    if kaggle_ready:
        st.badge("Kaggle GBDT artifacts loaded", icon=":material/check_circle:", color="green")
    else:
        st.badge("Baseline fallback mode", icon=":material/info:", color="orange")
    st.markdown("**Data & split configuration**")
    st.caption(
        "• Late threshold: `days_to_pay > 30`\n"
        "• Train: `2022-07-01` .. `2024-06-30`\n"
        "• Val: `2024-07-01` .. `2024-12-31`\n"
        "• Test: `2025-01-01` .. `2025-10-31`\n"
        "• Censoring buffer: `61 days` (>= p90)"
    )

st.title("Financial Accounts Payable Risk & Cashflow Forecasting")
st.caption(
    "End-to-end Data Science portfolio dashboard built on 1.31M Metro Nashville Vendor "
    "Payment records using DuckDB grain resolution, strict point-in-time features, and "
    "calibrated LightGBM / XGBoost dual-track models."
)

if not kaggle_ready:
    st.info(
        "Kaggle GBDT artifacts (`artifacts/kaggle_results/`) were not found. Displaying "
        "local linear baselines (`artifacts/baseline_metrics.json`) and EDA summaries. "
        "Run `notebooks/02_kaggle_training.ipynb` (or `python -m src.models.train_gbdt`) "
        "and place outputs in `artifacts/kaggle_results/` to unlock GBDT, SHAP, and "
        "invoice-level triage.",
        icon=":material/info:",
    )

if gbdt_metrics is not None:
    xgb_test_cls = gbdt_metrics["xgb"]["test"]["classification"]
    xgb_test_reg = gbdt_metrics["xgb"]["test"]["regression"]
    with st.container(horizontal=True):
        st.metric(
            "Clean invoices",
            f"{audit_summary['clean_invoice_rows']:,}" if audit_summary else "811,914",
            delta="From 1,307,617 raw distributions",
            delta_color="off",
            border=True,
        )
        st.metric(
            "Test PR-AUC (XGBoost)",
            f"{xgb_test_cls['pr_auc']:.4f}",
            delta=f"Lift@5% = {xgb_test_cls['lift_at_k']['0.05']:.2f}x",
            border=True,
        )
        st.metric(
            "Test MAE (XGBoost)",
            f"{xgb_test_reg['mae']:.2f} days",
            delta=f"MedianAE = {xgb_test_reg['median_ae']:.2f}d",
            delta_color="off",
            border=True,
        )
        st.metric(
            "14-day Cashflow MAPE",
            f"{xgb_test_reg['cashflow_horizons']['14d']['mape_pct']:.2f}%",
            delta="$1,769.6M vs $1,781.1M actual",
            delta_color="off",
            border=True,
        )
elif baseline_metrics is not None and "test" in baseline_metrics:
    base_cls = baseline_metrics["test"]["classification"]
    base_reg = baseline_metrics["test"]["regression"]
    with st.container(horizontal=True):
        st.metric(
            "Clean invoices",
            f"{audit_summary['clean_invoice_rows']:,}" if audit_summary else "811,914",
            border=True,
        )
        st.metric("Test PR-AUC (Logistic)", f"{base_cls['pr_auc']:.4f}", border=True)
        st.metric("Test MAE (Ridge)", f"{base_reg['mae']:.2f} days", border=True)
        st.metric(
            "30-day Cashflow MAPE (Ridge)",
            f"{base_reg['cashflow_horizons']['30d']['mape_pct']:.2f}%",
            border=True,
        )

tab_eda, tab_risk, tab_cashflow = st.tabs(
    [
        ":material/fact_check: Data Audit & Statistical EDA",
        ":material/warning: Late Payment Risk Scoring (Track 1)",
        ":material/payments: Cashflow Timing Forecast (Track 2)",
    ]
)

with tab_eda:
    st.subheader("Data quality audit funnel", anchor=False)
    if audit_summary is not None:
        funnel_df = build_audit_funnel_df(audit_summary)
        with st.container(horizontal=True):
            st.metric("Raw ArcGIS rows", f"{audit_summary['raw_rows']:,}", border=True)
            st.metric("Missing keys dropped", f"{audit_summary['dropped_missing_keys']:,}", border=True)
            st.metric("Bad dates dropped", f"{audit_summary['dropped_bad_dates']:,}", border=True)
            st.metric("Credit memos separated", f"{audit_summary['credit_memo_rows']:,}", border=True)
            st.metric("Clean invoices", f"{audit_summary['clean_invoice_rows']:,}", border=True)

        st.dataframe(
            funnel_df,
            column_config={
                "Row Count": st.column_config.NumberColumn("Row count", format="%,d"),
                "Share of Raw (%)": st.column_config.NumberColumn("Share of raw", format="%.2f%%"),
            },
            hide_index=True,
            key="audit_funnel_table",
        )
    else:
        st.info("Audit summary JSON (`data/interim/invoices_audit_summary.json`) not found.")

    st.subheader("Out-of-time split & censoring guardrails", anchor=False)
    splits_df = build_splits_summary_df()
    st.dataframe(
        splits_df,
        column_config={
            "Invoices": st.column_config.NumberColumn("Invoices", format="%,d"),
            "Late Rate (%)": st.column_config.NumberColumn("Late rate", format="%.2f%%"),
        },
        hide_index=True,
        key="splits_summary_table",
    )

    eda_images = list_existing_images(artifacts_dir / "eda", EDA_PLOT_SPECS)
    st.subheader("Statistical EDA & hypothesis test visuals", anchor=False)
    if eda_images:
        render_image_grid(eda_images, columns_per_row=2)
    else:
        st.info("No EDA PNGs found under `artifacts/eda/`. Run `notebooks/01_eda_statistical_audit.ipynb` to generate them.")

with tab_risk:
    st.subheader("Track 1 model benchmark (Dummy vs Linear vs Calibrated GBDT)", anchor=False)
    cls_comp_df = build_classification_comparison_df(baseline_metrics, gbdt_metrics)
    st.dataframe(
        cls_comp_df,
        column_config={
            "PR-AUC": st.column_config.NumberColumn("PR-AUC", format="%.4f"),
            "ROC-AUC": st.column_config.NumberColumn("ROC-AUC", format="%.4f"),
            "Brier Score": st.column_config.NumberColumn("Brier score", format="%.4f"),
            "Lift@5%": st.column_config.NumberColumn("Lift@5%", format="%.2fx"),
            "Lift@10%": st.column_config.NumberColumn("Lift@10%", format="%.2fx"),
            "Lift@20%": st.column_config.NumberColumn("Lift@20%", format="%.2fx"),
            "Value Captured@20% (%)": st.column_config.NumberColumn("Value captured@20%", format="%.2f%%"),
        },
        hide_index=True,
        key="cls_benchmark_table",
    )

    kaggle_dir = artifacts_dir / "kaggle_results"
    cls_images = list_existing_images(kaggle_dir, KAGGLE_CLS_PLOT_SPECS)
    if cls_images:
        st.subheader("Calibration, precision-recall & SHAP explainability", anchor=False)
        render_image_grid(cls_images, columns_per_row=3)

    shap_df = build_shap_top_features_df(gbdt_metrics)
    if not shap_df.empty:
        with st.expander("SHAP global feature importance table (all 20 features)", expanded=False):
            st.dataframe(
                shap_df,
                column_config={
                    "Rank": st.column_config.NumberColumn("Rank", format="%d"),
                    "Mean |SHAP|": st.column_config.NumberColumn("Mean |SHAP|", format="%.4f"),
                },
                hide_index=True,
                key="shap_table",
            )

    if test_predictions is not None:
        render_high_risk_triage_section(test_predictions)
    elif not kaggle_ready:
        st.info("Invoice-level risk triage requires `artifacts/kaggle_results/test_predictions.parquet`.")

with tab_cashflow:
    st.subheader("Track 2 regression & cashflow horizon benchmark", anchor=False)
    reg_comp_df = build_regression_comparison_df(baseline_metrics, gbdt_metrics)
    st.dataframe(
        reg_comp_df,
        column_config={
            "MAE (days)": st.column_config.NumberColumn("MAE", format="%.2f d"),
            "RMSE (days)": st.column_config.NumberColumn("RMSE", format="%.2f d"),
            "MedianAE (days)": st.column_config.NumberColumn("MedianAE", format="%.2f d"),
            "7d MAPE (%)": st.column_config.NumberColumn("7d MAPE", format="%.2f%%"),
            "14d MAPE (%)": st.column_config.NumberColumn("14d MAPE", format="%.2f%%"),
            "30d MAPE (%)": st.column_config.NumberColumn("30d MAPE", format="%.2f%%"),
        },
        hide_index=True,
        key="reg_benchmark_table",
    )

    reg_images = list_existing_images(artifacts_dir / "kaggle_results", KAGGLE_REG_PLOT_SPECS)
    if reg_images:
        render_image_grid(reg_images, columns_per_row=1)

    if test_predictions is not None:
        render_cashflow_interactive_section(test_predictions, artifacts_dir_str)
    else:
        st.info(
            "Interactive weekly cash outflow and department error analysis require "
            "`artifacts/kaggle_results/test_predictions.parquet`."
        )
