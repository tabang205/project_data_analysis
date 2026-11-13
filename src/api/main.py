"""FastAPI application serving the 5-zone AP Risk & Cashflow Workbench."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.api.deps import ArtifactBundle, get_artifact_bundle
from src.api.schemas import (
    AuditFunnelItemSchema,
    AuditSummarySchema,
    CashflowHorizonPointSchema,
    CashflowSeriesResponse,
    ClassificationBenchmarkItemSchema,
    DepartmentItemSchema,
    DepartmentListResponse,
    HealthResponse,
    InvoiceTriageItemSchema,
    InvoiceTriageResponse,
    InvoiceTriageSummarySchema,
    LearningDocDetailResponse,
    LearningDocMetaSchema,
    LearningDocsListResponse,
    OverviewResponse,
    PlotAssetSchema,
    RegressionBenchmarkItemSchema,
    ShapFeatureItemSchema,
    SplitSummaryItemSchema,
    WeeklyCashflowPointSchema,
)
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
)
from src.config import SETTINGS

app = FastAPI(
    title="Metro Nashville AP Risk & Cashflow Workbench API",
    version="1.0.0",
)

_STATIC_DIR = SETTINGS.base_dir / "src" / "web" / "static"
_STATIC_DIR.mkdir(parents=True, exist_ok=True)
if SETTINGS.artifacts_dir.is_dir():
    app.mount(
        "/artifacts",
        StaticFiles(directory=str(SETTINGS.artifacts_dir)),
        name="artifacts",
    )
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

BundleDep = Annotated[ArtifactBundle, Depends(get_artifact_bundle)]


def _normalize_audit_dict(raw: dict[str, Any] | None) -> dict[str, int] | None:
    if raw is None:
        return None
    grouped = int(raw.get("grouped_rows", raw.get("grouped_invoice_rows", 0)))
    return {
        "raw_rows": int(raw.get("raw_rows", 0)),
        "grouped_rows": grouped,
        "grouped_invoice_rows": grouped,
        "dropped_missing_keys": int(raw.get("dropped_missing_keys", 0)),
        "dropped_bad_dates": int(raw.get("dropped_bad_dates", 0)),
        "credit_memo_rows": int(raw.get("credit_memo_rows", 0)),
        "clean_invoice_rows": int(raw.get("clean_invoice_rows", 0)),
    }


def _map_plots(
    directory: Path,
    specs: tuple[tuple[str, str], ...],
    url_prefix: str,
) -> list[PlotAssetSchema]:
    existing = list_existing_images(directory, specs)
    return [
        PlotAssetSchema(
            filename=path.name,
            caption=caption,
            url=f"{url_prefix}/{path.name}",
        )
        for caption, path in existing
    ]


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_workbench_index() -> HTMLResponse:
    """Serve the 5-zone single-page workbench application."""
    index_path = _STATIC_DIR / "index.html"
    if not index_path.is_file():
        return HTMLResponse("<html><body>Workbench static index not built</body></html>")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/api/v1/health", response_model=HealthResponse)
def get_health(bundle: BundleDep) -> HealthResponse:
    """Return service health, artifact availability, and row counts."""
    norm_audit = _normalize_audit_dict(bundle.audit_summary)
    clean_rows = norm_audit["clean_invoice_rows"] if norm_audit else 811914
    test_rows = len(bundle.test_predictions) if bundle.test_predictions is not None else 195889
    return HealthResponse(
        status="ok",
        kaggle_ready=bundle.kaggle_ready,
        clean_invoices=clean_rows,
        test_invoices=test_rows,
        active_mode="KAGGLE_GBDT_ACTIVE" if bundle.kaggle_ready else "BASELINE_FALLBACK",
    )


@app.get("/api/v1/overview", response_model=OverviewResponse)
def get_overview(bundle: BundleDep) -> OverviewResponse:
    """Return full workbench overview tables, SHAP rankings, and plot asset URLs."""
    norm_audit = _normalize_audit_dict(bundle.audit_summary)
    funnel_items: list[AuditFunnelItemSchema] = []
    if norm_audit is not None:
        funnel_df = build_audit_funnel_df(norm_audit)
        funnel_items = [
            AuditFunnelItemSchema(
                stage=str(row["Stage"]),
                row_count=int(row["Row Count"]),
                share_of_raw_pct=float(row["Share of Raw (%)"]),
                disposition=str(row["Notes"]),
            )
            for _, row in funnel_df.iterrows()
        ]

    splits_df = build_splits_summary_df()
    split_items = [
        SplitSummaryItemSchema(
            split=str(row["Segment"]),
            window=str(row["Date Window"]),
            invoices=int(row["Invoices"]),
            late_rate_pct=float(row["Late Rate (%)"]),
            cold_start_pct="N/A",
            role_and_guardrail=str(row["Purpose"]),
        )
        for _, row in splits_df.iterrows()
    ]

    cls_df = build_classification_comparison_df(bundle.baseline_metrics, bundle.gbdt_metrics)
    cls_items = [
        ClassificationBenchmarkItemSchema(
            model=str(row["Model"]),
            split=str(row["Split"]),
            pr_auc=float(row["PR-AUC"]),
            roc_auc=float(row["ROC-AUC"]),
            brier_score=float(row["Brier Score"]),
            lift_5=float(row["Lift@5%"]),
            lift_10=float(row["Lift@10%"]),
            lift_20=float(row["Lift@20%"]),
            value_captured_20_pct=float(row["Value Captured@20% (%)"]),
        )
        for _, row in cls_df.iterrows()
    ]

    reg_df = build_regression_comparison_df(bundle.baseline_metrics, bundle.gbdt_metrics)
    reg_items = [
        RegressionBenchmarkItemSchema(
            model=str(row["Model"]),
            split=str(row["Split"]),
            mae=float(row["MAE (days)"]),
            rmse=float(row["RMSE (days)"]),
            median_ae=float(row["MedianAE (days)"]),
            mape_7d=float(row["7d MAPE (%)"]),
            mape_14d=float(row["14d MAPE (%)"]),
            mape_30d=float(row["30d MAPE (%)"]),
        )
        for _, row in reg_df.iterrows()
    ]

    shap_df = build_shap_top_features_df(bundle.gbdt_metrics)
    total_shap = float(shap_df["Mean |SHAP|"].sum()) if not shap_df.empty else 0.0
    shap_items = [
        ShapFeatureItemSchema(
            rank=int(row["Rank"]),
            feature=str(row["Feature"]),
            mean_abs_shap=float(row["Mean |SHAP|"]),
            share_of_total_pct=round((float(row["Mean |SHAP|"]) / total_shap) * 100.0, 2)
            if total_shap > 0
            else 0.0,
        )
        for _, row in shap_df.iterrows()
    ]

    kaggle_dir = bundle.artifacts_dir / "kaggle_results"
    return OverviewResponse(
        kaggle_ready=bundle.kaggle_ready,
        audit_summary=AuditSummarySchema(**norm_audit) if norm_audit else None,
        audit_funnel=funnel_items,
        splits=split_items,
        classification_benchmark=cls_items,
        regression_benchmark=reg_items,
        shap_top_features=shap_items,
        eda_plots=_map_plots(bundle.artifacts_dir / "eda", EDA_PLOT_SPECS, "/artifacts/eda"),
        cls_plots=_map_plots(kaggle_dir, KAGGLE_CLS_PLOT_SPECS, "/artifacts/kaggle_results"),
        reg_plots=_map_plots(kaggle_dir, KAGGLE_REG_PLOT_SPECS, "/artifacts/kaggle_results"),
    )


@app.get("/api/v1/departments", response_model=DepartmentListResponse)
def get_departments(
    bundle: BundleDep,
    min_invoices: Annotated[int, Query(ge=1, le=10000)] = 200,
) -> DepartmentListResponse:
    """Return per-department risk and regression breakdown for sidebar navigation."""
    if bundle.test_predictions is None:
        return DepartmentListResponse(kaggle_ready=False, total_departments=0, departments=[])

    dept_df = compute_department_error_breakdown(
        bundle.test_predictions,
        min_invoices=min_invoices,
    )
    items = [
        DepartmentItemSchema(
            department=str(row["Department"]),
            invoices=int(row["Invoices"]),
            total_spend_m=round(float(row["Total Spend ($M)"]), 2),
            actual_late_rate_pct=round(float(row["Actual Late Rate (%)"]), 2),
            xgb_avg_prob_pct=round(float(row["XGB Avg Prob (%)"]), 2),
            actual_avg_days=round(float(row["Actual Avg Days"]), 2),
            xgb_pred_avg_days=round(float(row["XGB Pred Avg Days"]), 2),
            lgb_mae_days=round(float(row["LGB MAE (days)"]), 2),
            xgb_mae_days=round(float(row["XGB MAE (days)"]), 2),
            xgb_rmse_days=round(float(row["XGB RMSE (days)"]), 2),
        )
        for _, row in dept_df.iterrows()
    ]
    return DepartmentListResponse(
        kaggle_ready=True,
        total_departments=len(items),
        departments=items,
    )


@app.get("/api/v1/invoices/triage", response_model=InvoiceTriageResponse)
def get_invoice_triage(
    bundle: BundleDep,
    department: Annotated[str | None, Query()] = None,
    min_amount: Annotated[float, Query(ge=0.0)] = 1000.0,
    prob_threshold: Annotated[float, Query(ge=0.0, le=1.0)] = 0.45,
    model: Annotated[Literal["xgb", "lgb"], Query()] = "xgb",
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> InvoiceTriageResponse:
    """Filter high-risk test invoices by department, dollar threshold, and probability."""
    dept_label = department if department else "All Departments"
    if bundle.test_predictions is None:
        return InvoiceTriageResponse(
            kaggle_ready=False,
            model=model,
            department=dept_label,
            summary=InvoiceTriageSummarySchema(
                matched_count=0,
                total_exposure=0.0,
                actual_late_count=0,
                actual_late_rate=0.0,
                actual_late_exposure=0.0,
            ),
            items=[],
        )

    model_col = "xgb_prob_late" if model == "xgb" else "lgb_prob_late"
    depts = [department] if department and department != "All Departments" else []
    display_df, summary_dict = filter_high_risk_invoices(
        bundle.test_predictions,
        model_col=model_col,
        departments=depts,
        min_amount=min_amount,
        prob_threshold=prob_threshold,
        top_n=limit,
    )
    pred_days_lookup: dict[str, tuple[float, float]] = {}
    if not display_df.empty and "xgb_pred_days" in bundle.test_predictions.columns:
        matched_rows = bundle.test_predictions[
            bundle.test_predictions["invoice_number"].isin(display_df["invoice_number"])
        ]
        for _, mrow in matched_rows.iterrows():
            pred_days_lookup[str(mrow["invoice_number"])] = (
                round(float(mrow["xgb_pred_days"]), 1),
                round(float(mrow["lgb_pred_days"]), 1),
            )

    items = [
        InvoiceTriageItemSchema(
            invoice_number=str(row["invoice_number"]),
            supplier_number=str(row["supplier_number"]),
            primary_department=str(row["primary_department"]),
            invoice_date=str(row["invoice_date"])[:10],
            invoice_amount=round(float(row["invoice_amount"]), 2),
            pred_prob_late=round(float(row["pred_prob_late"]), 4),
            expected_late_exposure=round(float(row["expected_late_exposure"]), 2),
            is_late=bool(row["is_late"]),
            days_to_pay=round(float(row["days_to_pay"]), 1),
            xgb_pred_days=pred_days_lookup.get(str(row["invoice_number"]), (None, None))[0],
            lgb_pred_days=pred_days_lookup.get(str(row["invoice_number"]), (None, None))[1],
        )
        for _, row in display_df.iterrows()
    ]
    return InvoiceTriageResponse(
        kaggle_ready=True,
        model=model,
        department=dept_label,
        summary=InvoiceTriageSummarySchema(
            matched_count=int(summary_dict["matched_count"]),
            total_exposure=round(float(summary_dict["total_exposure"]), 2),
            actual_late_count=int(summary_dict["actual_late_count"]),
            actual_late_rate=round(float(summary_dict["actual_late_rate"]), 4),
            actual_late_exposure=round(float(summary_dict["actual_late_exposure"]), 2),
        ),
        items=items,
    )


@app.get("/api/v1/cashflow/series", response_model=CashflowSeriesResponse)
def get_cashflow_series(
    bundle: BundleDep,
    department: Annotated[str | None, Query()] = None,
) -> CashflowSeriesResponse:
    """Return weekly actual vs predicted cash outflow ($M) and horizon summary."""
    dept_label = department if department else "All Departments"
    if bundle.test_predictions is None:
        return CashflowSeriesResponse(
            kaggle_ready=False,
            department=dept_label,
            weekly_series=[],
            horizons=[],
        )

    depts = [department] if department and department != "All Departments" else None
    weekly_df = compute_weekly_cashflow_series(bundle.test_predictions, departments=depts)
    sub_df = (
        bundle.test_predictions[bundle.test_predictions["primary_department"].isin(depts)]
        if depts
        else bundle.test_predictions
    )
    horizon_df = (
        compute_cashflow_horizon_comparison(sub_df)
        if not sub_df.empty
        else compute_cashflow_horizon_comparison(bundle.test_predictions)
    )

    weekly_items = [
        WeeklyCashflowPointSchema(
            payment_week=str(row["Payment Week"])[:10],
            series=str(row["Series"]),
            outflow_m=round(float(row["Outflow ($M)"]), 2),
        )
        for _, row in weekly_df.iterrows()
    ]
    horizon_items = [
        CashflowHorizonPointSchema(
            horizon=str(row["Horizon"]),
            horizon_days=int(row["Horizon Days"]),
            series=str(row["Series"]),
            outflow_m=float(row["Outflow ($M)"]),
            mape_pct=float(row["MAPE (%)"]),
        )
        for _, row in horizon_df.iterrows()
    ]
    return CashflowSeriesResponse(
        kaggle_ready=True,
        department=dept_label,
        weekly_series=weekly_items,
        horizons=horizon_items,
    )


def _discover_learning_docs(docs_dir: Path) -> dict[str, LearningDocMetaSchema]:
    if not docs_dir.is_dir():
        return {}
    mapping: dict[str, LearningDocMetaSchema] = {}
    for md_path in sorted(docs_dir.glob("*.md")):
        slug = md_path.stem
        first_line = md_path.read_text(encoding="utf-8").splitlines()[0] if md_path.stat().st_size > 0 else slug
        title = first_line.lstrip("# ").strip() or slug
        mapping[slug] = LearningDocMetaSchema(
            slug=slug,
            title=title,
            filename=md_path.name,
        )
    return mapping


@app.get("/api/v1/docs", response_model=LearningDocsListResponse)
def list_learning_docs(bundle: BundleDep) -> LearningDocsListResponse:
    """List Vietnamese Data Science learning documents available in docs/learning/."""
    docs_map = _discover_learning_docs(bundle.docs_dir)
    return LearningDocsListResponse(docs=list(docs_map.values()))


@app.get("/api/v1/docs/{slug}", response_model=LearningDocDetailResponse)
def get_learning_doc(slug: str, bundle: BundleDep) -> LearningDocDetailResponse:
    """Return full markdown content for an allowlisted learning document slug."""
    docs_map = _discover_learning_docs(bundle.docs_dir)
    if slug not in docs_map:
        raise HTTPException(status_code=404, detail=f"Learning doc '{slug}' not found.")
    meta = docs_map[slug]
    doc_path = bundle.docs_dir / meta.filename
    return LearningDocDetailResponse(
        slug=meta.slug,
        title=meta.title,
        filename=meta.filename,
        content=doc_path.read_text(encoding="utf-8"),
    )
