"""Pydantic v2 response schemas for the AP Risk & Cashflow Workbench API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class StrictSchema(BaseModel):
    """Base schema enforcing Pydantic v2 strict attribute and alias config."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class HealthResponse(StrictSchema):
    status: str
    kaggle_ready: bool
    clean_invoices: int
    test_invoices: int
    active_mode: str


class AuditSummarySchema(StrictSchema):
    raw_rows: int
    grouped_invoice_rows: int
    dropped_missing_keys: int
    dropped_bad_dates: int
    credit_memo_rows: int
    clean_invoice_rows: int


class AuditFunnelItemSchema(StrictSchema):
    stage: str
    row_count: int
    share_of_raw_pct: float
    disposition: str


class SplitSummaryItemSchema(StrictSchema):
    split: str
    window: str
    invoices: int
    late_rate_pct: float
    cold_start_pct: str
    role_and_guardrail: str


class ClassificationBenchmarkItemSchema(StrictSchema):
    model: str
    split: str
    pr_auc: float
    roc_auc: float
    brier_score: float
    lift_5: float
    lift_10: float
    lift_20: float
    value_captured_20_pct: float


class RegressionBenchmarkItemSchema(StrictSchema):
    model: str
    split: str
    mae: float
    rmse: float
    median_ae: float
    mape_7d: float
    mape_14d: float
    mape_30d: float


class ShapFeatureItemSchema(StrictSchema):
    rank: int
    feature: str
    mean_abs_shap: float
    share_of_total_pct: float


class PlotAssetSchema(StrictSchema):
    filename: str
    caption: str
    url: str


class OverviewResponse(StrictSchema):
    kaggle_ready: bool
    audit_summary: AuditSummarySchema | None
    audit_funnel: list[AuditFunnelItemSchema]
    splits: list[SplitSummaryItemSchema]
    classification_benchmark: list[ClassificationBenchmarkItemSchema]
    regression_benchmark: list[RegressionBenchmarkItemSchema]
    shap_top_features: list[ShapFeatureItemSchema]
    eda_plots: list[PlotAssetSchema]
    cls_plots: list[PlotAssetSchema]
    reg_plots: list[PlotAssetSchema]


class DepartmentItemSchema(StrictSchema):
    department: str
    invoices: int
    total_spend_m: float
    actual_late_rate_pct: float
    xgb_avg_prob_pct: float
    actual_avg_days: float
    xgb_pred_avg_days: float
    lgb_mae_days: float
    xgb_mae_days: float
    xgb_rmse_days: float


class DepartmentListResponse(StrictSchema):
    kaggle_ready: bool
    total_departments: int
    departments: list[DepartmentItemSchema]


class InvoiceTriageItemSchema(StrictSchema):
    invoice_number: str
    supplier_number: str
    primary_department: str
    invoice_date: str
    invoice_amount: float
    pred_prob_late: float
    expected_late_exposure: float
    is_late: bool
    days_to_pay: float
    xgb_pred_days: float | None = None
    lgb_pred_days: float | None = None


class InvoiceTriageSummarySchema(StrictSchema):
    matched_count: int
    total_exposure: float
    actual_late_count: int
    actual_late_rate: float
    actual_late_exposure: float


class InvoiceTriageResponse(StrictSchema):
    kaggle_ready: bool
    model: str
    department: str
    summary: InvoiceTriageSummarySchema
    items: list[InvoiceTriageItemSchema]


class WeeklyCashflowPointSchema(StrictSchema):
    payment_week: str
    series: str
    outflow_m: float


class CashflowHorizonPointSchema(StrictSchema):
    horizon: str
    horizon_days: int
    series: str
    outflow_m: float
    mape_pct: float


class CashflowSeriesResponse(StrictSchema):
    kaggle_ready: bool
    department: str
    weekly_series: list[WeeklyCashflowPointSchema]
    horizons: list[CashflowHorizonPointSchema]


class LearningDocMetaSchema(StrictSchema):
    slug: str
    title: str
    filename: str


class LearningDocsListResponse(StrictSchema):
    docs: list[LearningDocMetaSchema]


class LearningDocDetailResponse(StrictSchema):
    slug: str
    title: str
    filename: str
    content: str
