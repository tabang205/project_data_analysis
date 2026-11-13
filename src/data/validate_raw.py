import logging
import pandas as pd
import pandera.pandas as pa
import pyarrow as pa_arrow

logger = logging.getLogger(__name__)

RAW_AP_SCHEMA = pa.DataFrameSchema(
    columns={
        "FID": pa.Column(int, nullable=False, coerce=True),
        "Operating_Unit": pa.Column(pa.String, nullable=True, coerce=True),
        "Supplier_Number": pa.Column(pa.String, nullable=True, coerce=True),
        "Invoice_Number": pa.Column(pa.String, nullable=True, coerce=True),
        "Distribution_Amount": pa.Column(float, nullable=True, coerce=True),
        "Invoice_Date": pa.Column(pa.DateTime, nullable=True, required=True),
        "Payment_Date": pa.Column(pa.DateTime, nullable=True, required=True),
        "Payment_Number": pa.Column(float, nullable=True, coerce=True),
        "Check_Amount": pa.Column(float, nullable=True, coerce=True),
        "Department_Name": pa.Column(pa.String, nullable=True, coerce=True),
        "Department_Number": pa.Column(float, nullable=True, coerce=True),
    },
    coerce=True,
    strict=False,
)

ARROW_RAW_SCHEMA = pa_arrow.schema(
    [
        ("FID", pa_arrow.int64()),
        ("Operating_Unit", pa_arrow.string()),
        ("Supplier_Number", pa_arrow.string()),
        ("Invoice_Number", pa_arrow.string()),
        ("Distribution_Amount", pa_arrow.float64()),
        ("Invoice_Date", pa_arrow.timestamp("ns")),
        ("Payment_Date", pa_arrow.timestamp("ns")),
        ("Payment_Number", pa_arrow.float64()),
        ("Check_Amount", pa_arrow.float64()),
        ("Department_Name", pa_arrow.string()),
        ("Department_Number", pa_arrow.float64()),
    ]
)


def _coerce_to_naive_datetime64_ns(series: pd.Series) -> pd.Series:
    """Coerce numeric epoch-ms or datetime strings to UTC-naive datetime64[ns]."""
    if pd.api.types.is_datetime64_any_dtype(series):
        dt = series
        if getattr(dt.dt, "tz", None) is not None:
            dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
    elif pd.api.types.is_numeric_dtype(series):
        dt = pd.to_datetime(series, unit="ms", utc=True, errors="coerce").dt.tz_localize(None)
    else:
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() > 0 and numeric.notna().sum() == series.notna().sum():
            dt = pd.to_datetime(numeric, unit="ms", utc=True, errors="coerce").dt.tz_localize(None)
        else:
            dt = pd.to_datetime(series, utc=True, errors="coerce").dt.tz_localize(None)

    min_ns = pd.Timestamp.min
    max_ns = pd.Timestamp.max
    mask = (dt < min_ns) | (dt > max_ns)
    return dt.mask(mask).astype("datetime64[ns]")


def validate_raw_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Validate raw accounts payable records against expected schema."""
    df_copy = df.copy()
    if "Invoice_Date" in df_copy.columns:
        df_copy["Invoice_Date"] = _coerce_to_naive_datetime64_ns(df_copy["Invoice_Date"])
    if "Payment_Date" in df_copy.columns:
        df_copy["Payment_Date"] = _coerce_to_naive_datetime64_ns(df_copy["Payment_Date"])
    return RAW_AP_SCHEMA.validate(df_copy)
