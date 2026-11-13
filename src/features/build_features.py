import argparse
import heapq
import logging
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.config import SETTINGS

logger = logging.getLogger(__name__)

NUMERIC_FEATURES: tuple[str, ...] = (
    "log_invoice_amount",
    "line_item_count",
    "dept_count",
    "amount_vs_vendor_median_ratio",
    "vendor_prior_paid_count",
    "vendor_prior_late_rate_smoothed",
    "vendor_prior_avg_days_90d",
    "vendor_prior_avg_days_365d",
    "vendor_prior_std_days_365d",
    "vendor_has_prior_credit",
    "dept_inv_count_7d",
    "dept_inv_count_30d",
    "dept_prior_late_rate",
    "same_day_vendor_inv_count",
    "inv_month",
    "inv_quarter",
    "inv_dayofweek",
    "days_to_fiscal_year_end",
)

CATEGORICAL_FEATURES: tuple[str, ...] = (
    "operating_unit",
    "primary_department",
)

FEATURE_COLS: tuple[str, ...] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET_CLASSIFICATION: str = "is_late"
TARGET_REGRESSION: str = "days_to_pay"

PRESERVED_COLUMNS: tuple[str, ...] = (
    "operating_unit",
    "supplier_number",
    "invoice_number",
    "payment_number",
    "invoice_date",
    "payment_date",
    "primary_department",
    "invoice_amount",
    "days_to_pay",
    "is_late",
)

OUTPUT_COLUMNS: tuple[str, ...] = PRESERVED_COLUMNS + tuple(
    col for col in NUMERIC_FEATURES if col not in PRESERVED_COLUMNS
)

# Fixed Bayesian Beta prior matching ~25% Net-30 baseline avoids cross-split leakage.
_PRIOR_ALPHA: float = 2.5
_PRIOR_BETA: float = 7.5
_COLD_START_DAYS: float = 30.0
_DEFAULT_TRAIN_START: str = "2022-07-01"


def _prepare_credit_first_dates(credit_df: pd.DataFrame | None) -> pd.DataFrame:
    """Reduce credit memos to earliest effective date per supplier."""
    empty = pd.DataFrame(
        {
            "supplier_number": pd.Series(dtype="object"),
            "first_credit_day": pd.Series(dtype="datetime64[ns]"),
        }
    )
    if credit_df is None or credit_df.empty:
        return empty

    inv_dt = pd.to_datetime(credit_df["invoice_date"], errors="coerce")
    if "payment_date" in credit_df.columns:
        pay_dt = pd.to_datetime(credit_df["payment_date"], errors="coerce")
        effective_dt = pay_dt.fillna(inv_dt)
    else:
        effective_dt = inv_dt

    valid = pd.DataFrame(
        {
            "supplier_number": credit_df["supplier_number"].astype(str),
            "first_credit_day": effective_dt.dt.normalize(),
        }
    ).dropna(subset=["supplier_number", "first_credit_day"])

    if valid.empty:
        return empty
    return valid.groupby("supplier_number", as_index=False)["first_credit_day"].min()


def _compute_vendor_prior_medians(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Compute exact expanding median of prior paid invoice_amount per (supplier, inv_day)."""
    pay_events = con.sql(
        """
        SELECT supplier_number, pay_day, CAST(invoice_amount AS DOUBLE) AS invoice_amount
        FROM base_invoices
        ORDER BY supplier_number, pay_day
        """
    ).fetchnumpy()
    query_days = con.sql(
        """
        SELECT DISTINCT supplier_number, inv_day
        FROM base_invoices
        ORDER BY supplier_number, inv_day
        """
    ).fetchnumpy()

    sups = pay_events["supplier_number"].tolist()
    pdays = pay_events["pay_day"].tolist()
    amts = pay_events["invoice_amount"].tolist()

    qsups = query_days["supplier_number"].tolist()
    qdays = query_days["inv_day"].tolist()

    n_p = len(sups)
    n_q = len(qsups)
    medians = np.full(n_q, np.nan, dtype=np.float64)

    p_i = 0
    q_i = 0
    while q_i < n_q:
        sup = qsups[q_i]
        # Sequences of distinct suppliers match because both queries sort on DuckDB's base_invoices.
        while p_i < n_p and sups[p_i] != sup:
            p_i += 1
        lo: list[float] = []
        hi: list[float] = []
        while q_i < n_q and qsups[q_i] == sup:
            qd = qdays[q_i]
            while p_i < n_p and sups[p_i] == sup and pdays[p_i] < qd:
                val = float(amts[p_i])
                if not lo or val <= -lo[0]:
                    heapq.heappush(lo, -val)
                else:
                    heapq.heappush(hi, val)
                if len(lo) > len(hi) + 1:
                    heapq.heappush(hi, -heapq.heappop(lo))
                elif len(hi) > len(lo):
                    heapq.heappush(lo, -heapq.heappop(hi))
                p_i += 1
            if lo:
                medians[q_i] = -lo[0] if len(lo) > len(hi) else (-lo[0] + hi[0]) * 0.5
            q_i += 1

    return pd.DataFrame(
        {
            "supplier_number": query_days["supplier_number"],
            "inv_day": pd.to_datetime(query_days["inv_day"]),
            "prior_vendor_median_amount": medians,
        }
    )


def _empty_featured_frame() -> pd.DataFrame:
    """Return an empty DataFrame matching the full point-in-time output schema."""
    data: dict[str, pd.Series] = {}
    date_cols = {"invoice_date", "payment_date"}
    str_cols = {"operating_unit", "supplier_number", "invoice_number", "primary_department"}
    int_cols = {
        "days_to_pay",
        "is_late",
        "line_item_count",
        "dept_count",
        "vendor_prior_paid_count",
        "vendor_has_prior_credit",
        "dept_inv_count_7d",
        "dept_inv_count_30d",
        "same_day_vendor_inv_count",
        "inv_month",
        "inv_quarter",
        "inv_dayofweek",
        "days_to_fiscal_year_end",
    }
    for col in OUTPUT_COLUMNS:
        if col in date_cols:
            data[col] = pd.Series(dtype="datetime64[ns]")
        elif col in str_cols:
            data[col] = pd.Series(dtype="object")
        elif col in int_cols:
            data[col] = pd.Series(dtype="int64")
        else:
            data[col] = pd.Series(dtype="float64")
    return pd.DataFrame(data)


def _normalize_input_invoices(invoices_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure consistent column dtypes before DuckDB registration."""
    df = invoices_df.copy()
    df["operating_unit"] = df["operating_unit"].astype(str)
    df["supplier_number"] = df["supplier_number"].astype(str)
    df["invoice_number"] = df["invoice_number"].astype(str)
    df["primary_department"] = df["primary_department"].astype(str)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"]).dt.tz_localize(None)
    df["payment_date"] = pd.to_datetime(df["payment_date"]).dt.tz_localize(None)
    df["_row_order"] = np.arange(len(df), dtype=np.int64)
    return df


def build_point_in_time_features(
    invoices_df: pd.DataFrame,
    credit_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build strictly leak-free point-in-time features using only info knowable on invoice_date."""
    if invoices_df.empty:
        return _empty_featured_frame()

    norm_df = _normalize_input_invoices(invoices_df)
    credit_first_df = _prepare_credit_first_dates(credit_df)

    con = duckdb.connect()
    try:
        con.register("raw_invoices", norm_df)
        con.register("credit_first", credit_first_df)

        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE base_invoices AS
            SELECT
                _row_order,
                operating_unit,
                supplier_number,
                invoice_number,
                CAST(payment_number AS DOUBLE) AS payment_number,
                CAST(invoice_date AS TIMESTAMP) AS invoice_date,
                CAST(payment_date AS TIMESTAMP) AS payment_date,
                CAST(invoice_date AS DATE) AS inv_day,
                CAST(payment_date AS DATE) AS pay_day,
                primary_department,
                CAST(COALESCE(dept_count, 1) AS BIGINT) AS dept_count,
                CAST(COALESCE(line_item_count, 1) AS BIGINT) AS line_item_count,
                CAST(invoice_amount AS DOUBLE) AS invoice_amount,
                CAST(days_to_pay AS BIGINT) AS days_to_pay,
                CAST(is_late AS INTEGER) AS is_late
            FROM raw_invoices
            """
        )

        # Daily cumulative aggregates allow exact O(M log M) rolling & expanding lookups via ASOF JOIN.
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE vendor_pay_cum AS
            SELECT
                supplier_number,
                pay_day,
                SUM(day_cnt) OVER (PARTITION BY supplier_number ORDER BY pay_day) AS cum_cnt,
                SUM(day_late) OVER (PARTITION BY supplier_number ORDER BY pay_day) AS cum_late,
                SUM(day_days) OVER (PARTITION BY supplier_number ORDER BY pay_day) AS cum_days,
                SUM(day_days_sq) OVER (PARTITION BY supplier_number ORDER BY pay_day) AS cum_days_sq
            FROM (
                SELECT
                    supplier_number,
                    pay_day,
                    COUNT(*)::BIGINT AS day_cnt,
                    SUM(is_late)::BIGINT AS day_late,
                    SUM(CAST(days_to_pay AS DOUBLE)) AS day_days,
                    SUM(CAST(days_to_pay AS DOUBLE) * CAST(days_to_pay AS DOUBLE)) AS day_days_sq
                FROM base_invoices
                GROUP BY supplier_number, pay_day
            )
            """
        )

        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE dept_pay_cum AS
            SELECT
                primary_department,
                pay_day,
                SUM(day_cnt) OVER (PARTITION BY primary_department ORDER BY pay_day) AS cum_cnt,
                SUM(day_late) OVER (PARTITION BY primary_department ORDER BY pay_day) AS cum_late
            FROM (
                SELECT
                    primary_department,
                    pay_day,
                    COUNT(*)::BIGINT AS day_cnt,
                    SUM(is_late)::BIGINT AS day_late
                FROM base_invoices
                GROUP BY primary_department, pay_day
            )
            """
        )

        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE dept_inv_cum AS
            SELECT
                primary_department,
                inv_day,
                SUM(day_inv_cnt) OVER (PARTITION BY primary_department ORDER BY inv_day) AS cum_inv_cnt
            FROM (
                SELECT
                    primary_department,
                    inv_day,
                    COUNT(*)::BIGINT AS day_inv_cnt
                FROM base_invoices
                GROUP BY primary_department, inv_day
            )
            """
        )

        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE vendor_query_keys AS
            SELECT DISTINCT supplier_number, inv_day FROM base_invoices
            """
        )

        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE dept_query_keys AS
            SELECT DISTINCT primary_department, inv_day FROM base_invoices
            """
        )

        median_df = _compute_vendor_prior_medians(con)
        con.register("vendor_medians_raw", median_df)

        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE vendor_day_features AS
            SELECT
                q.supplier_number,
                q.inv_day,
                COALESCE(c1.cum_cnt, 0)::BIGINT AS vendor_prior_paid_count,
                (COALESCE(c1.cum_late, 0) + {_PRIOR_ALPHA}) / (COALESCE(c1.cum_cnt, 0) + {_PRIOR_ALPHA + _PRIOR_BETA}) AS vendor_prior_late_rate_smoothed,
                CASE
                    WHEN (COALESCE(c1.cum_cnt, 0) - COALESCE(c90.cum_cnt, 0)) > 0
                    THEN (COALESCE(c1.cum_days, 0.0) - COALESCE(c90.cum_days, 0.0))
                         / (COALESCE(c1.cum_cnt, 0) - COALESCE(c90.cum_cnt, 0))
                    ELSE {_COLD_START_DAYS}
                END AS vendor_prior_avg_days_90d,
                CASE
                    WHEN (COALESCE(c1.cum_cnt, 0) - COALESCE(c365.cum_cnt, 0)) > 0
                    THEN (COALESCE(c1.cum_days, 0.0) - COALESCE(c365.cum_days, 0.0))
                         / (COALESCE(c1.cum_cnt, 0) - COALESCE(c365.cum_cnt, 0))
                    ELSE {_COLD_START_DAYS}
                END AS vendor_prior_avg_days_365d,
                CASE
                    WHEN (COALESCE(c1.cum_cnt, 0) - COALESCE(c365.cum_cnt, 0)) > 1
                    THEN SQRT(
                        GREATEST(
                            (
                                (COALESCE(c1.cum_days_sq, 0.0) - COALESCE(c365.cum_days_sq, 0.0))
                                - POWER(COALESCE(c1.cum_days, 0.0) - COALESCE(c365.cum_days, 0.0), 2)
                                  / (COALESCE(c1.cum_cnt, 0) - COALESCE(c365.cum_cnt, 0))
                            ) / (COALESCE(c1.cum_cnt, 0) - COALESCE(c365.cum_cnt, 0) - 1.0),
                            0.0
                        )
                    )
                    ELSE 0.0
                END AS vendor_prior_std_days_365d,
                CASE
                    WHEN cf.first_credit_day IS NOT NULL
                     AND CAST(cf.first_credit_day AS DATE) < q.inv_day
                    THEN 1::BIGINT
                    ELSE 0::BIGINT
                END AS vendor_has_prior_credit,
                vm.prior_vendor_median_amount
            FROM vendor_query_keys q
            ASOF LEFT JOIN vendor_pay_cum c1
              ON q.supplier_number = c1.supplier_number
             AND (q.inv_day - 1) >= c1.pay_day
            ASOF LEFT JOIN vendor_pay_cum c90
              ON q.supplier_number = c90.supplier_number
             AND (q.inv_day - 91) >= c90.pay_day
            ASOF LEFT JOIN vendor_pay_cum c365
              ON q.supplier_number = c365.supplier_number
             AND (q.inv_day - 366) >= c365.pay_day
            LEFT JOIN credit_first cf
              ON q.supplier_number = cf.supplier_number
            LEFT JOIN vendor_medians_raw vm
              ON q.supplier_number = vm.supplier_number
             AND q.inv_day = CAST(vm.inv_day AS DATE)
            """
        )

        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE dept_day_features AS
            SELECT
                q.primary_department,
                q.inv_day,
                (COALESCE(di1.cum_inv_cnt, 0) - COALESCE(di7.cum_inv_cnt, 0))::BIGINT AS dept_inv_count_7d,
                (COALESCE(di1.cum_inv_cnt, 0) - COALESCE(di30.cum_inv_cnt, 0))::BIGINT AS dept_inv_count_30d,
                (COALESCE(dp.cum_late, 0) + {_PRIOR_ALPHA}) / (COALESCE(dp.cum_cnt, 0) + {_PRIOR_ALPHA + _PRIOR_BETA}) AS dept_prior_late_rate
            FROM dept_query_keys q
            ASOF LEFT JOIN dept_inv_cum di1
              ON q.primary_department = di1.primary_department
             AND (q.inv_day - 1) >= di1.inv_day
            ASOF LEFT JOIN dept_inv_cum di7
              ON q.primary_department = di7.primary_department
             AND (q.inv_day - 8) >= di7.inv_day
            ASOF LEFT JOIN dept_inv_cum di30
              ON q.primary_department = di30.primary_department
             AND (q.inv_day - 31) >= di30.inv_day
            ASOF LEFT JOIN dept_pay_cum dp
              ON q.primary_department = dp.primary_department
             AND (q.inv_day - 1) >= dp.pay_day
            """
        )

        result_df = con.sql(
            """
            SELECT
                b.operating_unit,
                b.supplier_number,
                b.invoice_number,
                b.payment_number,
                b.invoice_date,
                b.payment_date,
                b.primary_department,
                b.invoice_amount,
                b.days_to_pay,
                b.is_late,
                LN(1.0 + GREATEST(b.invoice_amount, 0.0)) AS log_invoice_amount,
                b.line_item_count,
                b.dept_count,
                CASE
                    WHEN vf.prior_vendor_median_amount IS NOT NULL
                     AND vf.prior_vendor_median_amount > 0.0
                    THEN b.invoice_amount / vf.prior_vendor_median_amount
                    ELSE 1.0
                END AS amount_vs_vendor_median_ratio,
                vf.vendor_prior_paid_count,
                vf.vendor_prior_late_rate_smoothed,
                vf.vendor_prior_avg_days_90d,
                vf.vendor_prior_avg_days_365d,
                vf.vendor_prior_std_days_365d,
                vf.vendor_has_prior_credit,
                df.dept_inv_count_7d,
                df.dept_inv_count_30d,
                df.dept_prior_late_rate,
                COUNT(*) OVER (PARTITION BY b.supplier_number, b.inv_day)::BIGINT AS same_day_vendor_inv_count,
                EXTRACT(MONTH FROM b.inv_day)::BIGINT AS inv_month,
                EXTRACT(QUARTER FROM b.inv_day)::BIGINT AS inv_quarter,
                (ISODOW(b.inv_day) - 1)::BIGINT AS inv_dayofweek,
                DATEDIFF(
                    'day',
                    b.inv_day,
                    CASE
                        WHEN MONTH(b.inv_day) <= 6 THEN MAKE_DATE(YEAR(b.inv_day), 6, 30)
                        ELSE MAKE_DATE(YEAR(b.inv_day) + 1, 6, 30)
                    END
                )::BIGINT AS days_to_fiscal_year_end
            FROM base_invoices b
            JOIN vendor_day_features vf
              ON b.supplier_number = vf.supplier_number
             AND b.inv_day = vf.inv_day
            JOIN dept_day_features df
              ON b.primary_department = df.primary_department
             AND b.inv_day = df.inv_day
            ORDER BY b._row_order
            """
        ).df()

        return result_df.loc[:, list(OUTPUT_COLUMNS)].copy()
    finally:
        con.close()


def split_out_of_time(
    featured_df: pd.DataFrame,
    train_end: str,
    val_end: str,
    test_end: str,
    train_start: str | None = _DEFAULT_TRAIN_START,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset strictly by invoice_date, excluding left-truncated and right-censored tails."""
    inv_dates = pd.to_datetime(featured_df["invoice_date"])
    t_end = pd.Timestamp(train_end)
    v_end = pd.Timestamp(val_end)
    te_end = pd.Timestamp(test_end)

    if train_start is not None:
        t_start = pd.Timestamp(train_start)
        train_mask = (inv_dates >= t_start) & (inv_dates <= t_end)
    else:
        train_mask = inv_dates <= t_end

    val_mask = (inv_dates > t_end) & (inv_dates <= v_end)
    test_mask = (inv_dates > v_end) & (inv_dates <= te_end)

    return (
        featured_df.loc[train_mask].reset_index(drop=True),
        featured_df.loc[val_mask].reset_index(drop=True),
        featured_df.loc[test_mask].reset_index(drop=True),
    )


def run_feature_pipeline(
    clean_parquet: Path,
    credit_parquet: Path,
    out_dir: Path,
    train_end: str | None = None,
    val_end: str | None = None,
    test_end: str | None = None,
) -> dict[str, tuple[int, float]]:
    """Load interim parquets, engineer point-in-time features, and write train/val/test splits."""
    t0 = time.perf_counter()
    invoices_df = pd.read_parquet(clean_parquet)
    credit_df = (
        pd.read_parquet(credit_parquet)
        if credit_parquet.exists()
        else pd.DataFrame(columns=["supplier_number", "invoice_date", "payment_date"])
    )
    logger.info(
        "Loaded %d clean invoices and %d credit memos in %.2fs",
        len(invoices_df),
        len(credit_df),
        time.perf_counter() - t0,
    )

    t1 = time.perf_counter()
    featured_df = build_point_in_time_features(invoices_df, credit_df)
    logger.info(
        "Engineered %d features across %d rows in %.2fs",
        len(FEATURE_COLS),
        len(featured_df),
        time.perf_counter() - t1,
    )

    tr_end = train_end or SETTINGS.train_end
    va_end = val_end or SETTINGS.val_end
    te_end = test_end or SETTINGS.test_end
    train_df, val_df, test_df = split_out_of_time(featured_df, tr_end, va_end, te_end)

    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {"train": train_df, "val": val_df, "test": test_df}
    summary: dict[str, tuple[int, float]] = {}
    for name, split_df in splits.items():
        out_path = out_dir / f"{name}.parquet"
        split_df.to_parquet(out_path, index=False, compression="snappy")
        late_rate = float(split_df["is_late"].mean()) if len(split_df) else 0.0
        summary[name] = (len(split_df), late_rate)
        logger.info(
            "Saved %s -> %s (%d rows, late_rate=%.4f)",
            name,
            out_path,
            len(split_df),
            late_rate,
        )
    return summary


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Build point-in-time AP features and out-of-time train/val/test parquets."
    )
    parser.add_argument(
        "--clean-parquet",
        type=Path,
        default=SETTINGS.interim_dir / "invoices_clean.parquet",
    )
    parser.add_argument(
        "--credit-parquet",
        type=Path,
        default=SETTINGS.interim_dir / "invoices_credit.parquet",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=SETTINGS.processed_dir,
    )
    args = parser.parse_args()
    run_feature_pipeline(args.clean_parquet, args.credit_parquet, args.out_dir)


if __name__ == "__main__":
    main()
