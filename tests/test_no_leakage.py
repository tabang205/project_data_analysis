import numpy as np
import pandas as pd

from src.features.build_features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLS,
    NUMERIC_FEATURES,
    TARGET_CLASSIFICATION,
    TARGET_REGRESSION,
    build_point_in_time_features,
    split_out_of_time,
)


def _make_invoice(
    *,
    supplier_number: str = "S1",
    invoice_number: str = "INV-1",
    payment_number: int = 1,
    invoice_date: str = "2024-01-01",
    payment_date: str = "2024-01-15",
    primary_department: str = "D1",
    operating_unit: str = "METRO_GG",
    invoice_amount: float = 100.0,
    days_to_pay: int = 14,
    is_late: int = 0,
    dept_count: int = 1,
    line_item_count: int = 1,
) -> dict:
    return {
        "operating_unit": operating_unit,
        "supplier_number": supplier_number,
        "invoice_number": invoice_number,
        "payment_number": payment_number,
        "invoice_date": pd.Timestamp(invoice_date),
        "payment_date": pd.Timestamp(payment_date),
        "primary_department": primary_department,
        "primary_dept_number": 1,
        "dept_count": dept_count,
        "line_item_count": line_item_count,
        "invoice_amount": invoice_amount,
        "check_amount": invoice_amount,
        "days_to_pay": days_to_pay,
        "is_late": is_late,
    }


def test_features_ignore_future_payments_after_invoice_date() -> None:
    inv_a = _make_invoice(
        invoice_number="A",
        payment_number=1,
        invoice_date="2024-01-01",
        payment_date="2024-03-01",
        days_to_pay=60,
        is_late=1,
    )
    inv_b = _make_invoice(
        invoice_number="B",
        payment_number=2,
        invoice_date="2024-02-01",
        payment_date="2024-02-10",
        days_to_pay=9,
        is_late=0,
    )
    df_feat = build_point_in_time_features(
        pd.DataFrame([inv_a, inv_b]),
        pd.DataFrame(columns=["supplier_number", "invoice_date"]),
    )
    row_b = df_feat.loc[df_feat["invoice_number"] == "B"].iloc[0]
    # On 2024-02-01, inv_a has NOT been paid yet (payment_date = 2024-03-01 > 2024-02-01)
    assert row_b["vendor_prior_paid_count"] == 0


def test_cold_start_vendor_finite_features() -> None:
    inv = pd.DataFrame(
        [
            _make_invoice(
                operating_unit="M",
                supplier_number="NEW",
                invoice_number="X",
                invoice_date="2024-05-01",
                payment_date="2024-05-01",
                invoice_amount=250.0,
                days_to_pay=0,
                is_late=0,
            )
        ]
    )
    df_feat = build_point_in_time_features(
        inv, pd.DataFrame(columns=["supplier_number", "invoice_date"])
    )
    assert df_feat.loc[0, "vendor_prior_paid_count"] == 0
    assert np.isfinite(df_feat.loc[0, "vendor_prior_late_rate_smoothed"])
    assert np.isfinite(df_feat.loc[0, "amount_vs_vendor_median_ratio"])
    for col in NUMERIC_FEATURES:
        assert np.isfinite(df_feat.loc[0, col]), f"Non-finite cold-start feature: {col}"


def test_time_split_enforces_censoring_cutoff() -> None:
    df = pd.DataFrame(
        {
            "invoice_date": pd.to_datetime(
                ["2023-01-01", "2024-08-01", "2025-05-01", "2025-11-15"]
            )
        }
    )
    tr, va, te = split_out_of_time(df, "2024-06-30", "2024-12-31", "2025-10-31")
    assert len(tr) == 1 and len(va) == 1 and len(te) == 1
    assert te["invoice_date"].max() <= pd.Timestamp("2025-10-31")


def test_feature_contract_excludes_target_and_payment_columns() -> None:
    forbidden = {
        "payment_date",
        "payment_number",
        "check_amount",
        "days_to_pay",
        "is_late",
        "supplier_number",
        "invoice_number",
        "invoice_date",
        "invoice_amount",
    }
    assert set(FEATURE_COLS) == set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES)
    assert len(FEATURE_COLS) == len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)
    assert forbidden.isdisjoint(set(FEATURE_COLS))
    assert TARGET_CLASSIFICATION == "is_late"
    assert TARGET_REGRESSION == "days_to_pay"


def test_same_day_payment_not_counted_as_prior() -> None:
    inv_a = _make_invoice(
        invoice_number="A",
        payment_number=1,
        invoice_date="2024-01-01",
        payment_date="2024-02-01",
        days_to_pay=31,
        is_late=1,
    )
    inv_b = _make_invoice(
        invoice_number="B",
        payment_number=2,
        invoice_date="2024-02-01",
        payment_date="2024-02-15",
        days_to_pay=14,
        is_late=0,
    )
    df_feat = build_point_in_time_features(
        pd.DataFrame([inv_a, inv_b]),
        pd.DataFrame(columns=["supplier_number", "invoice_date"]),
    )
    row_b = df_feat.loc[df_feat["invoice_number"] == "B"].iloc[0]
    assert row_b["vendor_prior_paid_count"] == 0
    assert row_b["vendor_prior_late_rate_smoothed"] == 0.25


def test_rolling_windows_median_credit_dept_and_fiscal_calendar() -> None:
    invoices = pd.DataFrame(
        [
            # Paid 150 days before target (2024-07-01): inside 365d, outside 90d
            _make_invoice(
                invoice_number="P1",
                payment_number=1,
                invoice_date="2024-01-15",
                payment_date="2024-02-02",
                invoice_amount=100.0,
                days_to_pay=18,
                is_late=0,
                primary_department="Public Works",
            ),
            # Paid 30 days before target (2024-07-01): inside both 90d & 365d
            _make_invoice(
                invoice_number="P2",
                payment_number=2,
                invoice_date="2024-04-20",
                payment_date="2024-06-01",
                invoice_amount=300.0,
                days_to_pay=42,
                is_late=1,
                primary_department="Public Works",
            ),
            # Issued 5 days before target in same dept (inside 7d & 30d dept workload),
            # but paid AFTER target (must not leak into prior payment stats!)
            _make_invoice(
                supplier_number="S2",
                invoice_number="D_RECENT",
                payment_number=3,
                invoice_date="2024-06-26",
                payment_date="2024-07-20",
                invoice_amount=500.0,
                days_to_pay=24,
                is_late=0,
                primary_department="Public Works",
            ),
            # Target invoice on 2024-07-01 + sibling on same day for S1
            _make_invoice(
                invoice_number="TARGET_1",
                payment_number=4,
                invoice_date="2024-07-01",
                payment_date="2024-07-10",
                invoice_amount=400.0,
                days_to_pay=9,
                is_late=0,
                primary_department="Public Works",
            ),
            _make_invoice(
                invoice_number="TARGET_2",
                payment_number=5,
                invoice_date="2024-07-01",
                payment_date="2024-07-11",
                invoice_amount=200.0,
                days_to_pay=10,
                is_late=0,
                primary_department="Parks",
            ),
        ]
    )
    credit_df = pd.DataFrame(
        [
            {
                "supplier_number": "S1",
                "invoice_date": pd.Timestamp("2024-05-01"),
                "payment_date": pd.Timestamp("2024-06-15"),
                "invoice_amount": -50.0,
            }
        ]
    )
    df_feat = build_point_in_time_features(invoices, credit_df)
    t1 = df_feat.loc[df_feat["invoice_number"] == "TARGET_1"].iloc[0]
    p2 = df_feat.loc[df_feat["invoice_number"] == "P2"].iloc[0]

    assert t1["vendor_prior_paid_count"] == 2
    assert np.isclose(t1["vendor_prior_avg_days_90d"], 42.0)
    assert np.isclose(t1["vendor_prior_avg_days_365d"], 30.0)
    assert np.isclose(t1["vendor_prior_std_days_365d"], np.std([18.0, 42.0], ddof=1))
    # Prior median of [100.0, 300.0] is 200.0 -> 400.0 / 200.0 = 2.0
    assert np.isclose(t1["amount_vs_vendor_median_ratio"], 2.0)
    assert t1["vendor_has_prior_credit"] == 1
    assert t1["same_day_vendor_inv_count"] == 2
    assert t1["dept_inv_count_7d"] == 1
    assert t1["dept_inv_count_30d"] == 1
    # Dept prior paid: P1 (0) + P2 (1) -> (1 + 2.5) / (2 + 10) = 3.5 / 12
    assert np.isclose(t1["dept_prior_late_rate"], 3.5 / 12.0)
    assert t1["inv_month"] == 7
    assert t1["inv_quarter"] == 3
    assert t1["inv_dayofweek"] == 0  # 2024-07-01 is Monday
    assert t1["days_to_fiscal_year_end"] == 364  # 2025-06-30 - 2024-07-01

    # On 2024-04-20, credit payment was 2024-06-15 >= 2024-04-20 -> no prior credit yet!
    assert p2["vendor_has_prior_credit"] == 0


def test_split_out_of_time_enforces_left_truncation_and_right_censoring() -> None:
    df = pd.DataFrame(
        {
            "invoice_date": pd.to_datetime(
                [
                    "2022-05-15",
                    "2022-07-01",
                    "2024-06-30",
                    "2024-07-01",
                    "2024-12-31",
                    "2025-01-01",
                    "2025-10-31",
                    "2025-11-01",
                ]
            )
        }
    )
    tr, va, te = split_out_of_time(df, "2024-06-30", "2024-12-31", "2025-10-31")
    assert len(tr) == 2
    assert len(va) == 2
    assert len(te) == 2
    assert tr["invoice_date"].min() >= pd.Timestamp("2022-07-01")
    assert tr["invoice_date"].max() <= pd.Timestamp("2024-06-30")
    assert va["invoice_date"].min() > pd.Timestamp("2024-06-30")
    assert va["invoice_date"].max() <= pd.Timestamp("2024-12-31")
    assert te["invoice_date"].min() > pd.Timestamp("2024-12-31")
    assert te["invoice_date"].max() <= pd.Timestamp("2025-10-31")


def test_empty_dataframe_returns_valid_schema() -> None:
    empty_inv = pd.DataFrame(
        columns=[
            "operating_unit",
            "supplier_number",
            "invoice_number",
            "payment_number",
            "invoice_date",
            "payment_date",
            "primary_department",
            "primary_dept_number",
            "dept_count",
            "line_item_count",
            "invoice_amount",
            "check_amount",
            "days_to_pay",
            "is_late",
        ]
    )
    empty_credit = pd.DataFrame(columns=["supplier_number", "invoice_date"])
    res = build_point_in_time_features(empty_inv, empty_credit)
    assert len(res) == 0
    for col in FEATURE_COLS:
        assert col in res.columns
