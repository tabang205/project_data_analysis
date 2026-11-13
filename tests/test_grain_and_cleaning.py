from pathlib import Path
import json
import pandas as pd
import pytest

from src.data.build_dataset import build_invoice_dataset


def test_grain_aggregates_multiline_multidept_invoice(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        [
            {
                "FID": 1,
                "Operating_Unit": "METRO",
                "Supplier_Number": "S1",
                "Invoice_Number": "INV-A",
                "Payment_Number": 100,
                "Invoice_Date": pd.Timestamp("2024-01-01"),
                "Payment_Date": pd.Timestamp("2024-02-05"),
                "Department_Name": "Health",
                "Department_Number": 20,
                "Distribution_Amount": 300.0,
                "Check_Amount": 500.0,
            },
            {
                "FID": 2,
                "Operating_Unit": "METRO",
                "Supplier_Number": "S1",
                "Invoice_Number": "INV-A",
                "Payment_Number": 100,
                "Invoice_Date": pd.Timestamp("2024-01-01"),
                "Payment_Date": pd.Timestamp("2024-02-05"),
                "Department_Name": "Admin",
                "Department_Number": 10,
                "Distribution_Amount": 200.0,
                "Check_Amount": 500.0,
            },
        ]
    )
    raw_path = tmp_path / "raw.parquet"
    raw.to_parquet(raw_path)
    build_invoice_dataset(
        raw_path,
        tmp_path / "clean.parquet",
        tmp_path / "audit.json",
        tmp_path / "credit.parquet",
    )
    clean = pd.read_parquet(tmp_path / "clean.parquet")
    assert len(clean) == 1
    assert clean.loc[0, "invoice_amount"] == 500.0
    assert clean.loc[0, "line_item_count"] == 2
    assert clean.loc[0, "dept_count"] == 2
    assert clean.loc[0, "primary_department"] == "Admin"
    assert clean.loc[0, "primary_dept_number"] == 10
    assert clean.loc[0, "days_to_pay"] == 35
    assert clean.loc[0, "is_late"] == 1


def test_audit_splits_bad_dates_and_credit_memos(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        [
            {
                "FID": 1,
                "Operating_Unit": "M",
                "Supplier_Number": "S1",
                "Invoice_Number": "OLD",
                "Payment_Number": 1,
                "Invoice_Date": pd.Timestamp("1991-01-01"),
                "Payment_Date": pd.Timestamp("2024-01-10"),
                "Department_Name": "D",
                "Department_Number": 1,
                "Distribution_Amount": 100.0,
                "Check_Amount": 100.0,
            },
            {
                "FID": 2,
                "Operating_Unit": "M",
                "Supplier_Number": "S1",
                "Invoice_Number": "FUT",
                "Payment_Number": 2,
                "Invoice_Date": pd.Timestamp("2083-01-01"),
                "Payment_Date": pd.Timestamp("2024-01-10"),
                "Department_Name": "D",
                "Department_Number": 1,
                "Distribution_Amount": 100.0,
                "Check_Amount": 100.0,
            },
            {
                "FID": 3,
                "Operating_Unit": "M",
                "Supplier_Number": "S1",
                "Invoice_Number": "REV",
                "Payment_Number": 3,
                "Invoice_Date": pd.Timestamp("2024-02-10"),
                "Payment_Date": pd.Timestamp("2024-02-01"),
                "Department_Name": "D",
                "Department_Number": 1,
                "Distribution_Amount": 100.0,
                "Check_Amount": 100.0,
            },
            {
                "FID": 4,
                "Operating_Unit": "M",
                "Supplier_Number": "S1",
                "Invoice_Number": "CRED",
                "Payment_Number": 4,
                "Invoice_Date": pd.Timestamp("2024-01-01"),
                "Payment_Date": pd.Timestamp("2024-01-05"),
                "Department_Name": "D",
                "Department_Number": 1,
                "Distribution_Amount": -50.0,
                "Check_Amount": 50.0,
            },
            {
                "FID": 5,
                "Operating_Unit": "M",
                "Supplier_Number": "S1",
                "Invoice_Number": "OK",
                "Payment_Number": 5,
                "Invoice_Date": pd.Timestamp("2024-01-01"),
                "Payment_Date": pd.Timestamp("2024-01-11"),
                "Department_Name": "D",
                "Department_Number": 1,
                "Distribution_Amount": 100.0,
                "Check_Amount": 100.0,
            },
        ]
    )
    raw_path = tmp_path / "raw.parquet"
    raw.to_parquet(raw_path)
    stats = build_invoice_dataset(
        raw_path,
        tmp_path / "clean.parquet",
        tmp_path / "audit.json",
        tmp_path / "credit.parquet",
    )
    clean = pd.read_parquet(tmp_path / "clean.parquet")
    credit = pd.read_parquet(tmp_path / "credit.parquet")
    assert len(clean) == 1 and clean.loc[0, "invoice_number"] == "OK"
    assert len(credit) == 1 and credit.loc[0, "invoice_number"] == "CRED"
    assert stats["raw_rows"] == 5
    assert stats["dropped_missing_keys"] == 0
    assert stats["dropped_bad_dates"] == 3
    assert stats["grouped_rows"] == 2
    assert stats["credit_memo_rows"] == 1
    assert stats["clean_invoice_rows"] == 1


def test_audit_handles_missing_keys(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        [
            {
                "FID": 1,
                "Operating_Unit": None,
                "Supplier_Number": "S1",
                "Invoice_Number": "INV1",
                "Payment_Number": 1,
                "Invoice_Date": pd.Timestamp("2024-01-01"),
                "Payment_Date": pd.Timestamp("2024-01-10"),
                "Department_Name": "D",
                "Department_Number": 1,
                "Distribution_Amount": 100.0,
                "Check_Amount": 100.0,
            },
            {
                "FID": 2,
                "Operating_Unit": "M",
                "Supplier_Number": "S1",
                "Invoice_Number": "INV2",
                "Payment_Number": 2,
                "Invoice_Date": pd.Timestamp("2024-01-01"),
                "Payment_Date": pd.Timestamp("2024-01-10"),
                "Department_Name": "D",
                "Department_Number": 1,
                "Distribution_Amount": 100.0,
                "Check_Amount": 100.0,
            },
        ]
    )
    raw_path = tmp_path / "raw.parquet"
    raw.to_parquet(raw_path)
    stats = build_invoice_dataset(
        raw_path,
        tmp_path / "clean.parquet",
        tmp_path / "audit.json",
        tmp_path / "credit.parquet",
    )
    assert stats["raw_rows"] == 2
    assert stats["dropped_missing_keys"] == 1
    assert stats["clean_invoice_rows"] == 1


def test_audit_json_and_column_schema(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        [
            {
                "FID": 1,
                "Operating_Unit": "M",
                "Supplier_Number": "S1",
                "Invoice_Number": "INV1",
                "Payment_Number": 1,
                "Invoice_Date": pd.Timestamp("2024-01-01"),
                "Payment_Date": pd.Timestamp("2024-01-15"),
                "Department_Name": "Public Works",
                "Department_Number": 42,
                "Distribution_Amount": 500.0,
                "Check_Amount": 500.0,
            }
        ]
    )
    raw_path = tmp_path / "raw.parquet"
    clean_path = tmp_path / "clean.parquet"
    audit_path = tmp_path / "audit.json"
    credit_path = tmp_path / "credit.parquet"
    raw.to_parquet(raw_path)

    stats = build_invoice_dataset(raw_path, clean_path, audit_path, credit_path)
    assert audit_path.exists()
    with open(audit_path, "r", encoding="utf-8") as f:
        loaded_audit = json.load(f)

    for key in [
        "raw_rows",
        "grouped_rows",
        "dropped_missing_keys",
        "dropped_bad_dates",
        "credit_memo_rows",
        "clean_invoice_rows",
    ]:
        assert key in stats
        assert stats[key] == loaded_audit[key]

    clean = pd.read_parquet(clean_path)
    expected_cols = [
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
    assert list(clean.columns) == expected_cols
