import pandas as pd
import pytest
from unittest.mock import MagicMock
from src.data.fetch_nashville import fetch_nashville_data
from src.data.validate_raw import validate_raw_schema


def test_fetch_paginates_via_exceeded_transfer_limit(tmp_path):
    mock_session = MagicMock()
    page1 = {
        "features": [
            {
                "attributes": {
                    "FID": 1,
                    "Operating_Unit": "A",
                    "Supplier_Number": "S1",
                    "Invoice_Number": "I1",
                    "Distribution_Amount": 100.0,
                    "Invoice_Date": 1665446400000,
                    "Payment_Date": 1665446400000,
                    "Payment_Number": 10,
                    "Check_Amount": 100.0,
                    "Department_Name": "D1",
                    "Department_Number": 1,
                }
            }
        ],
        "exceededTransferLimit": True,
    }
    page2 = {
        "features": [
            {
                "attributes": {
                    "FID": 2,
                    "Operating_Unit": "A",
                    "Supplier_Number": "S2",
                    "Invoice_Number": "I2",
                    "Distribution_Amount": 200.0,
                    "Invoice_Date": 1665532800000,
                    "Payment_Date": 1665619200000,
                    "Payment_Number": 11,
                    "Check_Amount": 200.0,
                    "Department_Name": "D1",
                    "Department_Number": 1,
                }
            }
        ],
        "exceededTransferLimit": False,
    }
    resp1, resp2 = MagicMock(), MagicMock()
    resp1.json.return_value, resp2.json.return_value = page1, page2
    resp1.raise_for_status = resp2.raise_for_status = lambda: None
    mock_session.get.side_effect = [resp1, resp2]

    out = fetch_nashville_data(tmp_path / "raw.parquet", page_size=1, session=mock_session)
    df = pd.read_parquet(out)
    assert len(df) == 2
    assert list(df["FID"]) == [1, 2]


def test_validate_raw_schema_parses_epoch_ms_dates():
    df = pd.DataFrame(
        [
            {
                "FID": 1,
                "Operating_Unit": "METRO",
                "Supplier_Number": "797",
                "Invoice_Number": "INV1",
                "Distribution_Amount": 50.0,
                "Invoice_Date": 1665446400000,
                "Payment_Date": 1665532800000,
                "Payment_Number": 10,
                "Check_Amount": 50.0,
                "Department_Name": "MNPS",
                "Department_Number": 80,
            }
        ]
    )
    cleaned = validate_raw_schema(df)
    assert pd.api.types.is_datetime64_any_dtype(cleaned["Invoice_Date"])
    assert cleaned.loc[0, "Invoice_Date"] == pd.Timestamp("2022-10-11")


def test_fetch_respects_max_rows(tmp_path):
    mock_session = MagicMock()
    page1 = {
        "features": [
            {
                "attributes": {
                    "FID": 1,
                    "Operating_Unit": "A",
                    "Supplier_Number": "S1",
                    "Invoice_Number": "I1",
                    "Distribution_Amount": 100.0,
                    "Invoice_Date": 1665446400000,
                    "Payment_Date": 1665446400000,
                    "Payment_Number": 10,
                    "Check_Amount": 100.0,
                    "Department_Name": "D1",
                    "Department_Number": 1,
                }
            },
            {
                "attributes": {
                    "FID": 2,
                    "Operating_Unit": "A",
                    "Supplier_Number": "S2",
                    "Invoice_Number": "I2",
                    "Distribution_Amount": 200.0,
                    "Invoice_Date": 1665532800000,
                    "Payment_Date": 1665619200000,
                    "Payment_Number": 11,
                    "Check_Amount": 200.0,
                    "Department_Name": "D1",
                    "Department_Number": 1,
                }
            },
        ],
        "exceededTransferLimit": True,
    }
    resp1 = MagicMock()
    resp1.json.return_value = page1
    resp1.raise_for_status = lambda: None
    mock_session.get.return_value = resp1

    out = fetch_nashville_data(tmp_path / "raw.parquet", page_size=2, max_rows=1, session=mock_session)
    df = pd.read_parquet(out)
    assert len(df) == 1
    assert df.loc[0, "FID"] == 1


def test_validate_raw_schema_rejects_null_fid():
    df = pd.DataFrame(
        [
            {
                "FID": None,
                "Operating_Unit": "METRO",
                "Supplier_Number": "797",
                "Invoice_Number": "INV1",
                "Distribution_Amount": 50.0,
                "Invoice_Date": 1665446400000,
                "Payment_Date": 1665532800000,
                "Payment_Number": 10,
                "Check_Amount": 50.0,
                "Department_Name": "MNPS",
                "Department_Number": 80,
            }
        ]
    )
    with pytest.raises(Exception):
        validate_raw_schema(df)


def test_validate_raw_schema_handles_out_of_bounds_and_null_dates():
    df = pd.DataFrame(
        [
            {
                "FID": 1,
                "Operating_Unit": "METRO",
                "Supplier_Number": "797",
                "Invoice_Number": "INV1",
                "Distribution_Amount": 50.0,
                "Invoice_Date": 3585945600000,  # Year 2083
                "Payment_Date": None,
                "Payment_Number": 10,
                "Check_Amount": 50.0,
                "Department_Name": "MNPS",
                "Department_Number": 80,
            },
            {
                "FID": 2,
                "Operating_Unit": "METRO",
                "Supplier_Number": "797",
                "Invoice_Number": "INV2",
                "Distribution_Amount": 50.0,
                "Invoice_Date": 10**16,  # Far future overflow
                "Payment_Date": -10**16,  # Far past overflow
                "Payment_Number": 10,
                "Check_Amount": 50.0,
                "Department_Name": "MNPS",
                "Department_Number": 80,
            },
        ]
    )
    cleaned = validate_raw_schema(df)
    assert cleaned.loc[0, "Invoice_Date"] == pd.Timestamp("2083-08-20")
    assert pd.isna(cleaned.loc[0, "Payment_Date"])
    assert pd.isna(cleaned.loc[1, "Invoice_Date"])
    assert pd.isna(cleaned.loc[1, "Payment_Date"])
