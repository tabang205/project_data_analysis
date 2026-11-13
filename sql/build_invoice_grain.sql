-- sql/build_invoice_grain.sql
-- DuckDB SQL script for AP line-item grain resolution and data quality audit.
-- Purpose:
--   1. Flags missing primary identifiers and anomalous date ranges.
--   2. Consolidates distribution line items to canonical invoice grain:
--      (Operating_Unit, Supplier_Number, Invoice_Number, Payment_Number, Invoice_Date, Payment_Date)
--   3. Calculates invoice-level aggregates and payment turnaround (days_to_pay, is_late).

CREATE OR REPLACE TEMP TABLE classified_raw AS
SELECT
    *,
    CASE
        WHEN Operating_Unit IS NULL
          OR Supplier_Number IS NULL
          OR Invoice_Number IS NULL
          OR Payment_Number IS NULL
          OR Invoice_Date IS NULL
          OR Payment_Date IS NULL
        THEN 'missing_keys'
        WHEN CAST(Invoice_Date AS DATE) < CAST($min_valid_invoice_date AS DATE)
          OR CAST(Invoice_Date AS DATE) > CAST($max_valid_payment_date AS DATE)
          OR CAST(Payment_Date AS DATE) < CAST($min_valid_invoice_date AS DATE)
          OR CAST(Payment_Date AS DATE) > CAST($max_valid_payment_date AS DATE)
          OR Invoice_Date > Payment_Date
        THEN 'bad_dates'
        ELSE 'valid'
    END AS audit_status
FROM raw_table;

CREATE OR REPLACE TEMP TABLE grouped_invoices AS
SELECT
    Operating_Unit AS operating_unit,
    Supplier_Number AS supplier_number,
    Invoice_Number AS invoice_number,
    Payment_Number AS payment_number,
    Invoice_Date AS invoice_date,
    Payment_Date AS payment_date,
    MIN(Department_Name) AS primary_department,
    MIN(Department_Number) AS primary_dept_number,
    COUNT(DISTINCT Department_Number) AS dept_count,
    COUNT(*) AS line_item_count,
    ROUND(SUM(Distribution_Amount), 2) AS invoice_amount,
    MAX(Check_Amount) AS check_amount,
    DATEDIFF('day', Invoice_Date, Payment_Date) AS days_to_pay,
    CASE
        WHEN DATEDIFF('day', Invoice_Date, Payment_Date) > CAST($late_threshold_days AS INTEGER) THEN 1
        ELSE 0
    END AS is_late
FROM classified_raw
WHERE audit_status = 'valid'
GROUP BY
    Operating_Unit,
    Supplier_Number,
    Invoice_Number,
    Payment_Number,
    Invoice_Date,
    Payment_Date;
