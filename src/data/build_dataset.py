import argparse
import json
import logging
from pathlib import Path
from typing import Any

import duckdb

from src.config import SETTINGS

logger = logging.getLogger(__name__)

DEFAULT_SQL_PATH = SETTINGS.BASE_DIR / "sql" / "build_invoice_grain.sql"


def build_invoice_dataset(
    raw_parquet: Path,
    out_clean_parquet: Path,
    out_audit_json: Path,
    out_credit_parquet: Path,
    min_valid_invoice_date: str | None = None,
    max_valid_payment_date: str | None = None,
    late_threshold_days: int | None = None,
    sql_path: Path | None = None,
) -> dict[str, int]:
    """Resolve raw line items to invoice grain, audit anomalies, and split outputs."""
    min_inv_date = min_valid_invoice_date or SETTINGS.min_valid_invoice_date
    max_pay_date = max_valid_payment_date or SETTINGS.max_valid_payment_date
    late_threshold = (
        late_threshold_days
        if late_threshold_days is not None
        else SETTINGS.late_threshold_days
    )
    resolved_sql_path = sql_path or DEFAULT_SQL_PATH

    raw_path_obj = Path(raw_parquet).resolve()
    clean_path_obj = Path(out_clean_parquet).resolve()
    credit_path_obj = Path(out_credit_parquet).resolve()
    audit_path_obj = Path(out_audit_json).resolve()

    clean_path_obj.parent.mkdir(parents=True, exist_ok=True)
    credit_path_obj.parent.mkdir(parents=True, exist_ok=True)
    audit_path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(resolved_sql_path, "r", encoding="utf-8") as f:
        sql_content = f.read()

    con = duckdb.connect()
    try:
        raw_posix = raw_path_obj.as_posix()
        con.execute(
            f"CREATE OR REPLACE TEMP VIEW raw_table AS SELECT * FROM read_parquet('{raw_posix}')"
        )

        params: dict[str, Any] = {
            "min_valid_invoice_date": min_inv_date,
            "max_valid_payment_date": max_pay_date,
            "late_threshold_days": late_threshold,
        }

        statements = [s.strip() for s in sql_content.split(";") if s.strip()]
        for stmt in statements:
            stmt_params = {k: v for k, v in params.items() if f"${k}" in stmt}
            con.execute(stmt, stmt_params)

        raw_rows = int(con.execute("SELECT COUNT(*) FROM raw_table").fetchone()[0])
        status_counts = dict(
            con.execute(
                "SELECT audit_status, COUNT(*) FROM classified_raw GROUP BY audit_status"
            ).fetchall()
        )
        dropped_missing_keys = int(status_counts.get("missing_keys", 0))
        dropped_bad_dates = int(status_counts.get("bad_dates", 0))

        clean_posix = clean_path_obj.as_posix()
        credit_posix = credit_path_obj.as_posix()

        con.execute(
            f"COPY (SELECT * FROM grouped_invoices WHERE invoice_amount > 0) "
            f"TO '{clean_posix}' (FORMAT PARQUET, COMPRESSION SNAPPY)"
        )
        con.execute(
            f"COPY (SELECT * FROM grouped_invoices WHERE invoice_amount <= 0) "
            f"TO '{credit_posix}' (FORMAT PARQUET, COMPRESSION SNAPPY)"
        )

        clean_invoice_rows = int(
            con.execute(
                f"SELECT COUNT(*) FROM read_parquet('{clean_posix}')"
            ).fetchone()[0]
        )
        credit_memo_rows = int(
            con.execute(
                f"SELECT COUNT(*) FROM read_parquet('{credit_posix}')"
            ).fetchone()[0]
        )
        grouped_rows = clean_invoice_rows + credit_memo_rows

        stats: dict[str, int] = {
            "raw_rows": raw_rows,
            "grouped_rows": grouped_rows,
            "dropped_missing_keys": dropped_missing_keys,
            "dropped_bad_dates": dropped_bad_dates,
            "credit_memo_rows": credit_memo_rows,
            "clean_invoice_rows": clean_invoice_rows,
        }

        with open(audit_path_obj, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        logger.info(
            "Invoice grain resolution complete. Raw: %d, Grouped: %d (Clean: %d, Credit: %d), Dropped: bad_dates=%d, missing_keys=%d",
            raw_rows,
            grouped_rows,
            clean_invoice_rows,
            credit_memo_rows,
            dropped_bad_dates,
            dropped_missing_keys,
        )
        return stats
    finally:
        con.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Consolidate raw AP lines to invoice grain and audit anomalies via DuckDB."
    )
    parser.add_argument(
        "--raw-parquet",
        type=Path,
        default=SETTINGS.raw_dir / "nashville_raw.parquet",
        help="Input raw AP Parquet file.",
    )
    parser.add_argument(
        "--out-clean-parquet",
        type=Path,
        default=SETTINGS.interim_dir / "invoices_clean.parquet",
        help="Output clean positive-amount invoices Parquet file.",
    )
    parser.add_argument(
        "--out-audit-json",
        type=Path,
        default=SETTINGS.interim_dir / "invoices_audit_summary.json",
        help="Output JSON audit summary metrics.",
    )
    parser.add_argument(
        "--out-credit-parquet",
        type=Path,
        default=SETTINGS.interim_dir / "invoices_credit.parquet",
        help="Output credit memos Parquet file.",
    )
    args = parser.parse_args()

    build_invoice_dataset(
        raw_parquet=args.raw_parquet,
        out_clean_parquet=args.out_clean_parquet,
        out_audit_json=args.out_audit_json,
        out_credit_parquet=args.out_credit_parquet,
    )


if __name__ == "__main__":
    main()
