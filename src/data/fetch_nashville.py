import logging
from pathlib import Path
import pandas as pd
import pyarrow as pa_arrow
import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import SETTINGS
from src.data.validate_raw import ARROW_RAW_SCHEMA, validate_raw_schema

logger = logging.getLogger(__name__)


def _create_retry_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def fetch_nashville_data(
    output_path: Path | None = None,
    page_size: int = 2000,
    max_rows: int | None = None,
    session: requests.Session | None = None,
    resume: bool = True,
) -> Path:
    """Fetch Metro Nashville AP records from ArcGIS REST endpoint with pagination and validation."""
    target_path = Path(output_path) if output_path is not None else (SETTINGS.raw_dir / "nashville_raw.parquet")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    chunk_dir = target_path.parent / f".tmp_chunks_{target_path.stem}"

    offset = 0
    total_fetched = 0
    if resume and chunk_dir.exists():
        existing_chunks = sorted(chunk_dir.glob("chunk_*.parquet"))
        if existing_chunks:
            total_fetched = sum(pq.read_metadata(cf).num_rows for cf in existing_chunks)
            offset = total_fetched
            logger.info("Resuming ingestion from offset %d (%d rows cached in %d chunks)", offset, total_fetched, len(existing_chunks))
    else:
        if chunk_dir.exists():
            for cf in chunk_dir.glob("chunk_*.parquet"):
                try:
                    cf.unlink()
                except OSError:
                    pass
        chunk_dir.mkdir(parents=True, exist_ok=True)

    http_session = session if session is not None else _create_retry_session()
    query_url = f"{SETTINGS.nashville_feature_server_url.rstrip('/')}/query"
    column_order = [f.name for f in ARROW_RAW_SCHEMA]

    while True:
        if max_rows is not None and total_fetched >= max_rows:
            break

        current_limit = min(page_size, max_rows - total_fetched) if max_rows is not None else page_size
        params = {
            "where": "1=1",
            "outFields": "*",
            "orderByFields": "FID ASC",
            "resultOffset": offset,
            "resultRecordCount": current_limit,
            "f": "json",
        }

        resp = http_session.get(query_url, params=params, timeout=(15.0, 90.0))
        resp.raise_for_status()
        payload = resp.json()

        if "error" in payload:
            raise RuntimeError(f"ArcGIS query returned error: {payload['error']}")

        features = payload.get("features", [])
        if not features:
            break

        if max_rows is not None and (total_fetched + len(features)) > max_rows:
            features = features[: max_rows - total_fetched]

        records = [f["attributes"] for f in features]
        batch_df = pd.DataFrame(records)
        validated_batch = validate_raw_schema(batch_df)

        table = pa_arrow.Table.from_pandas(validated_batch[column_order], schema=ARROW_RAW_SCHEMA, preserve_index=False)
        chunk_file = chunk_dir / f"chunk_{offset:010d}.parquet"
        pq.write_table(table, chunk_file, compression="snappy")

        batch_count = len(features)
        offset += batch_count
        total_fetched += batch_count

        if total_fetched % 50000 == 0 or not payload.get("exceededTransferLimit", False):
            logger.info("Ingestion progress: %d rows fetched (offset=%d)...", total_fetched, offset)

        if not payload.get("exceededTransferLimit", False):
            break

    chunk_files = sorted(chunk_dir.glob("chunk_*.parquet"))
    temp_final = target_path.with_suffix(".tmp.parquet")

    if not chunk_files:
        empty_table = pa_arrow.Table.from_batches([], schema=ARROW_RAW_SCHEMA)
        pq.write_table(empty_table, temp_final, compression="snappy")
    else:
        with pq.ParquetWriter(temp_final, schema=ARROW_RAW_SCHEMA, compression="snappy") as writer:
            for cf in chunk_files:
                writer.write_table(pq.read_table(cf))

    temp_final.replace(target_path)

    for cf in chunk_files:
        try:
            cf.unlink()
        except OSError:
            pass
    try:
        chunk_dir.rmdir()
    except OSError:
        pass

    logger.info("Ingestion finished: %d rows saved to %s", total_fetched, target_path)
    return target_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    fetch_nashville_data()
