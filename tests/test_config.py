from pathlib import Path
from src.config import SETTINGS, Settings


def test_settings_splits_are_strictly_ordered_and_buffered():
    assert SETTINGS.train_end < SETTINGS.val_end < SETTINGS.test_end
    assert SETTINGS.late_threshold_days == 30
    assert SETTINGS.test_end == "2025-10-31"


def test_settings_directories_and_url_configured():
    assert isinstance(SETTINGS.BASE_DIR, Path)
    assert SETTINGS.raw_dir == SETTINGS.BASE_DIR / "data" / "raw"
    assert SETTINGS.interim_dir == SETTINGS.BASE_DIR / "data" / "interim"
    assert SETTINGS.processed_dir == SETTINGS.BASE_DIR / "data" / "processed"
    assert SETTINGS.artifacts_dir == SETTINGS.BASE_DIR / "artifacts"
    assert "Metro_Vendor_Payments_view" in SETTINGS.nashville_feature_server_url
    assert SETTINGS.min_valid_invoice_date == "2015-01-01"
    assert SETTINGS.max_valid_payment_date == "2025-12-31"
