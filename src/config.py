from dataclasses import dataclass, field
from pathlib import Path

_DEFAULT_BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    """Project-wide immutable configuration settings."""

    BASE_DIR: Path = _DEFAULT_BASE_DIR
    raw_dir: Path = field(default=None)  # type: ignore[assignment]
    interim_dir: Path = field(default=None)  # type: ignore[assignment]
    processed_dir: Path = field(default=None)  # type: ignore[assignment]
    artifacts_dir: Path = field(default=None)  # type: ignore[assignment]

    nashville_feature_server_url: str = (
        "https://services2.arcgis.com/HdTo6HJqh92wn4D8/arcgis/rest/services/Metro_Vendor_Payments_view/FeatureServer/0"
    )

    late_threshold_days: int = 30
    train_end: str = "2024-06-30"
    val_end: str = "2024-12-31"
    test_end: str = "2025-10-31"
    min_valid_invoice_date: str = "2015-01-01"
    max_valid_payment_date: str = "2025-12-31"

    def __post_init__(self) -> None:
        if self.raw_dir is None:
            object.__setattr__(self, "raw_dir", self.BASE_DIR / "data" / "raw")
        if self.interim_dir is None:
            object.__setattr__(self, "interim_dir", self.BASE_DIR / "data" / "interim")
        if self.processed_dir is None:
            object.__setattr__(self, "processed_dir", self.BASE_DIR / "data" / "processed")
        if self.artifacts_dir is None:
            object.__setattr__(self, "artifacts_dir", self.BASE_DIR / "artifacts")

    @property
    def base_dir(self) -> Path:
        return self.BASE_DIR


SETTINGS = Settings()
