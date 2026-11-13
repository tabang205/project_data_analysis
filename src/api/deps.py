"""Cached artifact dependency provider for the FastAPI workbench."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from src.app.loaders import (
    load_audit_summary,
    load_baseline_metrics,
    load_gbdt_metrics,
    load_test_predictions,
    resolve_data_dirs,
)
from src.config import SETTINGS


@dataclass(frozen=True)
class ArtifactBundle:
    """In-memory snapshot of resolved directories and loaded project artifacts."""

    artifacts_dir: Path
    interim_dir: Path
    docs_dir: Path
    audit_summary: dict[str, Any] | None
    baseline_metrics: dict[str, Any] | None
    gbdt_metrics: dict[str, Any] | None
    test_predictions: pd.DataFrame | None

    @property
    def kaggle_ready(self) -> bool:
        return self.gbdt_metrics is not None and self.test_predictions is not None


@lru_cache(maxsize=4)
def _load_bundle_cached(artifacts_str: str, interim_str: str) -> ArtifactBundle:
    artifacts_dir = Path(artifacts_str)
    interim_dir = Path(interim_str)
    return ArtifactBundle(
        artifacts_dir=artifacts_dir,
        interim_dir=interim_dir,
        docs_dir=SETTINGS.base_dir / "docs" / "learning",
        audit_summary=load_audit_summary(interim_dir),
        baseline_metrics=load_baseline_metrics(artifacts_dir),
        gbdt_metrics=load_gbdt_metrics(artifacts_dir),
        test_predictions=load_test_predictions(artifacts_dir),
    )


def get_artifact_bundle() -> ArtifactBundle:
    """FastAPI dependency returning the cached ArtifactBundle for current env paths."""
    artifacts_dir, interim_dir = resolve_data_dirs()
    return _load_bundle_cached(str(artifacts_dir.resolve()), str(interim_dir.resolve()))


def clear_artifact_cache() -> None:
    """Clear the cached artifact bundle (used in tests when switching env dirs)."""
    _load_bundle_cached.cache_clear()
