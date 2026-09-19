"""Loader for compact, label-derived TSRD activity proxies.

The raw TSRD HDF5/PDW fields are intentionally kept outside SPECTRA.  The
files in ``tsrd_activity/`` contain only the generic ``band,time_slot,activity``
interface consumed by the existing scheduler.
"""
from __future__ import annotations
from pathlib import Path
from .activity_adapter import ActivityDatasetAdapter

TSRD_ACTIVITY_DIR = Path(__file__).resolve().parent / "tsrd_activity"
TSRD_SPLITS = (
    "train_scan", "val_scan", "test_scan",
    "train_stare", "val_stare", "test_stare",
)


def load_tsrd_activity(n_bands: int, horizon: int, split: str):
    if split not in TSRD_SPLITS:
        raise ValueError(f"unknown TSRD split: {split}")
    path = TSRD_ACTIVITY_DIR / f"{split}.csv"
    records = ActivityDatasetAdapter(path).load()
    records = [r for r in records if r.band < n_bands and r.time_slot < horizon]
    if not records:
        raise ValueError("TSRD activity proxy has no records inside the configured range")
    return ActivityDatasetAdapter.to_grid(records, n_bands=n_bands, horizon=horizon)
