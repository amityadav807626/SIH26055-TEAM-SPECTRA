"""Generic adapter for normalized band/time activity data.

Accepted schema: band,time_slot,activity. This adapter intentionally does not
parse source-specific radar/emitter fields.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Any
import csv

@dataclass(frozen=True)
class ActivityRecord:
    band: int
    time_slot: int
    activity: int

    def __post_init__(self) -> None:
        if self.band < 0:
            raise ValueError("band must be non-negative")
        if self.time_slot < 0:
            raise ValueError("time_slot must be non-negative")
        if self.activity not in (0, 1):
            raise ValueError("activity must be 0 or 1")

class ActivityDatasetAdapter:
    REQUIRED_COLUMNS = ("band", "time_slot", "activity")

    def __init__(self, path: str | Path):
        self.path = Path(path)

    @classmethod
    def _record(cls, row: Mapping[str, Any], line_no: int) -> ActivityRecord:
        missing = [c for c in cls.REQUIRED_COLUMNS if c not in row]
        if missing:
            raise ValueError(f"line {line_no}: missing columns: {', '.join(missing)}")
        try:
            return ActivityRecord(int(row["band"]), int(row["time_slot"]), int(row["activity"]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"line {line_no}: band/time_slot/activity must be integers") from exc

    def load(self) -> list[ActivityRecord]:
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        with self.path.open("r", newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                raise ValueError("dataset has no header")
            missing = [c for c in self.REQUIRED_COLUMNS if c not in reader.fieldnames]
            if missing:
                raise ValueError(f"dataset missing required columns: {', '.join(missing)}")
            records = [self._record(row, i) for i, row in enumerate(reader, start=2)]
        if not records:
            raise ValueError("dataset is empty")
        return records

    @staticmethod
    def validate(records: Iterable[ActivityRecord], n_bands: int | None = None,
                 horizon: int | None = None) -> list[ActivityRecord]:
        out = list(records)
        if not out:
            raise ValueError("no activity records supplied")
        for item in out:
            if n_bands is not None and item.band >= n_bands:
                raise ValueError(f"band {item.band} is outside configured range 0..{n_bands - 1}")
            if horizon is not None and item.time_slot >= horizon:
                raise ValueError(f"time_slot {item.time_slot} is outside configured range 0..{horizon - 1}")
        return out

    @staticmethod
    def to_grid(records: Iterable[ActivityRecord], n_bands: int, horizon: int):
        import numpy as np
        checked = ActivityDatasetAdapter.validate(records, n_bands, horizon)
        grid = np.zeros((horizon, n_bands), dtype=np.int8)
        for item in checked:
            grid[item.time_slot, item.band] = item.activity
        return grid


# Flexible jury-CSV normalization. This operates only on generic activity
# semantics; it does not interpret source-specific RF/PDW fields.
JURY_ALIASES = {
    "band": ("band", "channel", "channel_id", "band_id", "bin", "bin_id", "channel_index"),
    "time_slot": ("time_slot", "timeslot", "time", "timestep", "time_step", "slot", "timestamp", "sample", "sample_index"),
    "activity": ("activity", "active", "present", "presence", "occupied", "occupancy", "label", "target", "event")
}

def _norm_name(value: str) -> str:
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())

def detect_jury_columns(fieldnames: list[str]) -> dict[str, list[str]]:
    normalized = {_norm_name(f): f for f in fieldnames}
    out = {}
    for key, aliases in JURY_ALIASES.items():
        out[key] = list(dict.fromkeys(normalized[_norm_name(a)] for a in aliases if _norm_name(a) in normalized))
    return out

def _activity_value(value: str) -> int:
    v = str(value).strip().lower()
    if v in {"1", "true", "yes", "y", "on", "active", "present", "occupied"}: return 1
    if v in {"0", "false", "no", "n", "off", "inactive", "absent", "empty", "unoccupied"}: return 0
    try:
        return 1 if float(v) != 0 else 0
    except ValueError as exc:
        raise ValueError(f"activity value '{value}' is not interpretable as 0/1") from exc

def normalize_jury_csv(source: str | Path, destination: str | Path, band_col: str, time_col: str, activity_col: str) -> dict[str, int | str]:
    source, destination = Path(source), Path(destination)
    with source.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        for c in (band_col, time_col, activity_col):
            if c not in fields:
                raise ValueError(f"selected column '{c}' is not present in the CSV")
        rows = list(reader)
    if not rows:
        raise ValueError("dataset is empty")

    # Map distinct channel/time values to compact integer indices. This allows
    # harmless labels such as A/B or non-contiguous numeric IDs.
    bands = {}
    times = {}
    normalized_rows = []
    for line_no, row in enumerate(rows, start=2):
        bv, tv = str(row[band_col]).strip(), str(row[time_col]).strip()
        if bv == "" or tv == "":
            raise ValueError(f"line {line_no}: band/time value is empty")
        if bv not in bands: bands[bv] = len(bands)
        if tv not in times: times[tv] = len(times)
        normalized_rows.append((bands[bv], times[tv], _activity_value(row[activity_col])))

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["band", "time_slot", "activity"])
        writer.writerows(normalized_rows)
    return {"records": len(normalized_rows), "bands": len(bands), "time_slots": len(times), "schema": "band,time_slot,activity"}
