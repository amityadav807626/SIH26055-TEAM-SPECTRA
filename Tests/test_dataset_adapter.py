import csv
from pathlib import Path
import numpy as np
import pytest
from Backend.datasets.activity_adapter import ActivityDatasetAdapter, ActivityRecord

def test_activity_adapter_loads_and_builds_grid(tmp_path: Path):
    path = tmp_path / "activity.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["band", "time_slot", "activity"])
        w.writerow([0, 0, 1]); w.writerow([2, 1, 1]); w.writerow([2, 2, 0])
    records = ActivityDatasetAdapter(path).load()
    grid = ActivityDatasetAdapter.to_grid(records, 3, 3)
    assert isinstance(records[0], ActivityRecord)
    assert grid.shape == (3, 3) and grid.dtype == np.int8
    assert grid[0, 0] == 1 and grid[1, 2] == 1 and grid[2, 2] == 0

def test_activity_adapter_rejects_missing_columns(tmp_path: Path):
    path = tmp_path / "bad.csv"; path.write_text("band,time_slot\n0,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"): ActivityDatasetAdapter(path).load()

def test_activity_adapter_rejects_out_of_range_band():
    with pytest.raises(ValueError, match="outside configured range"):
        ActivityDatasetAdapter.validate([ActivityRecord(5, 0, 1)], n_bands=5, horizon=2)
