"""SQLite persistence for SPECTRA software-simulation experiments."""
from __future__ import annotations
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "outputs" / "smart_scan.db"


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _columns(conn, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def init_db() -> None:
    with get_connection() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, scheduler TEXT NOT NULL,
            bands INTEGER NOT NULL, time_slots INTEGER NOT NULL, seed INTEGER DEFAULT NULL,
            hits INTEGER NOT NULL DEFAULT 0, scans INTEGER NOT NULL DEFAULT 0, coverage REAL DEFAULT 0,
            accuracy REAL DEFAULT 0, p_d REAL DEFAULT 0, p_fa REAL DEFAULT 0, precision REAL DEFAULT 0,
            recall REAL DEFAULT 0, f1 REAL DEFAULT 0, tp INTEGER DEFAULT 0, fp INTEGER DEFAULT 0,
            fn INTEGER DEFAULT 0, tn INTEGER DEFAULT 0, avg_intercept_time REAL,
            dataset TEXT NOT NULL DEFAULT 'SPECTRA Synthetic Activity',
            dataset_version TEXT NOT NULL DEFAULT 'internal-v1',
            data_mode TEXT NOT NULL DEFAULT 'simulation',
            data_source TEXT)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS band_learning (
            id INTEGER PRIMARY KEY AUTOINCREMENT, experiment_id INTEGER NOT NULL, band INTEGER NOT NULL,
            visits INTEGER NOT NULL DEFAULT 0, hits INTEGER NOT NULL DEFAULT 0, reward REAL NOT NULL DEFAULT 0,
            FOREIGN KEY(experiment_id) REFERENCES experiments(id) ON DELETE CASCADE,
            UNIQUE(experiment_id, band))""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_band_learning_experiment ON band_learning(experiment_id)")
        conn.execute("""CREATE TABLE IF NOT EXISTS benchmark_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, bands INTEGER NOT NULL,
            time_slots INTEGER NOT NULL, epsilon REAL NOT NULL, seed INTEGER NOT NULL, trials INTEGER NOT NULL,
            payload_json TEXT NOT NULL)""")
        old = _columns(conn, "experiments")
        additions = {
            "scans":"INTEGER NOT NULL DEFAULT 0", "coverage":"REAL DEFAULT 0", "accuracy":"REAL DEFAULT 0",
            "tp":"INTEGER DEFAULT 0", "fp":"INTEGER DEFAULT 0", "fn":"INTEGER DEFAULT 0", "tn":"INTEGER DEFAULT 0",
            "seed":"INTEGER DEFAULT NULL",
            "dataset":"TEXT NOT NULL DEFAULT 'SPECTRA Synthetic Activity'",
            "dataset_version":"TEXT NOT NULL DEFAULT 'internal-v1'",
            "data_mode":"TEXT NOT NULL DEFAULT 'simulation'",
            "data_source":"TEXT",
        }
        for name, definition in additions.items():
            if name not in old:
                conn.execute(f"ALTER TABLE experiments ADD COLUMN {name} {definition}")


def _v(result: dict[str, Any], *keys: str, default: Any = 0.0) -> Any:
    for key in keys:
        if result.get(key) is not None:
            return result[key]
    return default


def save_experiment(
    scheduler: str,
    bands: int,
    time_slots: int,
    result: dict[str, Any],
    seed: int | None = None,
    dataset: str = "SPECTRA Synthetic Activity",
    dataset_version: str = "internal-v1",
    data_mode: str = "simulation",
    data_source: str | None = None,
) -> int:
    with get_connection() as conn:
        cur = conn.execute("""INSERT INTO experiments
            (created_at,scheduler,bands,time_slots,seed,hits,scans,coverage,accuracy,p_d,p_fa,precision,recall,f1,tp,fp,fn,tn,avg_intercept_time,
             dataset,dataset_version,data_mode,data_source)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
            datetime.now(timezone.utc).isoformat(timespec="seconds"), scheduler, int(bands), int(time_slots),
            None if seed is None else int(seed), int(_v(result,"hits","Detections",default=0)),
            int(_v(result,"scans","Scans",default=time_slots)), float(_v(result,"coverage","Coverage",default=0)),
            float(_v(result,"accuracy","Accuracy",default=0)), float(_v(result,"P_D","pd","recall",default=0)),
            float(_v(result,"P_FA","p_fa",default=0)), float(_v(result,"precision",default=0)),
            float(_v(result,"recall",default=0)), float(_v(result,"F1","f1",default=0)),
            int(_v(result,"tp","TP",default=0)), int(_v(result,"fp","FP",default=0)),
            int(_v(result,"fn","FN",default=0)), int(_v(result,"tn","TN",default=0)),
            _v(result,"avg_intercept_time","avg_delay",default=None),
            dataset, dataset_version, data_mode, data_source))
        eid = int(cur.lastrowid)
        rewards, visits, hits = list(result.get("final_rewards",[])), list(result.get("visits",[])), list(result.get("hits_per_band",[]))
        for band in range(max(len(rewards),len(visits),len(hits))):
            conn.execute("INSERT INTO band_learning(experiment_id,band,visits,hits,reward) VALUES(?,?,?,?,?)", (
                eid, band, int(visits[band]) if band<len(visits) else 0,
                int(hits[band]) if band<len(hits) else 0, float(rewards[band]) if band<len(rewards) else 0))
        return eid


def get_experiments() -> list[tuple]:
    with get_connection() as conn:
        rows = conn.execute("""SELECT id,created_at,scheduler,bands,time_slots,seed,hits,p_d,p_fa,precision,recall,f1,
            avg_intercept_time,scans,coverage,accuracy,tp,fp,fn,tn FROM experiments ORDER BY id DESC""").fetchall()
        return [tuple(r) for r in rows]


def get_experiment(experiment_id: int) -> dict[str, Any] | None:
    with get_connection() as conn:
        r=conn.execute("SELECT * FROM experiments WHERE id=?",(int(experiment_id),)).fetchone()
        return dict(r) if r else None


def get_band_learning(experiment_id: int) -> list[tuple]:
    with get_connection() as conn:
        return [tuple(r) for r in conn.execute("SELECT band,visits,hits,reward FROM band_learning WHERE experiment_id=? ORDER BY reward DESC,band ASC",(int(experiment_id),)).fetchall()]


def save_benchmark(payload: dict[str, Any], bands: int, time_slots: int, epsilon: float, seed: int, trials: int) -> int:
    with get_connection() as conn:
        cur=conn.execute("INSERT INTO benchmark_runs(created_at,bands,time_slots,epsilon,seed,trials,payload_json) VALUES(?,?,?,?,?,?,?)",(
            datetime.now(timezone.utc).isoformat(timespec="seconds"),bands,time_slots,epsilon,seed,trials,json.dumps(payload)))
        return int(cur.lastrowid)


def get_latest_benchmark() -> dict[str, Any] | None:
    with get_connection() as conn:
        r=conn.execute("SELECT * FROM benchmark_runs ORDER BY id DESC LIMIT 1").fetchone()
        if not r:return None
        d=dict(r); d["payload"]=json.loads(d.pop("payload_json")); return d


def clear_database() -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM band_learning"); conn.execute("DELETE FROM experiments"); conn.execute("DELETE FROM benchmark_runs")


init_db()
