"""REST API for the SPECTRA simulation dashboard."""
from __future__ import annotations
from typing import Any
import json
import queue
import threading

from fastapi.responses import StreamingResponse
import secrets
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from Backend.database import clear_database, get_band_learning, get_experiment, get_experiments, save_benchmark, save_experiment, get_latest_benchmark
from Backend.simulation_service import run_live, run_multi_benchmark
from Backend.datasets.activity_adapter import ActivityDatasetAdapter, detect_jury_columns, normalize_jury_csv
from Backend.datasets.tsrd_activity import load_tsrd_activity, TSRD_SPLITS
from pathlib import Path

router = APIRouter(prefix="/api", tags=["SPECTRA"])

class SimulationConfig(BaseModel):
    data_mode: str = Field("simulation", pattern="^(simulation|normalized-external|tsrd-derived|jury-csv)$")
    bands: int = Field(30, ge=10, le=60)
    time_slots: int = Field(260, ge=100, le=260)
    epsilon: float = Field(.10, ge=0, le=.40)
    seed: int = Field(0, ge=0, le=999999999)
    scenario: str = Field("baseline", pattern="^(baseline|agile_hopping)$")
    hop_interval: int = Field(12, ge=2, le=100)
    tsrd_split: str = Field("test_scan", pattern="^(train_scan|val_scan|test_scan|train_stare|val_stare|test_stare)$")

class BenchmarkConfig(SimulationConfig):
    trials: int = Field(10, ge=3, le=30)

class ExperimentCreate(BaseModel):
    scheduler: str = Field(min_length=1, max_length=80)
    bands: int = Field(ge=1, le=1000)
    time_slots: int = Field(ge=1, le=1_000_000)
    seed: int | None = None
    result: dict[str, Any]
    dataset: str = Field("SPECTRA Synthetic Activity", min_length=1, max_length=120)
    dataset_version: str = Field("internal-v1", min_length=1, max_length=60)
    data_mode: str = Field("simulation", pattern="^(simulation|normalized-external|tsrd-derived|jury-csv)$")
    data_source: str | None = Field(None, max_length=500)

KEYS=["id","created_at","scheduler","bands","time_slots","seed","hits","p_d","p_fa","precision","recall","f1","avg_intercept_time","scans","coverage","accuracy","tp","fp","fn","tn","dataset","dataset_version","data_mode","data_source"]

def exp_from_tuple(row): return dict(zip(KEYS,row))

def detail(eid:int):
    row=get_experiment(eid)
    if row is None: raise HTTPException(404,"Experiment not found")
    row["band_learning"]=[{"band":b,"visits":v,"hits":h,"reward":r} for b,v,h,r in get_band_learning(eid)]
    return row

@router.get("/health")
def health(): return {"status":"ok","service":"SPECTRA API","mode":"simulation-only"}

@router.get("/experiments")
def experiments(): return [exp_from_tuple(r) for r in get_experiments()]

@router.get("/experiments/{experiment_id}")
def experiment(experiment_id:int): return detail(experiment_id)

@router.get("/experiments/{experiment_id}/bands")
def experiment_bands(experiment_id:int): return detail(experiment_id)["band_learning"]

@router.post("/experiments", status_code=201)
def create_experiment(payload:ExperimentCreate):
    return detail(save_experiment(payload.scheduler,payload.bands,payload.time_slots,payload.result,payload.seed,payload.dataset,payload.dataset_version,payload.data_mode,payload.data_source))

EXTERNAL_DATASET = Path(__file__).resolve().parents[1] / "datasets" / "external_activity.csv"
JURY_DATASET = Path(__file__).resolve().parents[1] / "datasets" / "jury_activity.csv"

@router.post("/datasets/jury-csv")
def upload_jury_csv(
    file: UploadFile = File(...),
    band_col: str | None = Form(None),
    time_col: str | None = Form(None),
    activity_col: str | None = Form(None),
):
    """Accept a jury CSV and normalize generic activity columns.

    If column names are recognizable, mapping is automatic. Otherwise the API
    returns candidate columns so the browser can ask the user to map them.
    Source-specific radar/PDW fields are not interpreted.
    """
    if not file.filename or not file.filename.lower().endswith('.csv'):
        raise HTTPException(400, "Please upload a CSV file")
    temp = JURY_DATASET.with_suffix('.upload.csv')
    try:
        raw = file.file.read()
        temp.parent.mkdir(parents=True, exist_ok=True)
        temp.write_bytes(raw)
        import csv as _csv
        with temp.open("r", newline="", encoding="utf-8-sig") as fh:
            reader = _csv.DictReader(fh)
            fields = reader.fieldnames or []
        if not fields:
            raise ValueError("CSV has no header")
        detected = detect_jury_columns(fields)
        bc = band_col or (detected["band"][0] if len(detected["band"]) == 1 else None)
        tc = time_col or (detected["time_slot"][0] if len(detected["time_slot"]) == 1 else None)
        ac = activity_col or (detected["activity"][0] if len(detected["activity"]) == 1 else None)
        if not all((bc, tc, ac)):
            return {"status": "mapping_required", "filename": file.filename, "columns": fields, "detected": detected, "message": "Please map band, time and activity columns."}
        result = normalize_jury_csv(temp, JURY_DATASET, bc, tc, ac)
        result.update({"status": "ok", "filename": file.filename, "mapping": {"band": bc, "time_slot": tc, "activity": ac}, "path": "Backend/datasets/jury_activity.csv"})
        return result
    except Exception as exc:
        raise HTTPException(400, str(exc))
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        close = getattr(file.file, 'close', None)
        if close: close()


def load_external_activity(n_bands: int, horizon: int):
    records = ActivityDatasetAdapter(EXTERNAL_DATASET).load()
    records = [r for r in records if r.band < n_bands and r.time_slot < horizon]
    if not records:
        raise ValueError("external dataset has no records inside the configured band/time range")
    return ActivityDatasetAdapter.to_grid(records, n_bands=n_bands, horizon=horizon)

def load_selected_activity(config: SimulationConfig):
    if config.data_mode == "normalized-external":
        return load_external_activity(config.bands, config.time_slots)
    if config.data_mode == "tsrd-derived":
        return load_tsrd_activity(config.bands, config.time_slots, config.tsrd_split)
    if config.data_mode == "jury-csv":
        if not JURY_DATASET.exists():
            raise ValueError("No jury CSV has been uploaded yet")
        records = ActivityDatasetAdapter(JURY_DATASET).load()
        records = [r for r in records if r.band < config.bands and r.time_slot < config.time_slots]
        if not records:
            raise ValueError("jury CSV has no records inside the configured band/time range")
        return ActivityDatasetAdapter.to_grid(records, n_bands=config.bands, horizon=config.time_slots)
    return None


@router.post("/simulation/live/stream")
def simulation_live_stream(config: SimulationConfig):
    """Stream live synthetic scan progress as NDJSON, then persist the completed run."""
    seed = secrets.randbelow(1_000_000_000) if config.seed == 0 else config.seed
    q: queue.Queue = queue.Queue()

    def worker() -> None:
        try:
            def emit(item: dict[str, Any]) -> None:
                q.put(item)
            activity_grid = load_selected_activity(config)
            result = run_live(config.bands, config.time_slots, config.epsilon, seed, progress_callback=emit, scenario=config.scenario, hop_interval=config.hop_interval, activity_grid=activity_grid, source_mode=config.data_mode, source_name=("External Normalized Activity" if config.data_mode == "normalized-external" else "TSRD-Derived Activity Proxy" if config.data_mode == "tsrd-derived" else "Jury Normalized Activity" if config.data_mode == "jury-csv" else "SPECTRA Synthetic Activity"), source_version=("external-demo-v1" if config.data_mode == "normalized-external" else f"tsrd-proxy-v1:{config.tsrd_split}" if config.data_mode == "tsrd-derived" else "jury-csv-v1" if config.data_mode == "jury-csv" else "internal-v1"), source_detail=("Backend/datasets/external_activity.csv" if config.data_mode == "normalized-external" else f"Backend/datasets/tsrd_activity/{config.tsrd_split}.csv" if config.data_mode == "tsrd-derived" else "Backend/datasets/jury_activity.csv" if config.data_mode == "jury-csv" else "SPECTRA synthetic environment"))
            result["run_type"] = "live-randomized" if config.seed == 0 else "live-seeded"
            result["data_mode"] = config.data_mode
            result["tsrd_split"] = config.tsrd_split if config.data_mode == "tsrd-derived" else None
            result["experiment_id"] = save_experiment("Live Smart Scan", config.bands, config.time_slots, result, seed=seed, dataset=result["dataset"], dataset_version=result["dataset_version"], data_mode=result["data_mode"], data_source=result["data_source"])
            q.put({"type": "complete", "result": result})
        except Exception as exc:  # surfaced to the browser without hiding failures
            q.put({"type": "error", "message": str(exc)})

    threading.Thread(target=worker, daemon=True).start()

    def stream():
        while True:
            item = q.get()
            yield json.dumps(item, separators=(",", ":")) + "\n"
            if item.get("type") in {"complete", "error"}:
                break

    return StreamingResponse(stream(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/simulation/live")
def simulation_live(config:SimulationConfig):
    # Live mode is randomized by default. A non-zero seed intentionally enables replay/debugging.
    seed = secrets.randbelow(1_000_000_000) if config.seed == 0 else config.seed
    activity_grid = load_selected_activity(config)
    result=run_live(config.bands,config.time_slots,config.epsilon,seed,scenario=config.scenario,hop_interval=config.hop_interval,activity_grid=activity_grid,source_mode=config.data_mode,source_name=("External Normalized Activity" if config.data_mode == "normalized-external" else "TSRD-Derived Activity Proxy" if config.data_mode == "tsrd-derived" else "Jury Normalized Activity" if config.data_mode == "jury-csv" else "SPECTRA Synthetic Activity"),source_version=("external-demo-v1" if config.data_mode == "normalized-external" else f"tsrd-proxy-v1:{config.tsrd_split}" if config.data_mode == "tsrd-derived" else "jury-csv-v1" if config.data_mode == "jury-csv" else "internal-v1"),source_detail=("Backend/datasets/external_activity.csv" if config.data_mode == "normalized-external" else f"Backend/datasets/tsrd_activity/{config.tsrd_split}.csv" if config.data_mode == "tsrd-derived" else "Backend/datasets/jury_activity.csv" if config.data_mode == "jury-csv" else "SPECTRA synthetic environment"))
    result["run_type"]="live-randomized" if config.seed == 0 else "live-seeded"
    result["data_mode"] = config.data_mode
    result["tsrd_split"] = config.tsrd_split if config.data_mode == "tsrd-derived" else None
    result["experiment_id"] = save_experiment("Live Smart Scan",config.bands,config.time_slots,result,seed=seed,dataset=result["dataset"],dataset_version=result["dataset_version"],data_mode=result["data_mode"],data_source=result["data_source"])
    return result

@router.post("/simulation/benchmark")
def simulation_benchmark(config:BenchmarkConfig):
    # Benchmark mode is deliberately reproducible; 0 selects the documented default seed.
    seed = config.seed if config.seed != 0 else 7
    activity_grid = None
    try:
        activity_grid = load_selected_activity(config)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc))
    benchmark_scenario = config.scenario if config.data_mode == "simulation" else "baseline"
    result=run_multi_benchmark(config.bands,config.time_slots,config.epsilon,seed,config.trials,scenario=benchmark_scenario,hop_interval=config.hop_interval,activity_grid=activity_grid)
    result["data_mode"] = config.data_mode
    result["dataset"] = ("External Normalized Activity" if config.data_mode == "normalized-external" else "TSRD-Derived Activity Proxy" if config.data_mode == "tsrd-derived" else "Jury Normalized Activity" if config.data_mode == "jury-csv" else "SPECTRA Synthetic Activity")
    result["dataset_version"] = ("external-demo-v1" if config.data_mode == "normalized-external" else f"tsrd-proxy-v1:{config.tsrd_split}" if config.data_mode == "tsrd-derived" else "jury-csv-v1" if config.data_mode == "jury-csv" else "internal-v1")
    result["data_source"] = ("Backend/datasets/external_activity.csv" if config.data_mode == "normalized-external" else f"Backend/datasets/tsrd_activity/{config.tsrd_split}.csv" if config.data_mode == "tsrd-derived" else "Backend/datasets/jury_activity.csv" if config.data_mode == "jury-csv" else "generated simulation environment")
    result["tsrd_split"] = config.tsrd_split if config.data_mode == "tsrd-derived" else None
    result["run_id"]=save_benchmark(result,config.bands,config.time_slots,config.epsilon,seed,config.trials)
    return result

@router.get("/benchmark/latest")
def benchmark_latest():
    item=get_latest_benchmark()
    if item is None: raise HTTPException(404,"No benchmark has been run yet")
    return item

@router.delete("/experiments", status_code=204)
def delete_experiments(): clear_database()
