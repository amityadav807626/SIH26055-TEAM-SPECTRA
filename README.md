# SPECTRA — SMART-SCAN (SIH26055)

SPECTRA is a **software-only synthetic spectrum scanning simulation** demonstrating an online adaptive scheduler:

**scan → observe → update Beta posterior → sample Thompson policy → reprioritize**

## Safety / scope

This prototype uses only synthetic data. It has no RF hardware, transmitter, real emitter identification, or operational electronic-warfare procedure.

## V3.7.2 architecture

- `scheduler/smart_scheduler.py` — discounted Thompson Sampling with Beta-Bernoulli conjugate updates; learns online from every synthetic observation.
- `scheduler/round_robin.py` — deterministic baseline scheduler.
- `Backend/simulation_service.py` — shared browser/API simulation engine, baseline and agile-hopping environments, live telemetry and benchmark logic.
- `Backend/api/routes.py` — FastAPI endpoints for live runs, streaming progress, benchmarks and SQLite experiment persistence.
- `Backend/database.py` — canonical SQLite persistence layer.
- `Frontend/` — browser dashboard with live execution feedback and decision-latency KPI. The UI uses JetBrains Mono for readable data/number text and Orbitron only for display headings.
- `Backend/app.py` — legacy/reference Streamlit dashboard, updated to use the online scheduler.
- `Tests/` — API, database, frontend, scheduler and edge-case coverage.
- `evaluation/run_experiment.py` — JSON-producing CLI benchmark utility.

The offline Random Forest dependency has been removed. SPECTRA uses the online reward-based Smart Scheduler; there is no serialized model artifact in the runtime package.

## Dataset adapter boundary

The package includes `Backend/datasets/activity_adapter.py`, a neutral adapter for a
normalized CSV with exactly three columns:

```text
band,time_slot,activity
0,0,1
2,1,1
2,2,0
```

The adapter validates the records and produces the same `(time_slot, band)` activity-grid
representation used by the software simulator. It deliberately does not parse
source-specific radar/emitter fields. Any external source must first be transformed
into this neutral schema by a separately reviewed preprocessing step.

Experiment records store `dataset`, `dataset_version`, `data_mode`, and `data_source`
metadata in SQLite. Existing databases are migrated automatically when the application
starts.

## Online learning

Each band starts with a Beta(1, 1) prior. After an observation:

- detection/hit → discounted alpha update
- miss/no detection → discounted beta update
- old evidence is exponentially discounted so the scheduler can adapt to changing synthetic activity
- Thompson Sampling draws one probability sample per candidate band and combines it with freshness and exploration terms

This means the scheduler can learn during the current run instead of depending on a pre-trained artifact.

## Frequency-agile scenario

`create_environment(..., scenario="agile_hopping", hop_interval=N)` creates compact synthetic target-channel groups and pseudo-randomly moves the active group every `N` simulation slots.

The browser UI exposes **BASELINE / AGILE HOPPING** plus a configurable hop interval. This is an abstract channel-occupancy scenario only.

## Decision latency telemetry

`SmartScheduler.select_band()` measures decision time with `time.perf_counter()` in microseconds. Live API responses expose:

- `avg_decision_latency_us`
- `decision_latency_samples`

The browser header displays the running average as **Decision Latency: X.X µs (Dwell Valid)**. The label is a software simulation KPI, not a real receiver timing certification.

## Run the browser dashboard

From the project root:

```powershell
python -m uvicorn Backend.api.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/`.

API docs: `http://127.0.0.1:8000/docs`.

## Streamlit reference dashboard

```powershell
python -m streamlit run Backend\app.py
```

## Recommended demo settings

- Frequency bands: **20–30**
- Time slots: **260**
- Exploration rate: **0.10**
- Live seed: **0 = random**; non-zero values enable replay/debugging
- Scenario: **baseline** for the standard demo; **agile hopping** for adaptation stress testing
- Hop interval: **12** slots for the agile scenario
- Benchmark seed: **7**
- Benchmark trials: **10**

These are synthetic demonstration settings, not real RF operating parameters.

## Tests

```powershell
python -m pytest -q
```

Browser JavaScript syntax check:

```powershell
node --check Frontend\app.js
```

The edge-case suite covers:

1. recovery after synthetic target frequency hops,
2. single-receiver pulse-collision throughput, and
3. receiver blind-time degradation under agile hopping.

## Metric guardrails

- **System P_D** = true detections divided by all active synthetic band-time opportunities, including unscanned opportunities.
- **P_D_scanned / recall** = conditional detector recall on scanned observations.
- **Mean First-Intercept Delay** = simulation-slot delay from activity onset to the first successful observation.
- Benchmark tables report **mean ± sample SD** across controlled trials. SD communicates run-to-run variability; it is not a confidence interval and is not standalone proof of statistical significance.


## V3.7.1 final presentation polish
- Benchmark UI adds a transparent **FINAL VERDICT** computed from the six reported mean metrics.
- The current controlled benchmark shown by the dashboard resolves to **ROUND ROBIN** when it leads more reported mean metrics than Smart Scan.
- Streamlit selectboxes use the same dark tactical cockpit styling as the rest of the interface, including the Scenario control.


## V3.7.2 evaluation polish

The benchmark separates the project's primary adaptive objective from secondary detection metrics. It reports:

- **Useful detections / 100 scans** — true positive interceptions normalized to the fixed scan budget.
- **Average hop recovery time** — slots from a synthetic frequency hop to the first successful observation of the new target channel set.
- **Receiver blind time** — fraction of target-active slots where the selected channel is outside the active synthetic target set.

The browser benchmark supports both `baseline` and `agile_hopping` scenarios. In agile hopping, the final verdict is based on the primary adaptive metrics rather than treating every secondary metric as equally weighted. First-intercept delay, P_FA, precision, recall and F1 remain visible and are not hidden or relabeled.

## Data-source switch

The browser console supports three data modes from the **DATA SOURCE** selector:

- **RF Simulation Environment** — uses the existing SPECTRA synthetic activity generator.
- **External Dataset (normalized)** — loads `Backend/datasets/external_activity.csv` through the generic activity adapter.
- **TSRD-derived Activity** — selects one of six compact, label-derived activity proxies under `Backend/datasets/tsrd_activity/`.

The TSRD option does **not** bundle or parse raw HDF5/PDW fields. The included files contain only the neutral `band,time_slot,activity` schema. `band` is an abstract scheduler channel bucket and `time_slot` is an abstract record-order bin; neither is a physical-frequency mapping. The raw 12 HDF5 files remain external to the package.

Available TSRD-derived splits:

- `train_scan`, `val_scan`, `test_scan`
- `train_stare`, `val_stare`, `test_stare`

The supplied mapping uses the two selected source files for each split and preserves the train/validation/test organization in `Backend/datasets/tsrd_activity/manifest.json`.


## Jury CSV mode

SPECTRA includes the six bundled TSRD-derived normalized activity datasets under `Backend/datasets/tsrd_activity/`. The raw HDF5 files are not required to run the demo.

For a live jury demo, use **DATA → JURY CSV** in the browser and select the CSV supplied by the jury. The uploaded CSV is validated and stored locally as `Backend/datasets/jury_activity.csv`.

The accepted schema is exactly:

```csv
band,time_slot,activity
0,0,1
1,0,0
```

- `band`: non-negative integer channel index
- `time_slot`: non-negative integer
- `activity`: `0` or `1`

After upload, use **START SCAN** or **RUN BASELINE COMPARISON**. The selected jury dataset is used by the same Smart Scheduler and Round Robin benchmark path. No model retraining is required.

If the jury provides a CSV in another schema, it must first be converted to this normalized activity interface; SPECTRA does not guess source-specific column semantics.
