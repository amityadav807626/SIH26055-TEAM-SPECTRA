# SPECTRA Browser Frontend

The browser dashboard now mirrors the Streamlit demo: Live Radar, Performance, Benchmark and Experiment Log, backed by FastAPI + SQLite.

## Run
From the project root:

```powershell
python -m uvicorn Backend.api.main:app --reload
```

Open `http://127.0.0.1:8000/`.

The API docs are at `http://127.0.0.1:8000/docs`.

Everything is software-only and uses the synthetic spectrum environment.
