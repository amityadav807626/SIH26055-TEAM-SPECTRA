"""SPECTRA local FastAPI application and browser frontend host."""
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from Backend.database import init_db
from Backend.api.routes import router

PROJECT_ROOT=Path(__file__).resolve().parents[2]
FRONTEND=PROJECT_ROOT/"Frontend"
init_db()
app=FastAPI(title="SPECTRA / SMART-SCAN API",version="3.6.0",description="REST API for software-simulation experiment results.")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_credentials=False,allow_methods=["*"],allow_headers=["*"])
app.include_router(router)
if FRONTEND.exists(): app.mount("/",StaticFiles(directory=FRONTEND,html=True),name="frontend")
