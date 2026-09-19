@echo off
cd /d "%~dp0"
python -m uvicorn Backend.api.main:app --host 127.0.0.1 --port 8000
