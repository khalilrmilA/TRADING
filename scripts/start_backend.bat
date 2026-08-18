@echo off
rem ===========================================================================
rem  Trading AI Platform - start the FastAPI backend (PAPER TRADING ONLY)
rem  API: http://127.0.0.1:8000   docs: http://127.0.0.1:8000/docs
rem ===========================================================================
setlocal
cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found.
    echo         Run scripts\setup.bat first.
    pause
    exit /b 1
)

echo Starting backend API on http://127.0.0.1:8000  [PAPER TRADING ONLY]
".venv\Scripts\python.exe" -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000
exit /b %errorlevel%
