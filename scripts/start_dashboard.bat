@echo off
rem ===========================================================================
rem  Trading AI Platform - start the Streamlit dashboard (PAPER TRADING ONLY)
rem  Dashboard: http://localhost:8501  (requires the backend on port 8000)
rem ===========================================================================
setlocal
cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found.
    echo         Run scripts\setup.bat first.
    pause
    exit /b 1
)

echo Starting dashboard on http://localhost:8501  [PAPER TRADING ONLY]
".venv\Scripts\python.exe" -m streamlit run dashboard/app.py
exit /b %errorlevel%
