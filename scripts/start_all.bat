@echo off
rem ===========================================================================
rem  Trading AI Platform - start Ollama (if needed), backend and dashboard
rem  (PAPER TRADING ONLY)
rem
rem  Opens the backend and the dashboard in two separate console windows so
rem  each has its own logs. Checks whether Ollama is answering on port 11434
rem  first and launches the Ollama desktop app if it is not.
rem ===========================================================================
setlocal
cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found.
    echo         Run scripts\setup.bat first.
    pause
    exit /b 1
)

rem --- 1. Make sure Ollama is up (AI analysis needs it; rest works without) --
echo [1/3] Checking Ollama on http://localhost:11434 ...
curl.exe -s -m 3 -o NUL "http://localhost:11434/api/version"
if errorlevel 1 (
    echo        Ollama is not responding - trying to start it ...
    if exist "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe" (
        start "" "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe"
        echo        Waiting for Ollama to come up ...
        timeout /t 8 /nobreak >nul
    ) else (
        echo [WARN] Ollama app not found at "%LOCALAPPDATA%\Programs\Ollama".
        echo        Install it from https://ollama.com - AI analysis will be
        echo        unavailable until Ollama is running.
    )
) else (
    echo        Ollama is running.
)

rem --- 2. Backend --------------------------------------------------------------
echo [2/3] Starting backend API window  http://127.0.0.1:8000 ...
start "Trading AI - Backend (paper only)" cmd /k call "%~dp0start_backend.bat"

rem Give the API a moment to bind before the dashboard polls it.
timeout /t 3 /nobreak >nul

rem --- 3. Dashboard ------------------------------------------------------------
echo [3/3] Starting dashboard window    http://localhost:8501 ...
start "Trading AI - Dashboard (paper only)" cmd /k call "%~dp0start_dashboard.bat"

echo.
echo All components launched. Close the spawned windows to stop them.
echo   Backend:   http://127.0.0.1:8000  (docs at /docs)
echo   Dashboard: http://localhost:8501
echo.
exit /b 0
