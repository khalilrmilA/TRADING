@echo off
rem ===========================================================================
rem  Trading AI Platform - one-time setup (PAPER TRADING ONLY)
rem  Creates the .venv virtual environment, installs dependencies, creates the
rem  .env file from the template and initialises the SQLite database.
rem ===========================================================================
setlocal
cd /d "%~dp0.."

echo.
echo ============================================================
echo  Trading AI Platform - setup  [PAPER TRADING ONLY]
echo ============================================================
echo.

rem --- 1. Locate the Python launcher -----------------------------------------
where py >nul 2>nul
if errorlevel 1 (
    echo [ERROR] The Python launcher "py" was not found on PATH.
    echo         Install Python 3.12+ from https://www.python.org/downloads/
    echo         and make sure "py launcher" is selected during install.
    exit /b 1
)

rem --- 2. Create the virtual environment if missing ---------------------------
if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Creating virtual environment in .venv ...
    py -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create the virtual environment.
        exit /b 1
    )
) else (
    echo [1/4] Virtual environment already exists - skipping creation.
)

rem --- 3. Install dependencies ------------------------------------------------
echo [2/4] Installing dependencies from requirements.txt ...
".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] pip install failed. Check your internet connection and retry.
    exit /b 1
)

rem --- 4. Create .env from the template if missing -----------------------------
if not exist ".env" (
    echo [3/4] Creating .env from .env.example ...
    copy ".env.example" ".env" >nul
    if errorlevel 1 (
        echo [ERROR] Could not create .env from .env.example.
        exit /b 1
    )
) else (
    echo [3/4] .env already exists - leaving it untouched.
)

rem --- 5. Initialise the database ----------------------------------------------
echo [4/4] Initialising the SQLite database ...
".venv\Scripts\python.exe" -c "from backend.database.db import init_db; init_db(); print('Database ready.')"
if errorlevel 1 (
    echo [ERROR] Database initialisation failed.
    exit /b 1
)

echo.
echo ============================================================
echo  Setup complete.
echo    Start everything:   scripts\start_all.bat
echo    Backend only:       scripts\start_backend.bat
echo    Dashboard only:     scripts\start_dashboard.bat
echo.
echo  Reminder: pull an Ollama model first, e.g.
echo    ollama pull qwen3:14b
echo ============================================================
echo.
exit /b 0
