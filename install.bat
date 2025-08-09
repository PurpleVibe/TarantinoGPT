@echo off
setlocal
cd /d "%~dp0"

REM ----- 1) Detect/prepare Python -----
where py >nul 2>nul
if %ERRORLEVEL%==0 (
  set PYEXE=py -3.12
) else (
  where python >nul 2>nul && (set PYEXE=python) || (
    echo [ERROR] Python not found in PATH. Install Python 3.12+ and re-run.
    pause
    exit /b 1
  )
)

REM ----- 2) Create venv if missing -----
if not exist ".venv\Scripts\python.exe" (
  echo [INFO] Creating virtual environment .venv
  %PYEXE% -m venv .venv
  if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
  )
)

REM ----- 3) Upgrade pip & install requirements -----
".venv\Scripts\python.exe" -m pip install --upgrade pip
if exist requirements.txt (
  echo [INFO] Installing requirements
  ".venv\Scripts\pip.exe" install -r requirements.txt
) else (
  echo [WARN] requirements.txt not found. Skipping dependency install.
)


REM ----- 4) Create .env template if missing -----
if not exist .env (
  echo [INFO] Creating .env template
  > .env echo OPENAI_API_KEY=your_openai_api_key_here
  >> .env echo # Optionally override API URLs for frontend
  >> .env echo # API_URL=http://127.0.0.1:8000/api/query
  >> .env echo # API_URL_STREAM=http://127.0.0.1:8000/api/query_stream
)

REM ----- 5) Quick sanity checks -----
".venv\Scripts\python.exe" -c "import fastapi,uvicorn,streamlit,langchain_chroma; print('Sanity OK')" 1>nul 2>nul
if %ERRORLEVEL% neq 0 (
  echo [WARN] Some optional imports failed. Verify requirements were installed successfully.
)

echo.
echo [DONE] Setup complete.
echo - Activate venv:   .venv\Scripts\activate
echo - Run both apps:   run.bat


endlocal
