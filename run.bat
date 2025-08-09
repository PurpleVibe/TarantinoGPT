@echo off
setlocal
cd /d "%~dp0"

REM Ensure venv Python exists
if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv not found at .venv\Scripts\python.exe
  echo Create it first:
  echo   python -m venv .venv
  echo   .venv\Scripts\pip install -r requirements.txt
  pause
  exit /b 1
)

REM Start FastAPI backend in a new window
start "Tarantino RAG API" ".venv\Scripts\python.exe" -m uvicorn backend.src.api:app --host 127.0.0.1 --port 8000

REM Give backend a moment to boot
timeout /t 2 /nobreak >nul

REM Start Streamlit UI in a new window
start "Tarantino RAG UI" ".venv\Scripts\python.exe" -m streamlit run frontend\app.py

echo.
echo Started:
echo  - API: http://127.0.0.1:8000/docs

echo  - UI : http://localhost:8501
endlocal