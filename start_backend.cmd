@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=python"
if exist "E:\anaconda\python.exe" (
    set "PYTHON_EXE=E:\anaconda\python.exe"
)

echo Starting FastAPI backend on http://127.0.0.1:8000
"%PYTHON_EXE%" "%~dp0run_backend.py"

echo.
echo Backend exited. Press any key to close this window.
pause >nul
