@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=python"
if exist "E:\anaconda\python.exe" (
    set "PYTHON_EXE=E:\anaconda\python.exe"
)

echo Starting EEG simulator. Frontend user_id default is 99.
"%PYTHON_EXE%" "%~dp0eeg_simulator.py" --backend-url http://127.0.0.1:8000/api/eeg/upload --user-id 99 --sample-rate 250 --batch-ms 100 --speed 1

echo.
echo Simulator exited. Press any key to close this window.
pause >nul
