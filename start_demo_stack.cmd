@echo off
cd /d "%~dp0"

start "EEG Backend" cmd /k "%~dp0start_backend.cmd"
timeout /t 4 /nobreak >nul
start "EEG Simulator" cmd /k "%~dp0start_simulator.cmd"

echo Demo stack started.
echo Frontend: http://127.0.0.1:8000
echo Realtime mode + User ID = 99
