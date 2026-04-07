Set-Location $PSScriptRoot

$pythonExe = "python"
if (Test-Path "E:\anaconda\python.exe") {
    $pythonExe = "E:\anaconda\python.exe"
}

Write-Host "Starting FastAPI backend on http://127.0.0.1:8000"
& $pythonExe "$PSScriptRoot\run_backend.py"
