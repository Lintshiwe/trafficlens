$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Host "Project virtual environment not found. See README.md for setup." -ForegroundColor Yellow
    exit 1
}

Set-Location $root
& $python run.py @args
