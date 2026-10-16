# Runs the Gesture Slicer game using the first Python that has pygame.
# Usage:
#   .\run_game.ps1
#   .\run_game.ps1 -Python "C:\path\to\venv\Scripts\python.exe"

param(
    [string]$Python = $null
)

$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot

function Test-Pygame([string]$exe) {
    if (-not (Test-Path $exe)) { return $false }
    & $exe -c "import pygame" 2>$null
    return ($LASTEXITCODE -eq 0)
}

$candidates = @()

if ($Python) {
    $candidates += $Python
} else {
    $knownVenv = Join-Path $env:USERPROFILE "gesture_control_app\.venv\Scripts\python.exe"
    $localVenv = Join-Path (Split-Path $PSScriptRoot -Parent) ".venv\Scripts\python.exe"
    $candidates += $knownVenv, $localVenv, "python"
}

foreach ($exe in $candidates) {
    if ($PSVersionTable.PSVersion.Major -ge 6) {
        $exeName = $exe
    } else {
        $exeName = $exe
    }
    if (Test-Pygame $exe) {
        Write-Host "[run_game] Using $exe"
        & $exe game.py @args
        Pop-Location
        exit $LASTEXITCODE
    }
}

Write-Host ""
Write-Host "[run_game] ERROR: pygame is not installed in any searched interpreter." -ForegroundColor Red
Write-Host ""
Write-Host "  Fix:  python -m pip install -r requirements.txt"
Write-Host "  Or:   .\run_game.ps1 -Python 'C:\path\to\venv\Scripts\python.exe'"
Write-Host ""
Pop-Location
exit 1