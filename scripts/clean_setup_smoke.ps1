param(
    [string]$PythonCommand = "",
    [string]$VenvPath = ".smoke-flyvis-venv",
    [string]$ExistingPython = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot
$env:PIP_CONFIG_FILE = Join-Path $RepoRoot ".pip-flyvis.ini"

if ($ExistingPython) {
    $PythonExe = (Resolve-Path $ExistingPython).Path
} else {
    $RequestedVenv = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $VenvPath))
    if (-not $RequestedVenv.StartsWith($RepoRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "VenvPath must stay inside the repository: $RequestedVenv"
    }
    if (Test-Path -LiteralPath $RequestedVenv) {
        throw "Smoke-test environment already exists: $RequestedVenv"
    }
    if ($PythonCommand) {
        & $PythonCommand -m venv $RequestedVenv
    } else {
        & py -3.11 -m venv $RequestedVenv
    }
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
    $PythonExe = Join-Path $RequestedVenv "Scripts\python.exe"
    $Version = & $PythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($Version -ne "3.11") { throw "Python 3.11 required, got $Version" }
    & $PythonExe -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
    & $PythonExe -m pip install -r requirements-flyvis.txt
    if ($LASTEXITCODE -ne 0) { throw "dependency installation failed" }
}

& $PythonExe scripts\patch_flyvis_windows.py
if ($LASTEXITCODE -ne 0) { throw "FlyVis Windows patch failed" }
& $PythonExe scripts\patch_flyvis_windows.py --check
if ($LASTEXITCODE -ne 0) { throw "FlyVis Windows patch verification failed" }
& $PythonExe scripts\setup_flyvis_data.py --ensure
if ($LASTEXITCODE -ne 0) { throw "pretrained data setup failed" }
& $PythonExe -m src.cli verify
if ($LASTEXITCODE -ne 0) { throw "project regression verification failed" }
& $PythonExe -m src.experiments.phase15_flyvis_motion
if ($LASTEXITCODE -ne 0) { throw "FlyVis inference smoke test failed" }

Write-Host "CLEAN SETUP SMOKE TEST PASSED" -ForegroundColor Green
