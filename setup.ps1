# Musik — one-time setup.
# Right-click > "Run with PowerShell", or:  powershell -ExecutionPolicy Bypass -File setup.ps1
# Installs the Python engine (with CUDA PyTorch + music models) and the app, then
# puts a "Musik" shortcut on your desktop. Run once; after that just use the shortcut.

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root
Write-Host "== Musik setup ==" -ForegroundColor Cyan

# 1) Python 3.13 venv
if (-not (Test-Path "engine\.venv\Scripts\python.exe")) {
    Write-Host "[1/5] Creating Python 3.13 virtual environment..." -ForegroundColor Yellow
    py -3.13 -m venv engine\.venv
} else {
    Write-Host "[1/5] Python venv already exists." -ForegroundColor Green
}
$py = Join-Path $root "engine\.venv\Scripts\python.exe"

# 2) Engine + local API server
Write-Host "[2/5] Installing engine + server..." -ForegroundColor Yellow
& $py -m pip install -q -U pip
& $py -m pip install -q -e "engine[server]"

# 3) CUDA PyTorch (NVIDIA, e.g. RTX 5080) + music model backends
Write-Host "[3/5] Installing CUDA PyTorch (cu128) + music models (large download, be patient)..." -ForegroundColor Yellow
& $py -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
& $py -m pip install -q -e "engine[models]"
$cuda = & $py -c "import torch; print(torch.cuda.is_available())"
Write-Host "    PyTorch CUDA available: $cuda" -ForegroundColor Green

# 4) App (frontend) dependencies
Write-Host "[4/5] Installing app dependencies (npm)..." -ForegroundColor Yellow
Push-Location app
npm install
Pop-Location

# 5) Desktop shortcut
Write-Host "[5/5] Creating desktop shortcut..." -ForegroundColor Yellow
& (Join-Path $root "scripts\New-MusikShortcut.ps1")

Write-Host "`nDone. Launch Musik from the desktop shortcut (or run Musik.bat)." -ForegroundColor Green
Write-Host "First launch compiles the app once (~1-2 min); after that it's instant." -ForegroundColor Green
