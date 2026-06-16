# Musik launcher — starts the desktop app (which auto-starts the engine sidecar).
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

# Prefer a built release exe (instant); otherwise run the dev build (compiles once).
$exe = Join-Path $root "app\src-tauri\target\release\Musik.exe"
if (Test-Path $exe) {
    Write-Host "Launching Musik..." -ForegroundColor Cyan
    Start-Process $exe
} else {
    Write-Host "Building & launching Musik (first run takes ~1-2 min)..." -ForegroundColor Cyan
    Set-Location (Join-Path $root "app")
    npm run tauri dev
}
