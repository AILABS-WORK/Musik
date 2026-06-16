# Creates a "Musik" shortcut on the desktop pointing at Musik.bat.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot          # scripts\ -> repo root
$target = Join-Path $root "Musik.bat"
$icon = Join-Path $root "app\src-tauri\icons\icon.ico"
$desktop = [Environment]::GetFolderPath("Desktop")
$lnk = Join-Path $desktop "Musik.lnk"

$shell = New-Object -ComObject WScript.Shell
$sc = $shell.CreateShortcut($lnk)
$sc.TargetPath = $target
$sc.WorkingDirectory = $root
$sc.Description = "Musik - Music Hub"
$sc.WindowStyle = 7   # minimized console
if (Test-Path $icon) { $sc.IconLocation = $icon }
$sc.Save()
Write-Host "Desktop shortcut created: $lnk"
