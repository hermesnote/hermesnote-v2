@echo off
setlocal
set "MONITOR=%~dp0nas-training-monitor.ps1"
if not exist "%MONITOR%" (
  echo Missing nas-training-monitor.ps1. Keep both files in the same folder.
  pause
  exit /b 1
)
start "NAS Hardware" powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%MONITOR%" -Mode Hardware
start "Training Epochs" powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%MONITOR%" -Mode Progress
exit /b
