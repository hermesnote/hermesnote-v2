@echo off
setlocal
title HermesNote V2 Deploy

set "DEPLOY_SCRIPT=%~dp0..\deploy.ps1"
if not exist "%DEPLOY_SCRIPT%" (
    echo ERROR: deploy.ps1 was not found in the project root.
    pause
    exit /b 1
)

echo Starting HermesNote V2 deployment...
echo SMB credentials will be requested by deploy.ps1 if needed.
echo Database migrations must be applied separately before deploying code that requires them.
echo.

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%DEPLOY_SCRIPT%"
set "DEPLOY_EXIT=%ERRORLEVEL%"

echo.
if "%DEPLOY_EXIT%"=="0" (
    echo Deployment finished successfully.
) else (
    echo Deployment failed. Exit code: %DEPLOY_EXIT%
)
pause
exit /b %DEPLOY_EXIT%
