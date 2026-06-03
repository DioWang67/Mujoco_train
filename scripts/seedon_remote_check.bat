@echo off
setlocal

set "SCRIPT_DIR=%~dp0"

echo.
echo === Seedon remote check ===
echo.
call "%SCRIPT_DIR%remote_auto_deploy.bat" --check-only
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if "%EXIT_CODE%"=="0" (
    echo Seedon remote check completed successfully.
) else (
    echo Seedon remote check failed with exit code %EXIT_CODE%.
)
echo.
pause
exit /b %EXIT_CODE%
