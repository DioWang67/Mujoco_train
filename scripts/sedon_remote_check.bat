@echo off
setlocal

set "SCRIPT_DIR=%~dp0"

echo.
echo === Sedon remote check ===
echo.
call "%SCRIPT_DIR%remote_auto_deploy.bat" --check-only
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if "%EXIT_CODE%"=="0" (
    echo Sedon remote check completed successfully.
) else (
    echo Sedon remote check failed with exit code %EXIT_CODE%.
)
echo.
pause
exit /b %EXIT_CODE%
