@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%.."
set "PYTHON_EXE=%REPO_ROOT%\.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=python"
)

echo.
echo === Sedon remote training status ===
echo.
pushd "%REPO_ROOT%"
"%PYTHON_EXE%" -m tools.remote_training --project sedon --status
set "EXIT_CODE=%ERRORLEVEL%"
popd
echo.
if "%EXIT_CODE%"=="0" (
    echo Sedon remote training status completed successfully.
) else (
    echo Sedon remote training status failed with exit code %EXIT_CODE%.
)
echo.
pause
exit /b %EXIT_CODE%
