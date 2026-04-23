@echo off
setlocal
set "TARGET=%~1"
if "%TARGET%"=="" (
  echo Usage: scripts\run_remote_train.bat ^<h1^|grasp^> [train args...]
  exit /b 1
)

if /I "%TARGET%"=="h1" (
  call "%~dp0run_remote_h1_train.bat" %2 %3 %4 %5 %6 %7 %8 %9
  exit /b %errorlevel%
)

if /I "%TARGET%"=="grasp" (
  call "%~dp0run_remote_grasp_train.bat" %2 %3 %4 %5 %6 %7 %8 %9
  exit /b %errorlevel%
)

echo Unsupported target: %TARGET%
echo Supported targets: h1, grasp
exit /b 1
