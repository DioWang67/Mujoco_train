@echo off
setlocal
cd /d "%~dp0.."
set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"

if not "%~1"=="" (
  %PYTHON% eval.py --project sedon %*
  exit /b %ERRORLEVEL%
)

echo Sedon Eval Options:
echo.
echo   1. Watch best checkpoint in MuJoCo viewer
echo   2. Record best checkpoint to reports\sedon_eval.gif
echo   3. Headless numeric eval, 5 episodes
echo   4. Watch latest_model.zip in MuJoCo viewer
echo.
set /p choice="Select (1-4): "

if "%choice%"=="1" %PYTHON% eval.py --project sedon --episodes 1 --render
if "%choice%"=="2" %PYTHON% eval.py --project sedon --episodes 1 --record
if "%choice%"=="3" %PYTHON% eval.py --project sedon --episodes 5
if "%choice%"=="4" %PYTHON% eval.py --project sedon --episodes 1 --render --model-path models\sedon\latest_model.zip

pause
