@echo off
setlocal
cd /d "%~dp0.."
set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"
set "BLUE_MODEL=models\sedon\reference_march_blue_shuffle_v3_best\best_model.zip"
set "BLUE_VECNORM=models\sedon\reference_march_blue_shuffle_v3_best\vecnorm.pkl"
set "BLUE_CONFIG=configs\sedon\reference_march_pose_1_4_blue_shuffle_v3.json"

if not "%~1"=="" (
  %PYTHON% eval.py --project sedon %*
  exit /b %ERRORLEVEL%
)

echo Sedon Eval Options:
echo.
echo   1. Watch Blue-like shuffle v3 best in MuJoCo viewer
echo   2. Record Blue-like shuffle v3 best to reports\sedon_eval.gif
echo   3. Headless numeric eval, Blue-like shuffle v3 best, 5 episodes
echo   4. Watch legacy models\sedon\latest_model.zip in MuJoCo viewer
echo   5. Watch legacy best available checkpoint in MuJoCo viewer
echo.
set /p choice="Select (1-5): "

if "%choice%"=="1" call :run_blue --episodes 1 --render
if "%choice%"=="2" call :run_blue --episodes 1 --record
if "%choice%"=="3" call :run_blue --episodes 5
if "%choice%"=="4" %PYTHON% eval.py --project sedon --episodes 1 --render --model-path models\sedon\latest_model.zip
if "%choice%"=="5" %PYTHON% eval.py --project sedon --episodes 1 --render

pause
exit /b %ERRORLEVEL%

:run_blue
if not exist "%BLUE_MODEL%" (
  echo Missing Blue-like shuffle model: %BLUE_MODEL%
  echo Run remote retrieval first, then try again.
  exit /b 1
)
if not exist "%BLUE_VECNORM%" (
  echo Missing Blue-like shuffle VecNormalize file: %BLUE_VECNORM%
  echo Run remote retrieval first, then try again.
  exit /b 1
)
if not exist "%BLUE_CONFIG%" (
  echo Missing Blue-like shuffle config: %BLUE_CONFIG%
  exit /b 1
)
set "SEDON_CONFIG_OVERRIDES=%BLUE_CONFIG%"
%PYTHON% eval.py --project sedon %* --model-path "%BLUE_MODEL%" --vecnorm-path "%BLUE_VECNORM%" --ignore-train-config
exit /b %ERRORLEVEL%
