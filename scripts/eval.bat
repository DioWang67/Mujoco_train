@echo off
setlocal
cd /d "%~dp0.."
set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"

echo Eval Options:
echo.
echo H1:
echo   1. Base eval (best_model/h1_ppo)
echo   2. DR eval (prefer h1_ppo_dr + h1_vecnorm_dr)
echo   3. Auto eval (auto detect DR artifacts)
echo   4. Record video (base)
echo   5. Record video (DR)
echo   6. Compare base ^& DR (headless, 3 eps each)
echo   7. Compare base ^& DR (numeric report script)
echo   8. Compare + gate check (release gates)
echo   9. Multi-seed aggregate compare (95%% CI)
echo   10. Benchmark matrix report
echo.
echo Grasp:
echo   11. Local grasp eval with viewer (5 eps)
echo   12. Local grasp eval headless (10 eps)
echo   13. Remote grasp eval headless (10 eps)
echo.
set /p choice="Select (1-13): "

if "%choice%"=="1" %PYTHON% -m h1_baseline.eval
if "%choice%"=="2" %PYTHON% -m h1_baseline.eval --dr
if "%choice%"=="3" %PYTHON% -m h1_baseline.eval --auto-dr
if "%choice%"=="4" %PYTHON% -m h1_baseline.eval --record
if "%choice%"=="5" %PYTHON% -m h1_baseline.eval --record --dr
if "%choice%"=="6" (
  echo [BASE]
  %PYTHON% -m h1_baseline.eval --episodes 3 --no-render
  echo [DR]
  %PYTHON% -m h1_baseline.eval --episodes 3 --no-render --dr
)
if "%choice%"=="7" %PYTHON% -m tools.compare_eval --episodes 8 --vel 1.0
if "%choice%"=="8" (
  %PYTHON% -m tools.compare_eval --episodes 8 --vel 1.0 --out-json reports/compare_report.json --out-csv reports/compare_report.csv
  %PYTHON% -m tools.gate_check --report reports/compare_report.json --gates configs/gate_profiles.json --mode single --profile preprod
)
if "%choice%"=="9" (
  %PYTHON% -m tools.aggregate_compare --seeds 3 --seed-start 42 --episodes 5 --vel 1.0 --out-json reports/aggregate_compare.json --out-csv reports/aggregate_compare.csv
  %PYTHON% -m tools.gate_check --report reports/aggregate_compare.json --gates configs/gate_profiles.json --mode aggregate --profile preprod
)
if "%choice%"=="10" %PYTHON% -m tools.benchmark_matrix --matrix configs/benchmark_matrix.json --out-json reports/benchmark_report.json --out-csv reports/benchmark_report.csv
if "%choice%"=="11" %PYTHON% -m tools.eval_grasp --episodes 5
if "%choice%"=="12" %PYTHON% -m tools.eval_grasp --episodes 10 --no-render
if "%choice%"=="13" call "%~dp0grasp_remote_eval.bat"
pause
