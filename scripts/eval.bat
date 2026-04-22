@echo off
cd /d "%~dp0.."
echo H1 Eval Options:
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
set /p choice="Select (1-10): "

if "%choice%"=="1" python eval.py
if "%choice%"=="2" python eval.py --dr
if "%choice%"=="3" python eval.py --auto-dr
if "%choice%"=="4" python eval.py --record
if "%choice%"=="5" python eval.py --record --dr
if "%choice%"=="6" (
  echo [BASE]
  python eval.py --episodes 3 --no-render
  echo [DR]
  python eval.py --episodes 3 --no-render --dr
)
if "%choice%"=="7" python -m tools.compare_eval --episodes 8 --vel 1.0
if "%choice%"=="8" (
  python -m tools.compare_eval --episodes 8 --vel 1.0 --out-json reports/compare_report.json --out-csv reports/compare_report.csv
  python -m tools.gate_check --report reports/compare_report.json --gates configs/gate_profiles.json --mode single --profile preprod
)
if "%choice%"=="9" (
  python -m tools.aggregate_compare --seeds 3 --seed-start 42 --episodes 5 --vel 1.0 --out-json reports/aggregate_compare.json --out-csv reports/aggregate_compare.csv
  python -m tools.gate_check --report reports/aggregate_compare.json --gates configs/gate_profiles.json --mode aggregate --profile preprod
)
if "%choice%"=="10" python -m tools.benchmark_matrix --matrix configs/benchmark_matrix.json --out-json reports/benchmark_report.json --out-csv reports/benchmark_report.csv
pause
