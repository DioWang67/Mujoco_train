@echo off
cd /d %~dp0
echo Starting H1 training with Domain Randomization...
echo   - Friction randomization  : x0.5 ~ x1.5 (progressive by DR level)
echo   - Mass randomization      : +-10%% (progressive by DR level)
echo   - Observation noise       : sigma=0.02 * DR level
echo   - Motor delay             : disabled by default (0 steps)
echo   - Command randomization   : follows target velocity curriculum
echo   - DR ramp                 : start=0.0, full at 35%% progress
echo.
echo Note: DR checkpoints/models will use h1_ppo_dr naming.
echo.
python train.py --dr --dr-start-level 0.0 --dr-ramp-end 0.35
pause
