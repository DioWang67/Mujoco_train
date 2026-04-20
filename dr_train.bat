@echo off
cd /d %~dp0
echo Starting H1 training with Domain Randomization...
echo   - Friction randomization  : x0.5 ~ x1.5
echo   - Mass randomization      : +-10%%
echo   - Observation noise       : Gaussian sigma=0.02
echo   - Motor delay             : 0 ~ 40ms
echo   - Command randomization   : vx=[0.3,1.5] vy/vyaw=[-0.3,0.3]
echo   - Curriculum Learning     : 0.3 m/s -> 1.0 m/s
echo.
echo Note: Expect slower convergence (25-30M steps)
echo.
python train.py --dr
pause
