@echo off
cd /d "%~dp0.."
echo Fine-tuning with Domain Randomization from pre-trained model...
echo   Source model : models\best_model.zip  (peak performance checkpoint)
echo   VecNormalize : load from base training stats
echo   DR enabled   : friction, mass, noise, delay, command randomization
echo   DR ramp      : start=0.05, full at 70%% progress
echo   LR           : 1e-4 -> 1e-5 (linear+floor)
echo   ent_coef     : 0.02 (higher than base for DR adaptation)
echo.
echo Note: DR models are saved as models\h1_ppo_dr.zip
echo.
python train.py --finetune models\best_model.zip --dr --dr-start-level 0.05 --dr-ramp-end 0.7
pause
