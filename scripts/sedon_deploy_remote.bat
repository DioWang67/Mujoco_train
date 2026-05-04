@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0deploy_remote_release.ps1" -ProjectSlug sedon -VerifyProject sedon -IncludePrivateAssets -CleanRelease
set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo sedon_deploy_remote.bat finished with exit code %EXIT_CODE%.
if not "%EXIT_CODE%"=="0" (
  echo Deployment failed. Review the error output above.
)
pause
exit /b %EXIT_CODE%
