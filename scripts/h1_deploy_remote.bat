@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0deploy_remote_release.ps1" -ProjectSlug h1 -VerifyProject h1 -CleanRelease
pause
