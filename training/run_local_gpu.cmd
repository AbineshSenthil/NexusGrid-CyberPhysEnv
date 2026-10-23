@echo off
rem Double-click to run the NexusGrid GPU pipeline (baseline eval, GRPO, eval after GRPO).
cd /d "%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_local_gpu.ps1" %*
echo.
echo Finished. Logs: runs\grpo\*.log
pause
