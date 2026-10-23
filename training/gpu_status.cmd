@echo off
rem Double-click to write the current GPU memory, load and power draw to runs\grpo\gpu_status.txt.
cd /d "%~dp0.."
nvidia-smi > runs\grpo\gpu_status.txt 2>&1
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv >> runs\grpo\gpu_status.txt 2>&1
