@echo off
echo Starting Multimodal BCI Master Dashboard Studio...
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" visualizers/multimodal_bci_dashboard.py
) else (
    uv run python visualizers/multimodal_bci_dashboard.py
)
pause
