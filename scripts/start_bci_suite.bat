@echo off
echo Starting BCI Motor Imagery & BIDS Studio...
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" run_bci_suite.py
) else (
    uv run python run_bci_suite.py
)
pause
