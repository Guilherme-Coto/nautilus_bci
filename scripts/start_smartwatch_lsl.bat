@echo off
echo Starting Smartwatch PPG and IMU LSL Bridge...
cd /d "C:\Users\VR\Documents\GitHub\nautilus_bci\scripts"
uv run python bridges/smartwatch_lsl_bridge.py --mode udp --port 5005
pause
