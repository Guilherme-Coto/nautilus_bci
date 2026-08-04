# Nautilus BCI: Development & Testing Setup

This branch (`dev/mock-event-testing`) is designed for development and testing on a standard PC without requiring access to the physical g.Nautilus EEG hardware.

The primary goals of this setup are:
1. Provide a lightweight GUI that strictly uses mock EEG data.
2. Ensure the LSL marker system correctly formats and exports into BIDS `_events.tsv` format across all tasks.
3. Allow easy installation on any OS without proprietary hardware SDKs.

## Installation

We use [uv](https://github.com/astral-sh/uv) to manage Python dependencies rapidly.

1. **Clone the repository and checkout this branch:**
   ```bash
   git clone <repository_url>
   cd nautilus_bci
   git checkout dev/mock-event-testing
   ```

2. **Sync the basic dependencies:**
   This command installs all the required tools (PySide6, MNE, BIDS, pylsl, etc) *without* the hardware-specific dependencies (like `pygds`).
   ```bash
   cd scripts
   uv sync
   ```

*(Note: If you are on the actual data collection machine with the g.Nautilus plugged in, you would install the hardware dependencies by running `uv sync --extra hardware` instead).*

## How to Test the Event System

You can run an automated integration test that spins up the mock streamer, loops through all major task paradigms (Motor Imagery, Music Memory, Video Dataset), injects test markers, exports them via the BIDS recorder, and finally verifies that every single marker was successfully saved to the BIDS `_events.tsv` file.

```bash
cd scripts
uv run python test_all_tasks_events.py
```

If it succeeds, you will see a `[SUCCESS] All tasks successfully generated and recorded LSL event markers!` message in the terminal.

## How to Use the Development GUI

If you want to manually test the UI workflows, launch the development suite:

```bash
cd scripts
uv run python run_dev_suite.py
```

This launches a version of the main Control Panel where:
- The EEG Streamer is hardcoded to launch `mock_lsl_streamer.py`.
- You can manually test starting the BIDS recording and launching the task GUIs without needing actual hardware locks.
