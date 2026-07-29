"""
Multimodal BCI Master Dashboard & Data Collection Studio
=========================================================

Unified GUI application displaying:
  1. 🧠 EEG Scalp Quality (2D 10-20 Headmap & Channel Goodness Table)
  2. ⚡ Key EEG Waves & Motor Cortex Signals (C3, C4, Cz, O1, O2 + Alpha/Beta/Delta Power)
  3. ⌚ Smartwatch PPG & 6-DOF IMU Motion Vectors (Heart Rate BPM, Accel XYZ, Gyro XYZ, Motion Magnitude)
  4. 🔴 1-Click Multimodal BIDS Recording Engine

Usage:
  uv run python multimodal_bci_dashboard.py
"""
import sys
import os
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
_curr_dir = os.path.dirname(__file__)
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)


import sys
import os
import time
import subprocess
import numpy as np
import scipy.signal as signal

from PySide6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg

try:
    from pylsl import StreamInlet, resolve_streams
    HAS_LSL = True
except ImportError:
    HAS_LSL = False


from recorders.multimodal_bids_recorder import MultimodalBIDSRecorder
from visualizers.eeg_headmap_quality_visualizer import HeadMapCanvas, ELECTRODE_POSITIONS_1020


class MultimodalBCIDashboard(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multimodal BCI Master Dashboard (EEG + Smartwatch PPG/IMU + BIDS)")
        self.resize(1300, 850)

        # Process & Recording handles
        self.streamer_process = None
        self.watch_bridge_process = None
        self.recorder = None

        # LSL Stream Inlets
        self.eeg_inlet = None
        self.imu_inlet = None
        self.ppg_inlet = None

        # Data Buffers
        self.fs_eeg = 250.0
        self.buf_samples_eeg = int(self.fs_eeg * 3.0)
        self.eeg_buffer = np.zeros((self.buf_samples_eeg, 33))  # 32 channels + Battery
        self.eeg_ch_names = [
            'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 
            'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 
            'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2', 'CP1', 'CP2', 
            'FC5', 'FC6', 'CP5', 'CP6', 'FT9', 'FT10', 'TP9', 'TP10', 'Battery'
        ]

        self.buf_samples_imu = 150  # 3 seconds @ 50 Hz
        self.imu_buffer = np.zeros((self.buf_samples_imu, 6))  # Accel XYZ, Gyro XYZ
        
        self.buf_samples_ppg = 30  # 30 seconds @ 1 Hz
        self.ppg_history = np.zeros(self.buf_samples_ppg)

        # Filters for EEG
        nyq = 0.5 * self.fs_eeg
        self.b_bp, self.a_bp = signal.butter(2, [2.0 / nyq, 45.0 / nyq], btype='band')

        self.init_ui()

        # Timers
        self.lsl_discover_timer = QtCore.QTimer()
        self.lsl_discover_timer.timeout.connect(self.discover_lsl_streams)
        self.lsl_discover_timer.start(1000)

        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self.poll_and_render_all)
        self.update_timer.start(33)  # ~30 FPS

    def init_ui(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        # ----------------------------------------------------
        # TOP CONTROL BAR (1-Click Streamers & BIDS Recording)
        # ----------------------------------------------------
        top_bar = QtWidgets.QGroupBox("🎛️ Master Control & Stream Launcher")
        top_bar.setStyleSheet("QGroupBox { font-weight: bold; color: #ECF0F1; border: 1px solid #34495E; border-radius: 6px; padding: 10px; }")
        top_layout = QtWidgets.QHBoxLayout(top_bar)

        # Metadata inputs
        top_layout.addWidget(QtWidgets.QLabel("Sub:"))
        self.txt_sub = QtWidgets.QLineEdit("01")
        self.txt_sub.setFixedWidth(45)
        top_layout.addWidget(self.txt_sub)

        top_layout.addWidget(QtWidgets.QLabel("Ses:"))
        self.txt_ses = QtWidgets.QLineEdit("01")
        self.txt_ses.setFixedWidth(45)
        top_layout.addWidget(self.txt_ses)

        # Streamer Launch Buttons
        self.btn_streamer = QtWidgets.QPushButton("▶ Launch EEG Streamer")
        self.btn_streamer.setStyleSheet("background-color: #2980B9; color: white; font-weight: bold; padding: 6px 12px; border-radius: 4px;")
        self.btn_streamer.clicked.connect(self.toggle_eeg_streamer)
        top_layout.addWidget(self.btn_streamer)

        self.btn_watch_bridge = QtWidgets.QPushButton("⌚ Launch Watch Bridge")
        self.btn_watch_bridge.setStyleSheet("background-color: #8E44AD; color: white; font-weight: bold; padding: 6px 12px; border-radius: 4px;")
        self.btn_watch_bridge.clicked.connect(self.toggle_watch_bridge)
        top_layout.addWidget(self.btn_watch_bridge)

        # BIDS Recording Toggle Button
        self.btn_record = QtWidgets.QPushButton("🔴 Start Multimodal BIDS Recording")
        self.btn_record.setStyleSheet("background-color: #C0392B; color: white; font-weight: bold; padding: 6px 16px; border-radius: 4px;")
        self.btn_record.clicked.connect(self.toggle_bids_recording)
        top_layout.addWidget(self.btn_record)

        # Stream Status Badges
        self.lbl_status_eeg = QtWidgets.QLabel("EEG: 🔴 Offline")
        self.lbl_status_eeg.setStyleSheet("color: #E74C3C; font-weight: bold;")
        top_layout.addWidget(self.lbl_status_eeg)

        self.lbl_status_watch = QtWidgets.QLabel("Watch: 🔴 Offline")
        self.lbl_status_watch.setStyleSheet("color: #E74C3C; font-weight: bold;")
        top_layout.addWidget(self.lbl_status_watch)

        main_layout.addWidget(top_bar)

        # ----------------------------------------------------
        # TABBED MULTI-PANEL VIEW
        # ----------------------------------------------------
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setStyleSheet("QTabWidget::pane { border: 1px solid #34495E; background: #1E222B; } QTabBar::tab { background: #2C3E50; color: #ECF0F1; padding: 8px 16px; font-weight: bold; } QTabBar::tab:selected { background: #2980B9; }")

        # Tab 1: EEG Scalp Goodness & 10-20 Headmap
        self.tab_headmap = self.build_headmap_tab()
        self.tabs.addTab(self.tab_headmap, "🧠 10-20 Headmap & Electrode Quality")

        # Tab 2: Motor Cortex Waves & Brain Rhythms
        self.tab_waves = self.build_waves_tab()
        self.tabs.addTab(self.tab_waves, "⚡ Motor Waves & Brain Rhythms")

        # Tab 3: Smartwatch PPG & 6-DOF IMU Vectors
        self.tab_watch = self.build_watch_tab()
        self.tabs.addTab(self.tab_watch, "⌚ Smartwatch PPG & IMU Vectors")

        main_layout.addWidget(self.tabs)

    # ----------------------------------------------------
    # TAB 1 BUILDER: Headmap & Quality Table
    # ----------------------------------------------------
    def build_headmap_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)

        self.headmap_canvas = HeadMapCanvas(self.eeg_ch_names)
        layout.addWidget(self.headmap_canvas, stretch=3)

        right_panel = QtWidgets.QVBoxLayout()
        self.lbl_headmap_summary = QtWidgets.QLabel("Active Channels: Waiting for EEG stream...")
        self.lbl_headmap_summary.setStyleSheet("font-size: 15px; font-weight: bold; color: #ECF0F1;")
        right_panel.addWidget(self.lbl_headmap_summary)

        # Legend
        legend_layout = QtWidgets.QHBoxLayout()
        legend_layout.addWidget(self.create_legend_dot("🟢 Good", "#2ECC71"))
        legend_layout.addWidget(self.create_legend_dot("🟡 Noisy", "#F1C40F"))
        legend_layout.addWidget(self.create_legend_dot("🟧 Railed", "#E67E22"))
        legend_layout.addWidget(self.create_legend_dot("🔴 Flatline", "#E74C3C"))
        right_panel.addLayout(legend_layout)

        # Table
        self.table_quality = QtWidgets.QTableWidget(len(self.eeg_ch_names) - 1, 3)
        self.table_quality.setHorizontalHeaderLabels(["Channel", "Metrics (uV)", "Status"])
        self.table_quality.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.table_quality.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.table_quality.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.table_quality.setStyleSheet("background-color: #151821; color: #ECF0F1; gridline-color: #2C3E50; font-size: 11px;")

        for i, name in enumerate(self.eeg_ch_names[:-1]):
            self.table_quality.setItem(i, 0, QtWidgets.QTableWidgetItem(name))
            self.table_quality.setItem(i, 1, QtWidgets.QTableWidgetItem("std: 0.0 | p-p: 0.0"))
            self.table_quality.setItem(i, 2, QtWidgets.QTableWidgetItem("Checking..."))

        right_panel.addWidget(self.table_quality)
        layout.addLayout(right_panel, stretch=2)
        return widget

    def create_legend_dot(self, text, color_hex):
        lbl = QtWidgets.QLabel(text)
        lbl.setStyleSheet(f"background-color: {color_hex}; color: #000; font-weight: bold; border-radius: 4px; padding: 4px;")
        return lbl

    # ----------------------------------------------------
    # TAB 2 BUILDER: Waves & Rhythm Meters
    # ----------------------------------------------------
    def build_waves_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)

        # Scope Plot
        self.glw_waves = pg.GraphicsLayoutWidget()
        self.glw_waves.setBackground('#0D1117')
        layout.addWidget(self.glw_waves, stretch=3)

        self.plot_waves = self.glw_waves.addPlot(title="Motor Cortex & Occipital Waves (Cz, C3, C4, O1, O2)")
        self.plot_waves.setLabel('bottom', 'Time (s)')
        self.plot_waves.setLabel('left', 'Amplitude (uV)')
        self.plot_waves.showGrid(x=True, y=True, alpha=0.3)

        self.target_wave_channels = ['Cz', 'C3', 'C4', 'O1', 'O2']
        self.wave_colors = ['#00E676', '#74B9FF', '#E040FB', '#FFEAA7', '#FF7675']
        self.wave_curves = []
        self.t_vec_eeg = np.linspace(-3.0, 0, self.buf_samples_eeg)

        for color, name in zip(self.wave_colors, self.target_wave_channels):
            c = self.plot_waves.plot(pen=pg.mkPen(color=color, width=2.0), name=name)
            self.wave_curves.append(c)

        # Right Panel: Rhythm Bars
        right_panel = QtWidgets.QVBoxLayout()
        right_panel.setSpacing(10)
        right_panel.addWidget(QtWidgets.QLabel("<b>Live Brain Rhythm Power</b>"))

        right_panel.addWidget(QtWidgets.QLabel("<font color='#00E676'>Alpha (8-12 Hz) - Relaxation</font>"))
        self.bar_alpha = QtWidgets.QProgressBar()
        self.bar_alpha.setStyleSheet("QProgressBar::chunk { background: #00E676; }")
        right_panel.addWidget(self.bar_alpha)

        right_panel.addWidget(QtWidgets.QLabel("<font color='#74B9FF'>Beta (12-30 Hz) - Active Focus</font>"))
        self.bar_beta = QtWidgets.QProgressBar()
        self.bar_beta.setStyleSheet("QProgressBar::chunk { background: #74B9FF; }")
        right_panel.addWidget(self.bar_beta)

        right_panel.addWidget(QtWidgets.QLabel("<font color='#FFEAA7'>Delta (0.5-4 Hz) - Baseline</font>"))
        self.bar_delta = QtWidgets.QProgressBar()
        self.bar_delta.setStyleSheet("QProgressBar::chunk { background: #FFEAA7; }")
        right_panel.addWidget(self.bar_delta)

        right_panel.addStretch()
        layout.addLayout(right_panel, stretch=1)
        return widget

    # ----------------------------------------------------
    # TAB 3 BUILDER: Smartwatch PPG & IMU Vectors
    # ----------------------------------------------------
    def build_watch_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        # Top Section: Heart Rate Gauge / Display
        hr_box = QtWidgets.QGroupBox("🫀 Live Smartwatch Heart Rate (PPG)")
        hr_box.setStyleSheet("QGroupBox { font-weight: bold; color: #ECF0F1; border: 1px solid #34495E; }")
        hr_layout = QtWidgets.QHBoxLayout(hr_box)

        self.lbl_hr_val = QtWidgets.QLabel("-- BPM")
        self.lbl_hr_val.setStyleSheet("font-size: 32px; font-weight: bold; color: #E74C3C;")
        hr_layout.addWidget(self.lbl_hr_val)

        self.lbl_hr_status = QtWidgets.QLabel("Status: Waiting for PPG stream...")
        self.lbl_hr_status.setStyleSheet("color: #BDC3C7;")
        hr_layout.addWidget(self.lbl_hr_status)
        layout.addWidget(hr_box)

        # Bottom Section: 6-DOF IMU Waveforms
        imu_box = QtWidgets.QGroupBox("⌚ 6-DOF IMU Motion Vectors (Accelerometer & Gyroscope)")
        imu_box.setStyleSheet("QGroupBox { font-weight: bold; color: #ECF0F1; border: 1px solid #34495E; }")
        imu_layout = QtWidgets.QVBoxLayout(imu_box)

        self.glw_imu = pg.GraphicsLayoutWidget()
        self.glw_imu.setBackground('#0D1117')
        imu_layout.addWidget(self.glw_imu)

        # Plot A: Accel XYZ
        self.plot_accel = self.glw_imu.addPlot(row=0, col=0, title="3-Axis Accelerometer (m/s²)")
        self.plot_accel.showGrid(x=True, y=True, alpha=0.3)
        self.t_vec_imu = np.linspace(-3.0, 0, self.buf_samples_imu)

        self.curve_ax = self.plot_accel.plot(pen=pg.mkPen('#E74C3C', width=2), name="Accel X")
        self.curve_ay = self.plot_accel.plot(pen=pg.mkPen('#2ECC71', width=2), name="Accel Y")
        self.curve_az = self.plot_accel.plot(pen=pg.mkPen('#3498DB', width=2), name="Accel Z")

        # Plot B: Total Motion Magnitude
        self.plot_mag = self.glw_imu.addPlot(row=0, col=1, title="Total Motion Magnitude |A|")
        self.plot_mag.showGrid(x=True, y=True, alpha=0.3)
        self.curve_mag = self.plot_mag.plot(pen=pg.mkPen('#F1C40F', width=2), name="Magnitude")

        layout.addWidget(imu_box)
        return widget

    # ----------------------------------------------------
    # LSL DISCOVERY & POLLING LOGIC
    # ----------------------------------------------------
    def discover_lsl_streams(self):
        if not HAS_LSL:
            return

        streams = resolve_streams(wait_time=0.2)
        found_eeg = False
        found_watch = False

        for s in streams:
            stype = s.type().upper()
            sname = s.name()

            if (stype == 'EEG' or 'gNautilus' in sname) and self.eeg_inlet is None:
                try:
                    self.eeg_inlet = StreamInlet(s, max_buflen=360)
                    found_eeg = True
                except Exception:
                    pass

            if (stype == 'IMU' or 'Smartwatch_IMU' in sname) and self.imu_inlet is None:
                try:
                    self.imu_inlet = StreamInlet(s, max_buflen=360)
                    found_watch = True
                except Exception:
                    pass

            if (stype == 'PPG' or 'Smartwatch_PPG' in sname) and self.ppg_inlet is None:
                try:
                    self.ppg_inlet = StreamInlet(s, max_buflen=360)
                    found_watch = True
                except Exception:
                    pass

        if self.eeg_inlet:
            self.lbl_status_eeg.setText("EEG: 🟢 Connected")
            self.lbl_status_eeg.setStyleSheet("color: #2ECC71; font-weight: bold;")
        else:
            self.lbl_status_eeg.setText("EEG: 🔴 Offline")
            self.lbl_status_eeg.setStyleSheet("color: #E74C3C; font-weight: bold;")

        if self.imu_inlet or self.ppg_inlet:
            self.lbl_status_watch.setText("Watch: 🟢 Connected")
            self.lbl_status_watch.setStyleSheet("color: #2ECC71; font-weight: bold;")
        else:
            self.lbl_status_watch.setText("Watch: 🔴 Offline")
            self.lbl_status_watch.setStyleSheet("color: #E74C3C; font-weight: bold;")

    def poll_and_render_all(self):
        # 1. Poll EEG
        if self.eeg_inlet:
            try:
                samples, _ = self.eeg_inlet.pull_chunk(max_samples=250)
                if samples:
                    chunk = np.array(samples, dtype=np.float64)
                    if np.max(np.abs(chunk)) < 0.01:
                        chunk *= 1e6  # Volts to uV
                    n = len(chunk)
                    if chunk.shape[1] == self.eeg_buffer.shape[1]:
                        self.eeg_buffer = np.roll(self.eeg_buffer, -n, axis=0)
                        self.eeg_buffer[-n:, :] = chunk
            except Exception:
                self.eeg_inlet = None

        # 2. Poll IMU
        if self.imu_inlet:
            try:
                samples, _ = self.imu_inlet.pull_chunk(max_samples=100)
                if samples:
                    chunk = np.array(samples, dtype=np.float64)
                    n = len(chunk)
                    if chunk.shape[1] >= 6:
                        self.imu_buffer = np.roll(self.imu_buffer, -n, axis=0)
                        self.imu_buffer[-n:, :] = chunk[:, :6]
            except Exception:
                self.imu_inlet = None

        # 3. Poll PPG / Heart Rate
        if self.ppg_inlet:
            try:
                samples, _ = self.ppg_inlet.pull_chunk(max_samples=10)
                if samples:
                    hr_val = samples[-1][0]
                    if hr_val > 0:
                        self.lbl_hr_val.setText(f"{hr_val:.0f} BPM")
                        self.lbl_hr_status.setText("Status: Live PPG Stream Active 🟢")
                    else:
                        self.lbl_hr_val.setText("Locking...")
                        self.lbl_hr_status.setText("Status: PPG Active — Searching for Pulse Lock 🟡")
            except Exception:
                self.ppg_inlet = None

        # Render active tab
        idx = self.tabs.currentIndex()
        if idx == 0:
            self.render_headmap_tab()
        elif idx == 1:
            self.render_waves_tab()
        elif idx == 2:
            self.render_watch_tab()

    def render_headmap_tab(self):
        stds = np.std(self.eeg_buffer, axis=0)
        ranges = np.ptp(self.eeg_buffer, axis=0)
        means = np.abs(np.mean(self.eeg_buffer, axis=0))

        channel_stds = {}
        channel_maxs = {}
        good_count = 0
        flat_count = 0

        for i, name in enumerate(self.eeg_ch_names[:-1]):
            std_val = stds[i]
            ptp_val = ranges[i]
            mean_val = means[i]
            channel_stds[name] = std_val
            channel_maxs[name] = ptp_val

            self.table_quality.item(i, 1).setText(f"std:{std_val:.1f} | p-p:{ptp_val:.1f}")

            if std_val < 0.5 or ptp_val < 0.5:
                item = QtWidgets.QTableWidgetItem("FLATLINE")
                item.setForeground(QtGui.QColor("#E74C3C"))
                flat_count += 1
            elif mean_val > 300.0 or ptp_val > 500.0:
                item = QtWidgets.QTableWidgetItem("RAILED/SATURATED")
                item.setForeground(QtGui.QColor("#E67E22"))
            elif std_val > 80.0:
                item = QtWidgets.QTableWidgetItem("HIGH NOISE")
                item.setForeground(QtGui.QColor("#F1C40F"))
            else:
                item = QtWidgets.QTableWidgetItem("GOOD")
                item.setForeground(QtGui.QColor("#2ECC71"))
                good_count += 1

            self.table_quality.setItem(i, 2, item)

        self.headmap_canvas.update_channel_status(channel_stds, channel_maxs)
        self.lbl_headmap_summary.setText(f"Active Channels: {good_count}/32 Good ({flat_count} Flat)")

    def render_waves_tab(self):
        filt = self.eeg_buffer - np.mean(self.eeg_buffer, axis=0)
        try:
            filt = signal.filtfilt(self.b_bp, self.a_bp, filt, axis=0)
        except Exception:
            pass

        spacing = max(20.0, np.std(filt) * 3.5)
        offsets = [0, spacing, spacing * 2, spacing * 3, spacing * 4]

        for i, name in enumerate(self.target_wave_channels):
            if name in self.eeg_ch_names:
                ch_idx = self.eeg_ch_names.index(name)
                y = filt[:, ch_idx] + offsets[i]
                self.wave_curves[i].setData(self.t_vec_eeg, y)

        self.plot_waves.setYRange(-spacing * 0.8, spacing * 4.8)

        # FFT Power
        if len(filt) >= 64:
            fft_vals = np.abs(np.fft.rfft(filt, axis=0))
            freqs = np.fft.rfftfreq(len(filt), 1.0 / self.fs_eeg)

            alpha = np.mean(fft_vals[(freqs >= 8) & (freqs <= 12), :]) if np.any((freqs >= 8) & (freqs <= 12)) else 0
            beta = np.mean(fft_vals[(freqs >= 12) & (freqs <= 30), :]) if np.any((freqs >= 12) & (freqs <= 30)) else 0
            delta = np.mean(fft_vals[(freqs >= 0.5) & (freqs <= 4), :]) if np.any((freqs >= 0.5) & (freqs <= 4)) else 0

            tot = alpha + beta + delta + 1e-6
            self.bar_alpha.setValue(int(min(100, (alpha / tot) * 100)))
            self.bar_beta.setValue(int(min(100, (beta / tot) * 100)))
            self.bar_delta.setValue(int(min(100, (delta / tot) * 100)))

    def render_watch_tab(self):
        ax = self.imu_buffer[:, 0]
        ay = self.imu_buffer[:, 1]
        az = self.imu_buffer[:, 2]
        mag = np.sqrt(ax**2 + ay**2 + az**2)

        self.curve_ax.setData(self.t_vec_imu, ax)
        self.curve_ay.setData(self.t_vec_imu, ay)
        self.curve_az.setData(self.t_vec_imu, az)
        self.curve_mag.setData(self.t_vec_imu, mag)

    # ----------------------------------------------------
    # STREAM & RECORDING CONTROLLER ACTIONS
    # ----------------------------------------------------
    def toggle_eeg_streamer(self):
        if self.streamer_process is None:
            script_path = os.path.join(os.path.dirname(__file__), "gds_to_lsl.py")
            self.streamer_process = subprocess.Popen([sys.executable, script_path, "--non-interactive"])
            self.btn_streamer.setText("⏹ Stop Hardware EEG Streamer")
            self.btn_streamer.setStyleSheet("background-color: #C0392B; color: white; font-weight: bold; padding: 6px 12px; border-radius: 4px;")
        else:
            self.streamer_process.terminate()
            self.streamer_process = None
            self.btn_streamer.setText("▶ Launch Hardware EEG Streamer")
            self.btn_streamer.setStyleSheet("background-color: #2980B9; color: white; font-weight: bold; padding: 6px 12px; border-radius: 4px;")

    def toggle_watch_bridge(self):
        if self.watch_bridge_process is None:
            script_path = os.path.join(os.path.dirname(__file__), "smartwatch_lsl_bridge.py")
            self.watch_bridge_process = subprocess.Popen([sys.executable, script_path, "--mode", "udp", "--port", "5005"])
            self.btn_watch_bridge.setText("⏹ Stop Watch Bridge")
            self.btn_watch_bridge.setStyleSheet("background-color: #C0392B; color: white; font-weight: bold; padding: 6px 12px; border-radius: 4px;")
        else:
            self.watch_bridge_process.terminate()
            self.watch_bridge_process = None
            self.btn_watch_bridge.setText("⌚ Launch Watch Bridge")
            self.btn_watch_bridge.setStyleSheet("background-color: #8E44AD; color: white; font-weight: bold; padding: 6px 12px; border-radius: 4px;")

    def toggle_bids_recording(self):
        if self.recorder is None or not self.recorder.is_recording:
            try:
                self.recorder = MultimodalBIDSRecorder(bids_root="bids_dataset_multimodal")
                connected = self.recorder.discover_and_connect_streams(timeout=2.0)
                if not connected:
                    QtWidgets.QMessageBox.warning(self, "No Streams", "No LSL streams found! Start your streamers first.")
                    return
                self.recorder.start_recording()
                self.btn_record.setText("⏹ Stop & Export Multimodal BIDS")
                self.btn_record.setStyleSheet("background-color: #D63031; color: white; font-weight: bold; padding: 6px 16px; border-radius: 4px;")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Could not start recorder: {e}")
        else:
            sub = self.txt_sub.text().strip()
            ses = self.txt_ses.text().strip()
            self.recorder.stop_and_export_bids(subject_id=sub, session_id=ses)
            self.btn_record.setText("🔴 Start Multimodal BIDS Recording")
            self.btn_record.setStyleSheet("background-color: #C0392B; color: white; font-weight: bold; padding: 6px 16px; border-radius: 4px;")
            QtWidgets.QMessageBox.information(self, "BIDS Exported", "Successfully exported Multimodal BIDS dataset!")
            self.recorder = None


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = MultimodalBCIDashboard()
    win.show()
    sys.exit(app.exec())
