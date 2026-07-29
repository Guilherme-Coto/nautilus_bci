"""
Video Dataset Presentation Task for BCI / Multimodal Data Collection
======================================================================
Displays video stimuli for 3 seconds (configurable), toggles audio output,
supports optional external WAV audio files (in videos/audio/), provides Peak 
Audio Normalization (-1.0 dB), applies 1.5s rest intervals, displays video title 
prominently on top, and broadcasts LSL event markers for BIDS synchronization.
"""

import sys
import os
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
_curr_dir = os.path.dirname(__file__)
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)

import random
import time
import wave
import numpy as np

from PySide6.QtCore import Qt, QTimer, QUrl, Slot
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QFormLayout, QGroupBox,
    QProgressBar, QStackedWidget, QMessageBox, QCheckBox, QFileDialog,
    QDoubleSpinBox, QSpinBox
)
from PySide6.QtGui import QFont, QColor, QPalette
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget

try:
    from pylsl import StreamInfo, StreamOutlet, local_clock
    HAS_LSL = True
except ImportError:
    HAS_LSL = False


def normalize_wav_file(input_path, output_path, target_peak_dB=-1.0):
    """Normalize a PCM WAV audio file to target peak dB (-1.0 dBFS by default)."""
    try:
        with wave.open(input_path, 'rb') as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            raw_bytes = wf.readframes(n_frames)

        if n_frames == 0 or len(raw_bytes) == 0:
            return input_path

        dtype = np.int16 if sampwidth == 2 else (np.int32 if sampwidth == 4 else np.uint8)
        data = np.frombuffer(raw_bytes, dtype=dtype).astype(np.float32)

        max_val = np.max(np.abs(data))
        if max_val > 0:
            target_linear = 10.0 ** (target_peak_dB / 20.0)
            if dtype == np.int16:
                max_possible = 32767.0
            elif dtype == np.int32:
                max_possible = 2147483647.0
            else:
                max_possible = 127.5
                data = data - 127.5

            scale = (target_linear * max_possible) / max_val
            normalized_data = data * scale
            normalized_data = np.clip(normalized_data, -max_possible, max_possible)

            if dtype == np.int16:
                out_bytes = normalized_data.astype(np.int16).tobytes()
            elif dtype == np.int32:
                out_bytes = normalized_data.astype(np.int32).tobytes()
            else:
                out_bytes = (normalized_data + 127.5).astype(np.uint8).tobytes()
        else:
            out_bytes = raw_bytes

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(n_channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(framerate)
            wf.writeframes(out_bytes)
        return output_path
    except Exception as e:
        print(f"[-] WAV Normalization error for {input_path}: {e}")
        return input_path


class VideoDatasetTaskApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BCI Video Dataset Presentation Suite")
        self.resize(1080, 840)

        # Default paths
        base_dir = os.path.dirname(os.path.dirname(__file__))
        self.video_dir = os.path.abspath(os.path.join(base_dir, "videos"))
        self.audio_dir = os.path.abspath(os.path.join(self.video_dir, "audio"))
        self.norm_cache_dir = os.path.abspath(os.path.join(self.audio_dir, "normalized"))
        
        os.makedirs(self.video_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)
        os.makedirs(self.norm_cache_dir, exist_ok=True)

        # LSL Marker Outlet setup
        self.outlet = None
        self.init_lsl()

        # Dual Media Engine Setup (Video Player + External WAV Audio Player)
        self.video_player = QMediaPlayer()
        self.video_audio = QAudioOutput()
        self.video_player.setAudioOutput(self.video_audio)
        self.video_audio.setVolume(0.85)

        self.custom_audio_player = QMediaPlayer()
        self.custom_audio = QAudioOutput()
        self.custom_audio_player.setAudioOutput(self.custom_audio)
        self.custom_audio.setVolume(0.85)

        # Paradigm Parameters (Defaults: Video 3.0s, Rest 1.5s)
        self.t_video = 3.0   # seconds
        self.t_rest = 1.5    # seconds
        self.audio_enabled = True
        self.normalize_audio = True
        self.use_external_audio = True
        self.repetitions = 1
        self.randomize_order = True

        # State tracking
        self.video_files = []
        self.trials = []  # List of dicts: {'video': path, 'audio': path_or_None}
        self.current_trial_idx = 0
        self.current_phase = "INIT"  # "VIDEO" or "REST"

        self.trial_timer = QTimer()
        self.trial_timer.setSingleShot(True)
        self.trial_timer.timeout.connect(self.advance_trial_phase)

        self.tick_timer = QTimer()
        self.tick_timer.timeout.connect(self.update_progress)
        self.phase_start_time = 0.0
        self.current_phase_duration = 1.0

        self.init_ui()

    def init_lsl(self):
        if HAS_LSL:
            try:
                info = StreamInfo(
                    name='VideoTaskMarkers',
                    type='Markers',
                    channel_count=1,
                    nominal_srate=0,
                    channel_format='string',
                    source_id='Video_Task_Markers_2026'
                )
                self.outlet = StreamOutlet(info)
                print("[+] LSL Marker Outlet created successfully ('VideoTaskMarkers').")
            except Exception as e:
                print(f"[-] Failed to create LSL Marker Outlet: {e}")
                self.outlet = None
        else:
            print("[!] PyLSL not installed. Running in standalone visual mode.")

    def send_marker(self, marker_str):
        timestamp = local_clock() if HAS_LSL else time.time()
        print(f"[MARKER @ {timestamp:.3f}] {marker_str}")
        if self.outlet:
            self.outlet.push_sample([marker_str], timestamp)

    def init_ui(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(15, 18, 25))
        palette.setColor(QPalette.WindowText, QColor(240, 240, 245))
        palette.setColor(QPalette.Base, QColor(25, 30, 42))
        palette.setColor(QPalette.Text, QColor(240, 240, 245))
        palette.setColor(QPalette.Button, QColor(35, 45, 65))
        palette.setColor(QPalette.ButtonText, QColor(240, 240, 245))
        self.setPalette(palette)

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.config_screen = self.create_config_screen()
        self.stacked_widget.addWidget(self.config_screen)

        self.task_screen = self.create_task_screen()
        self.stacked_widget.addWidget(self.task_screen)

        self.stacked_widget.setCurrentWidget(self.config_screen)

    def create_config_screen(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(40, 25, 40, 25)
        layout.setSpacing(12)

        title = QLabel("Video Dataset Presentation Task")
        title.setFont(QFont("Arial", 22, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #4DEEEA;")
        layout.addWidget(title)

        subtitle = QLabel("3s Video Stimulus • Audio Toggle & Peak Normalization • 1.5s Rest • LSL Synchronized")
        subtitle.setFont(QFont("Arial", 11))
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #A0A5B5;")
        layout.addWidget(subtitle)

        # Form Group
        form_group = QGroupBox("Experiment & Paradigm Settings")
        form_group.setFont(QFont("Arial", 11, QFont.Bold))
        form_group.setStyleSheet("QGroupBox { color: #4DEEEA; border: 1px solid #2C354A; border-radius: 8px; margin-top: 5px; padding: 12px; }")
        form_layout = QFormLayout(form_group)
        form_layout.setSpacing(10)

        self.sub_input = QLineEdit("sub-01")
        self.ses_input = QLineEdit("ses-01")
        form_layout.addRow("Subject ID:", self.sub_input)
        form_layout.addRow("Session ID:", self.ses_input)

        # Video directory selector
        dir_layout = QHBoxLayout()
        self.lbl_video_dir = QLineEdit(self.video_dir)
        btn_browse = QPushButton("Browse...")
        btn_browse.setStyleSheet("background-color: #2C354A; color: white; padding: 5px 12px; border-radius: 4px;")
        btn_browse.clicked.connect(self.browse_video_dir)
        dir_layout.addWidget(self.lbl_video_dir)
        dir_layout.addWidget(btn_browse)
        form_layout.addRow("Video Folder:", dir_layout)

        # Audio Options
        self.chk_audio = QCheckBox("Enable Audio Output during Video Playback")
        self.chk_audio.setChecked(True)
        self.chk_audio.setStyleSheet("color: #FFEAA7; font-weight: bold;")
        form_layout.addRow("Audio Master Switch:", self.chk_audio)

        self.chk_normalize = QCheckBox("Normalize Audio Volume (-1.0 dB Peak)")
        self.chk_normalize.setChecked(True)
        self.chk_normalize.setStyleSheet("color: #00E676; font-weight: bold;")
        self.chk_normalize.setToolTip("Normalizes WAV audio files to -1.0 dBFS peak amplitude to eliminate volume discrepancies.")
        form_layout.addRow("Audio Normalization:", self.chk_normalize)

        self.chk_ext_audio = QCheckBox("Use External Audio Folder (videos/audio/*.wav)")
        self.chk_ext_audio.setChecked(True)
        self.chk_ext_audio.setStyleSheet("color: #74B9FF; font-weight: bold;")
        self.chk_ext_audio.setToolTip("If a matching .wav file is found in videos/audio/, it will play in sync with the video.")
        form_layout.addRow("External Audio Source:", self.chk_ext_audio)

        # Video Duration (default 3.0s)
        self.spn_video_dur = QDoubleSpinBox()
        self.spn_video_dur.setRange(1.0, 60.0)
        self.spn_video_dur.setValue(3.0)
        self.spn_video_dur.setSingleStep(0.5)
        self.spn_video_dur.setSuffix(" sec")
        form_layout.addRow("Video Presentation Time:", self.spn_video_dur)

        # Rest Duration (default 1.5s)
        self.spn_rest_dur = QDoubleSpinBox()
        self.spn_rest_dur.setRange(0.5, 30.0)
        self.spn_rest_dur.setValue(1.5)
        self.spn_rest_dur.setSingleStep(0.5)
        self.spn_rest_dur.setSuffix(" sec")
        form_layout.addRow("Rest Interval Duration:", self.spn_rest_dur)

        # Repetitions
        self.spn_reps = QSpinBox()
        self.spn_reps.setRange(1, 20)
        self.spn_reps.setValue(1)
        form_layout.addRow("Repetitions per Video:", self.spn_reps)

        # Randomize Order
        self.chk_random = QCheckBox("Randomize Video Playback Order")
        self.chk_random.setChecked(True)
        form_layout.addRow("Order:", self.chk_random)

        layout.addWidget(form_group)

        # Video scan status label
        self.lbl_scan_status = QLabel("Scanning video directory...")
        self.lbl_scan_status.setStyleSheet("color: #74B9FF;")
        layout.addWidget(self.lbl_scan_status)
        self.scan_video_directory()

        # Start Button
        btn_start = QPushButton("▶ START VIDEO DATASET EXPERIMENT")
        btn_start.setFont(QFont("Arial", 14, QFont.Bold))
        btn_start.setStyleSheet("QPushButton { background-color: #00B894; color: white; padding: 12px; border-radius: 6px; } QPushButton:hover { background-color: #00ECB5; }")
        btn_start.clicked.connect(self.start_experiment)
        layout.addWidget(btn_start)

        return widget

    def browse_video_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Video Stimulus Directory", self.video_dir)
        if folder:
            self.video_dir = folder
            self.audio_dir = os.path.abspath(os.path.join(self.video_dir, "audio"))
            self.norm_cache_dir = os.path.abspath(os.path.join(self.audio_dir, "normalized"))
            os.makedirs(self.audio_dir, exist_ok=True)
            os.makedirs(self.norm_cache_dir, exist_ok=True)

            self.lbl_video_dir.setText(folder)
            self.scan_video_directory()

    def find_matching_audio(self, video_path):
        """Look for matching WAV file in videos/audio/ or alongside the video file."""
        if not self.chk_ext_audio.isChecked():
            return None

        base_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Check candidate paths
        candidates = [
            os.path.join(self.audio_dir, f"{base_name}.wav"),
            os.path.join(self.audio_dir, f"{base_name}.mp3"),
            os.path.join(os.path.dirname(video_path), f"{base_name}.wav"),
            os.path.join(os.path.dirname(video_path), f"{base_name}.mp3")
        ]

        for cand in candidates:
            if os.path.exists(cand):
                return cand
        return None

    def scan_video_directory(self):
        target_dir = self.lbl_video_dir.text().strip()
        if not os.path.exists(target_dir):
            self.lbl_scan_status.setText(f"⚠️ Folder path does not exist: {target_dir}")
            self.video_files = []
            return

        valid_exts = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
        self.video_files = [
            os.path.join(target_dir, f) for f in os.listdir(target_dir)
            if f.lower().endswith(valid_exts)
        ]
        
        if self.video_files:
            paired_audio_count = sum(1 for v in self.video_files if self.find_matching_audio(v) is not None)
            msg = f"✅ Found {len(self.video_files)} video file(s)."
            if paired_audio_count > 0:
                msg += f" 🎵 {paired_audio_count} paired WAV audio track(s) detected in videos/audio/."
            else:
                msg += f" (Optional audio tracks can be placed in: {self.audio_dir})"
            
            self.lbl_scan_status.setText(msg)
            self.lbl_scan_status.setStyleSheet("color: #00E676;")
        else:
            self.lbl_scan_status.setText(f"⚠️ No video files (.mp4, .avi, .mov) found in {target_dir}. Please add video files to begin.")
            self.lbl_scan_status.setStyleSheet("color: #FF7675;")

    def create_task_screen(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        # Video Title Header (Title displayed on top of stimulus)
        self.lbl_video_title = QLabel("VIDEO TITLE")
        self.lbl_video_title.setFont(QFont("Arial", 20, QFont.Bold))
        self.lbl_video_title.setAlignment(Qt.AlignCenter)
        self.lbl_video_title.setStyleSheet("color: #FFEAA7; background-color: #191E2A; padding: 10px; border-radius: 6px;")
        layout.addWidget(self.lbl_video_title)

        # Center Display Stack (Video Widget vs. Rest Fixation Screen)
        self.display_stack = QStackedWidget()
        
        # 1. Video Player Widget
        self.video_widget = QVideoWidget()
        self.video_widget.setStyleSheet("background-color: #000000; border-radius: 8px;")
        self.video_player.setVideoOutput(self.video_widget)
        self.display_stack.addWidget(self.video_widget)

        # 2. Rest / Fixation Screen
        self.rest_widget = QWidget()
        rest_layout = QVBoxLayout(self.rest_widget)
        rest_layout.setAlignment(Qt.AlignCenter)

        self.lbl_fixation = QLabel("+")
        self.lbl_fixation.setFont(QFont("Arial", 90, QFont.Bold))
        self.lbl_fixation.setAlignment(Qt.AlignCenter)
        self.lbl_fixation.setStyleSheet("color: #4DEEEA;")

        self.lbl_rest_sub = QLabel("REST (1.5s)")
        self.lbl_rest_sub.setFont(QFont("Arial", 16, QFont.Bold))
        self.lbl_rest_sub.setAlignment(Qt.AlignCenter)
        self.lbl_rest_sub.setStyleSheet("color: #A0A5B5;")

        rest_layout.addWidget(self.lbl_fixation)
        rest_layout.addWidget(self.lbl_rest_sub)
        self.display_stack.addWidget(self.rest_widget)

        layout.addWidget(self.display_stack, stretch=1)

        # Bottom Progress Bar & Info
        bottom_box = QHBoxLayout()
        
        self.lbl_status = QLabel("Trial 0 / 0")
        self.lbl_status.setFont(QFont("Arial", 12, QFont.Bold))
        self.lbl_status.setStyleSheet("color: #00E676;")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(12)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("QProgressBar { background-color: #191E2A; border-radius: 6px; } QProgressBar::chunk { background-color: #4DEEEA; }")

        btn_stop = QPushButton("⏹ STOP EXPERIMENT")
        btn_stop.setStyleSheet("background-color: #D63031; color: white; font-weight: bold; padding: 6px 14px; border-radius: 4px;")
        btn_stop.clicked.connect(self.stop_experiment)

        bottom_box.addWidget(self.lbl_status)
        bottom_box.addWidget(self.progress_bar, stretch=1)
        bottom_box.addWidget(btn_stop)

        layout.addLayout(bottom_box)
        return widget

    def start_experiment(self):
        self.scan_video_directory()
        if not self.video_files:
            QMessageBox.warning(self, "No Videos Found", f"No video files found in:\n{self.lbl_video_dir.text()}\n\nPlease place video files (.mp4, .avi, .mov) in this directory.")
            return

        self.t_video = self.spn_video_dur.value()
        self.t_rest = self.spn_rest_dur.value()
        self.audio_enabled = self.chk_audio.isChecked()
        self.normalize_audio = self.chk_normalize.isChecked()
        self.use_external_audio = self.chk_ext_audio.isChecked()
        self.repetitions = self.spn_reps.value()
        self.randomize_order = self.chk_random.isChecked()

        # Build Trial List and process Audio Normalization if enabled
        base_trials = []
        for v_path in self.video_files:
            audio_path = self.find_matching_audio(v_path)
            
            # Apply Normalization to WAV audio if enabled
            if audio_path and audio_path.lower().endswith('.wav') and self.normalize_audio:
                base_name = os.path.splitext(os.path.basename(audio_path))[0]
                norm_path = os.path.join(self.norm_cache_dir, f"{base_name}_norm.wav")
                audio_path = normalize_wav_file(audio_path, norm_path, target_peak_dB=-1.0)
            
            base_trials.append({'video': v_path, 'audio': audio_path})

        self.trials = []
        for _ in range(self.repetitions):
            v_list = list(base_trials)
            if self.randomize_order:
                random.shuffle(v_list)
            self.trials.extend(v_list)

        self.current_trial_idx = 0
        self.stacked_widget.setCurrentWidget(self.task_screen)
        
        self.send_marker("Video_Dataset_Experiment_Start")
        self.send_marker(f"Audio_Enabled_{self.audio_enabled}")
        self.send_marker(f"Audio_Normalized_{self.normalize_audio}")
        
        # Start initial rest phase (1.5s) before first video
        self.start_rest_phase()

    def start_video_phase(self):
        if self.current_trial_idx >= len(self.trials):
            self.finish_experiment()
            return

        self.current_phase = "VIDEO"
        trial = self.trials[self.current_trial_idx]
        video_path = trial['video']
        audio_path = trial['audio']

        file_name = os.path.splitext(os.path.basename(video_path))[0]

        self.lbl_video_title.setText(f"📹 {file_name}")
        self.lbl_status.setText(f"Trial {self.current_trial_idx + 1} / {len(self.trials)}: Playing Video")
        self.display_stack.setCurrentWidget(self.video_widget)

        # Configure Audio Outputs
        if not self.audio_enabled:
            self.video_audio.setMuted(True)
            self.custom_audio.setMuted(True)
            audio_source_str = "AUDIO_MUTED"
        elif audio_path is not None:
            # Custom external audio track exists
            self.video_audio.setMuted(True)
            self.custom_audio.setMuted(False)
            self.custom_audio_player.setSource(QUrl.fromLocalFile(audio_path))
            audio_source_str = f"EXTERNAL_AUDIO_{os.path.basename(audio_path)}"
        else:
            # Native embedded video audio
            self.video_audio.setMuted(False)
            self.custom_audio.setMuted(True)
            audio_source_str = "NATIVE_VIDEO_AUDIO"

        # Broadcast LSL Markers
        norm_str = "NORM_ACTIVE" if (self.normalize_audio and audio_path) else "NORM_OFF"
        self.send_marker(f"Trial_Start_{self.current_trial_idx + 1}")
        self.send_marker(f"Video_Start_{file_name}_{audio_source_str}_{norm_str}")

        # Start Playback
        self.video_player.setSource(QUrl.fromLocalFile(video_path))
        self.video_player.play()
        if self.audio_enabled and audio_path is not None:
            self.custom_audio_player.play()

        # Schedule timer for 3.0s (or configured video duration)
        self.phase_start_time = time.time()
        self.current_phase_duration = self.t_video
        self.trial_timer.start(int(self.t_video * 1000))
        self.tick_timer.start(30)

    def start_rest_phase(self):
        self.current_phase = "REST"
        self.video_player.stop()
        self.custom_audio_player.stop()

        self.lbl_video_title.setText("REST INTERVAL")
        self.lbl_rest_sub.setText(f"REST ({self.t_rest:.1f}s)")
        self.lbl_status.setText(f"Trial {self.current_trial_idx + 1} / {len(self.trials)}: Rest")
        self.display_stack.setCurrentWidget(self.rest_widget)

        self.send_marker("Rest_Start")

        self.phase_start_time = time.time()
        self.current_phase_duration = self.t_rest
        self.trial_timer.start(int(self.t_rest * 1000))
        self.tick_timer.start(30)

    def advance_trial_phase(self):
        self.tick_timer.stop()
        if self.current_phase == "VIDEO":
            trial = self.trials[self.current_trial_idx]
            file_name = os.path.splitext(os.path.basename(trial['video']))[0]
            
            self.video_player.stop()
            self.custom_audio_player.stop()

            self.send_marker(f"Video_End_{file_name}")
            self.send_marker(f"Trial_End_{self.current_trial_idx + 1}")
            self.current_trial_idx += 1
            self.start_rest_phase()
        elif self.current_phase == "REST":
            self.send_marker("Rest_End")
            if self.current_trial_idx < len(self.trials):
                self.start_video_phase()
            else:
                self.finish_experiment()

    def update_progress(self):
        elapsed = time.time() - self.phase_start_time
        pct = min(100, int((elapsed / max(0.01, self.current_phase_duration)) * 100))
        self.progress_bar.setValue(pct)

    def finish_experiment(self):
        self.trial_timer.stop()
        self.tick_timer.stop()
        self.video_player.stop()
        self.custom_audio_player.stop()
        self.send_marker("Video_Dataset_Experiment_Complete")
        
        QMessageBox.information(self, "Experiment Complete", "All video trials have been completed successfully!")
        self.stacked_widget.setCurrentWidget(self.config_screen)

    def stop_experiment(self):
        self.trial_timer.stop()
        self.tick_timer.stop()
        self.video_player.stop()
        self.custom_audio_player.stop()
        self.send_marker("Video_Dataset_Experiment_Aborted")
        self.stacked_widget.setCurrentWidget(self.config_screen)

    def closeEvent(self, event):
        self.trial_timer.stop()
        self.tick_timer.stop()
        self.video_player.stop()
        self.custom_audio_player.stop()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoDatasetTaskApp()
    window.show()
    sys.exit(app.exec())
