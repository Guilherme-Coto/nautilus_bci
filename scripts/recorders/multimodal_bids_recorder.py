"""
Multimodal BIDS Dataset Recorder (EEG + Smartwatch IMU + Smartwatch PPG + Markers)
===================================================================================

Connects to all active LSL streams on your local network:
  - EEG Stream        (e.g., 'gNautilus' 32-channel EEG)
  - Smartwatch IMU   (e.g., 'Smartwatch_IMU' 6-DOF Accel + Gyro)
  - Smartwatch PPG   (e.g., 'Smartwatch_PPG' Heart Rate BPM)
  - Event Markers    (e.g., 'MotorImageryMarkers')

Saves everything into a standardized Multimodal BIDS Dataset with aligned timestamps.

Usage:
  uv run python multimodal_bids_recorder.py --duration 30 --sub 01 --ses 01 --task leftright
"""
import sys
import os
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
_curr_dir = os.path.dirname(__file__)
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)


import os
import sys
import time
import json
import argparse
import threading
import numpy as np
import pandas as pd

from pylsl import StreamInlet, resolve_streams, local_clock
import mne
import mne_bids
from mne_bids import BIDSPath, write_raw_bids


class MultimodalBIDSRecorder:
    def __init__(self, bids_root="bids_dataset_multimodal"):
        self.bids_root = os.path.abspath(bids_root)
        
        self.inlets = {}
        self.data_buffers = {}
        self.timestamp_buffers = {}
        self.marker_events = []

        self.is_recording = False
        self.recording_thread = None
        self.start_time_lsl = 0.0

    def discover_and_connect_streams(self, timeout=3.0):
        print("=" * 70)
        print(" Discovering Active LSL Streams ".center(70, "="))
        print("=" * 70)
        streams = resolve_streams(wait_time=timeout)
        
        if not streams:
            print("[!] Warning: No active LSL streams detected on local network.")
            return False

        for s in streams:
            stype = s.type().upper()
            sname = s.name()
            print(f"[+] Found Stream: '{sname}' | Type: '{stype}' | Channels: {s.channel_count()} | Rate: {s.nominal_srate()} Hz")

            inlet = StreamInlet(s, max_chunklen=64)
            self.inlets[sname] = {
                'inlet': inlet,
                'info': s,
                'type': stype,
                'srate': s.nominal_srate(),
                'n_ch': s.channel_count()
            }
            self.data_buffers[sname] = []
            self.timestamp_buffers[sname] = []

        return True

    def start_recording(self):
        if not self.inlets:
            print("[-] Cannot start: No LSL streams connected.")
            return

        self.is_recording = True
        self.start_time_lsl = local_clock()

        self.recording_thread = threading.Thread(target=self._record_loop, daemon=True)
        self.recording_thread.start()
        print("\n[START] Multimodal Recording actively capturing all streams...")

    def _record_loop(self):
        while self.is_recording:
            for sname, item in self.inlets.items():
                inlet = item['inlet']
                stype = item['type']
                try:
                    chunk, timestamps = inlet.pull_chunk(timeout=0.05)
                    if timestamps:
                        if stype == 'MARKERS':
                            for m, t in zip(chunk, timestamps):
                                m_str = m[0] if isinstance(m, list) else str(m)
                                self.marker_events.append((t - self.start_time_lsl, m_str))
                        else:
                            self.data_buffers[sname].extend(chunk)
                            self.timestamp_buffers[sname].extend([t - self.start_time_lsl for t in timestamps])
                except Exception as e:
                    pass
            time.sleep(0.005)

    def stop_and_export_bids(self, subject_id="01", session_id="01", task_name="leftright"):
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join()

        print("\n" + "=" * 70)
        print(" Exporting Multimodal BIDS Dataset ".center(70, "="))
        print("=" * 70)

        sub_clean = subject_id.replace("sub-", "")
        ses_clean = session_id.replace("ses-", "")

        # 1. Export EEG Stream via MNE-BIDS if present
        eeg_stream_name = None
        for sname, item in self.inlets.items():
            if item['type'] == 'EEG' or 'gNautilus' in sname:
                eeg_stream_name = sname
                break

        if eeg_stream_name and len(self.data_buffers[eeg_stream_name]) > 0:
            item = self.inlets[eeg_stream_name]
            info_xml = item['info']
            eeg_raw_data = np.array(self.data_buffers[eeg_stream_name]).T
            srate = item['srate'] if item['srate'] > 0 else 250.0

            # Parse XML channel labels
            ch_names = []
            ch_types = []
            ch_child = info_xml.desc().child("channels").child("channel")
            for i in range(item['n_ch']):
                if not ch_child.empty():
                    c_name = ch_child.child_value("label")
                    c_type = ch_child.child_value("type").lower()
                    ch_names.append(c_name if c_name else f"EEG{i+1:03d}")
                    ch_types.append('eeg' if c_type in ['eeg', ''] else 'misc')
                    ch_child = ch_child.next_sibling("channel")
                else:
                    ch_names.append(f"EEG{i+1:03d}")
                    ch_types.append('eeg')

            # Convert uV to V if needed
            if np.max(np.abs(eeg_raw_data)) > 1.0:
                eeg_raw_data = eeg_raw_data * 1e-6

            mne_info = mne.create_info(ch_names=ch_names, sfreq=srate, ch_types=ch_types)
            raw = mne.io.RawArray(eeg_raw_data, mne_info)

            # Add Marker annotations
            if self.marker_events:
                onsets = [t for t, _ in self.marker_events]
                durations = [0.1] * len(onsets)
                descriptions = [m for _, m in self.marker_events]
                annot = mne.Annotations(onset=onsets, duration=durations, description=descriptions)
                raw.set_annotations(annot)

            bids_path = BIDSPath(
                subject=sub_clean,
                session=ses_clean,
                task=task_name,
                datatype="eeg",
                root=self.bids_root
            )
            write_raw_bids(raw, bids_path=bids_path, allow_preload=True, format='BrainVision', overwrite=True, verbose=False)
            print(f"[+] Exported EEG BIDS data: sub-{sub_clean}_ses-{ses_clean}_task-{task_name}_eeg")

        # 2. Export Smartwatch IMU & PPG Stream as BIDS Extensions (TSV files)
        bids_sub_dir = os.path.join(self.bids_root, f"sub-{sub_clean}", f"ses-{ses_clean}")
        os.makedirs(bids_sub_dir, exist_ok=True)

        for sname, item in self.inlets.items():
            if item['type'] in ('IMU', 'PPG') or 'Smartwatch' in sname:
                data = np.array(self.data_buffers[sname])
                timestamps = np.array(self.timestamp_buffers[sname])

                if len(data) > 0:
                    df = pd.DataFrame(data)
                    df.insert(0, 'timestamp_sec', timestamps)
                    
                    category = 'motion' if item['type'] == 'IMU' else 'physio'
                    out_folder = os.path.join(bids_sub_dir, category)
                    os.makedirs(out_folder, exist_ok=True)

                    tsv_filename = f"sub-{sub_clean}_ses-{ses_clean}_task-{task_name}_{category}.tsv"
                    tsv_path = os.path.join(out_folder, tsv_filename)
                    df.to_csv(tsv_path, sep='\t', index=False)

                    # Export metadata JSON
                    json_path = tsv_path.replace('.tsv', '.json')
                    meta = {
                        "SamplingFrequency": item['srate'],
                        "StartTime": self.start_time_lsl,
                        "Columns": ["timestamp_sec"] + [f"ch_{i+1}" for i in range(data.shape[1])]
                    }
                    with open(json_path, 'w') as f:
                        json.dump(meta, f, indent=2)

                    print(f"[+] Exported Smartwatch {item['type']} BIDS data to: {tsv_path}")

        print("=" * 70)
        print(f" Multimodal BIDS Export Complete! Root: {self.bids_root} ".center(70, "="))
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Multimodal BIDS Dataset Recorder")
    parser.add_argument("--duration", type=float, default=15.0, help="Recording duration in seconds")
    parser.add_argument("--sub", type=str, default="01", help="Subject ID")
    parser.add_argument("--ses", type=str, default="01", help="Session ID")
    parser.add_argument("--task", type=str, default="leftright", help="Task Name")
    parser.add_argument("--root", type=str, default="bids_dataset_multimodal", help="BIDS root folder")
    args = parser.parse_args()

    recorder = MultimodalBIDSRecorder(bids_root=args.root)
    connected = recorder.discover_and_connect_streams(timeout=2.0)
    
    if not connected:
        print("\n[-] Please launch your LSL streams first!")
        print("    1. g.Nautilus EEG: uv run python mock_lsl_streamer.py (or gds_to_lsl.py)")
        print("    2. Smartwatch:     uv run python smartwatch_lsl_bridge.py --mode udp --port 5005")
        sys.exit(1)

    recorder.start_recording()
    
    print(f"[*] Recording for {args.duration} seconds... Press Ctrl+C to stop early.")
    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        print("\n[*] Manual stop requested.")

    recorder.stop_and_export_bids(subject_id=args.sub, session_id=args.ses, task_name=args.task)


if __name__ == '__main__':
    main()
