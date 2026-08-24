"""
Tower Defense BCI Recall & Visual Blinking Analysis Studio
============================================================
Analyzes the BIDS Tower Defense dataset (`scripts/bids_tower_defense`), where:
  1. Visual Flicker Phase: Participant looks at a blinking box (`Box start blinking`).
  2. Auditory Recall / Song Imagery Phase: Participant imagines the selected song/element
     (`FIRE`, `WATER`, `WIND`, `ELECTRICITY`) after `Box stop blinking`.

Applies decoding and neuro-statistical algorithms from `scripts/analysis`:
  - Robust Spatial Referencing & Bad Channel Detection (`spatial_filters.py`)
  - Relative Band Power & Spectral Density Extraction (Delta, Theta, Alpha, Beta, Gamma)
  - Riemannian Geometry Decoders (Covariances + Tangent Space Logistic Regression / SVM, MDM)
  - Common Spatial Patterns (One-vs-Rest CSP + Shrinkage LDA)
  - Bandpower feature decoders (Multi-Band PSD + Random Forest / LDA)
  - Deep Learning EEGNet (PyTorch Architecture)
  - Multi-class cross-validation, confusion matrices, and publication-quality plots.
"""

import sys
import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure scripts directory and analysis directory are in path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.abspath(os.path.join(_script_dir, ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import mne
from mne_bids import BIDSPath, read_raw_bids
from mne.decoding import CSP

# Machine Learning Imports
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
import scipy.stats as stats

# Riemannian Geometry
import pyriemann
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Internal module imports from scripts/analysis
from spatial_filters import detect_bad_channels, apply_spatial_filter


# ---------------------------------------------------------
# PyTorch EEGNet Architecture (Lawhern et al., 2018)
# ---------------------------------------------------------
class EEGNet(nn.Module):
    def __init__(self, n_channels=32, n_samples=1000, n_classes=4, F1=8, D=2, F2=16, kernel_length=32, dropout_rate=0.25):
        super(EEGNet, self).__init__()
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout_rate)

        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        )
        self.bn3 = nn.BatchNorm2d(F2)
        self.act2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout_rate)

        self._flatten_dim = None
        self._calculate_flatten_dim(n_channels, n_samples)
        self.classifier = nn.Linear(self._flatten_dim, n_classes)

    def _calculate_flatten_dim(self, n_ch, n_s):
        with torch.no_grad():
            x = torch.zeros(1, 1, n_ch, n_s)
            x = self.drop1(self.pool1(self.act1(self.bn2(self.depthwise(self.bn1(self.conv1(x)))))))
            x = self.drop2(self.pool2(self.act2(self.bn3(self.separable(x)))))
            self._flatten_dim = x.view(1, -1).size(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.separable(x)
        x = self.bn3(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def train_eval_eegnet(X, y, n_classes=4, n_splits=5, epochs=35, lr=0.005, batch_size=16):
    """5-Fold Cross Validation for EEGNet."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    n_epochs_count, n_channels, n_samples = X.shape
    acc_scores = []
    y_preds_all = np.zeros_like(y)

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        t_X_train = torch.tensor(X_train[:, np.newaxis, :, :], dtype=torch.float32)
        t_y_train = torch.tensor(y_train, dtype=torch.long)
        t_X_test = torch.tensor(X_test[:, np.newaxis, :, :], dtype=torch.float32)
        t_y_test = torch.tensor(y_test, dtype=torch.long)

        train_loader = DataLoader(TensorDataset(t_X_train, t_y_train), batch_size=batch_size, shuffle=True)

        model = EEGNet(n_channels=n_channels, n_samples=n_samples, n_classes=n_classes).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for ep in range(epochs):
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(t_X_test.to(device)).argmax(dim=1).cpu().numpy()
            acc = accuracy_score(y_test, preds)
            acc_scores.append(acc)
            y_preds_all[test_idx] = preds

    return np.array(acc_scores), y_preds_all


# ---------------------------------------------------------
# Multi-class One-vs-Rest CSP Feature Extraction
# ---------------------------------------------------------
def extract_ovr_csp_features(X, y, n_components=4):
    """Extracts One-vs-Rest CSP features for multi-class EEG."""
    classes = np.unique(y)
    feature_blocks = []
    for c in classes:
        y_bin = (y == c).astype(int)
        csp = CSP(n_components=n_components, reg='oas', log=True, norm_trace=False)
        feat = csp.fit_transform(X, y_bin)
        feature_blocks.append(feat)
    return np.hstack(feature_blocks)


# ---------------------------------------------------------
# Bandpower Feature Extraction (Welch PSD)
# ---------------------------------------------------------
def extract_bandpower_features(X, sfreq=250.0):
    """
    Computes relative power across 5 classical EEG bands:
    Delta (1-4 Hz), Theta (4-8 Hz), Alpha (8-12 Hz), Beta (13-30 Hz), Gamma (30-45 Hz)
    """
    n_epochs, n_ch, n_samples = X.shape
    bands = {
        'delta': (1.0, 4.0),
        'theta': (4.0, 8.0),
        'alpha': (8.0, 12.0),
        'beta': (13.0, 30.0),
        'gamma': (30.0, 45.0)
    }
    
    n_per_seg = min(int(sfreq * 1.5), n_samples)
    from scipy.signal import welch
    freqs, psd = welch(X, fs=sfreq, nperseg=n_per_seg, axis=-1)
    
    total_power = np.sum(psd, axis=-1, keepdims=True) + 1e-12
    rel_psd = psd / total_power
    
    feats = []
    for (fmin, fmax) in bands.values():
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        band_p = np.mean(rel_psd[:, :, idx], axis=-1)
        feats.append(band_p)
        
    return np.hstack(feats)


# ---------------------------------------------------------
# Main Analysis Pipeline
# ---------------------------------------------------------
def run_tower_defense_recall_analysis(
    bids_root="scripts/bids_tower_defense",
    subject_id="01",
    session_id="01",
    out_dir="scripts/analysis_results/tower_defense_recall",
    recall_tmin=0.5,
    recall_tmax=4.5,
    blink_tmin=0.5,
    blink_tmax=4.5,
    spatial_filter="robust_car"
):
    print("=" * 80)
    print(" BCI TOWER DEFENSE: SONG RECALL & BLINKING IMAGERY ANALYSIS ".center(80, "="))
    print("=" * 80)

    bids_root = os.path.abspath(bids_root)
    if not os.path.exists(bids_root):
        raise FileNotFoundError(f"BIDS root directory not found: {bids_root}")

    os.makedirs(out_dir, exist_ok=True)
    sub_clean = subject_id.replace("sub-", "")
    ses_clean = session_id.replace("ses-", "")

    bids_path = BIDSPath(
        subject=sub_clean,
        session=ses_clean,
        task="recall",
        datatype="eeg",
        root=bids_root
    )

    print(f"[*] Reading BIDS dataset from: {bids_path.directory}")
    raw = read_raw_bids(bids_path=bids_path, verbose=False)
    raw.load_data()

    # Channel handling & units
    misc_chans = [ch for ch in raw.ch_names if ch.upper() in ['BATTERY', 'STATUS', 'AUX'] or ch == 'EEG033']
    if misc_chans:
        raw.set_channel_types({ch: 'misc' for ch in misc_chans})
    
    # Keep only active EEG channels
    raw.pick('eeg')
    
    sfreq = raw.info['sfreq']
    duration_s = raw.times[-1]
    ch_names = raw.ch_names
    print(f"[+] Loaded raw dataset: {sfreq} Hz | {len(ch_names)} active EEG channels | {duration_s:.1f}s duration")

    # Highpass baseline detrending for screening
    data_raw_arr = raw.get_data().T  # (n_samples, n_channels)
    # Estimate standard deviations and scale
    stds = np.std(data_raw_arr, axis=0)
    is_volts = np.mean(stds) < 1e-3
    scale_factor = 1e6 if is_volts else 1.0
    data_uv = (data_raw_arr - np.mean(data_raw_arr, axis=0, keepdims=True)) * scale_factor

    bad_idx, bad_status = detect_bad_channels(data_uv, ch_names=ch_names)
    good_ch_names = [ch_names[i] for i in range(len(ch_names)) if i not in bad_idx]
    print(f"[+] Bad Channel Screening: {len(bad_idx)} bad channels detected: {[ch_names[i] for i in bad_idx]}")
    print(f"[+] Clean Channel Count: {len(good_ch_names)} / {len(ch_names)}")

    if spatial_filter != "none":
        print(f"[*] Applying spatial filter: '{spatial_filter}' referencing...")
        filt_data_uv = apply_spatial_filter(data_uv, ch_names, mode=spatial_filter)
        raw._data = (filt_data_uv / scale_factor).T

    # Bandpass filter (1.0 to 45.0 Hz) and 50 Hz Notch
    print("[*] Applying Bandpass Filter (1.0 - 45.0 Hz) & 50 Hz Notch filter...")
    raw_filt = raw.copy().filter(l_freq=1.0, h_freq=45.0, verbose=False)
    raw_filt.notch_filter(freqs=50.0, verbose=False)

    # Parse Events TSV
    events_tsv_path = os.path.join(bids_path.directory, f"sub-{sub_clean}_ses-{ses_clean}_task-recall_events.tsv")
    df_events = pd.read_csv(events_tsv_path, sep='\t')
    
    # Class dictionary
    class_map = {'FIRE': 0, 'WATER': 1, 'WIND': 2, 'ELECTRICITY': 3}
    class_names = ['FIRE', 'WATER', 'WIND', 'ELECTRICITY']

    # Match Box stop blinking with element selections
    recall_events = []
    blink_events = []
    
    events_list = df_events.to_dict('records')
    for i, ev in enumerate(events_list):
        if ev['trial_type'] == 'Box stop blinking':
            label_str = None
            # Search nearby for selected element
            for j in range(max(0, i - 2), min(len(events_list), i + 3)):
                tt = str(events_list[j]['trial_type'])
                if 'selected' in tt:
                    if abs(events_list[j]['onset'] - ev['onset']) < 0.2:
                        label_str = tt.replace(' selected', '').strip()
                        break
            
            if label_str in class_map:
                class_id = class_map[label_str]
                sample_idx = int(ev['sample']) if 'sample' in ev and not np.isnan(ev['sample']) else int(ev['onset'] * sfreq)
                recall_events.append([sample_idx, 0, class_id])

                # Check preceding blinking onset
                if i > 0 and events_list[i-1]['trial_type'] == 'Box start blinking':
                    b_sample = int(events_list[i-1]['sample']) if 'sample' in events_list[i-1] and not np.isnan(events_list[i-1]['sample']) else int(events_list[i-1]['onset'] * sfreq)
                    blink_events.append([b_sample, 0, class_id])

    recall_events = np.array(recall_events)
    blink_events = np.array(blink_events)

    print(f"\n[+] Identified Trials:")
    print(f"    - Song Recall (Imagination) Trials: {len(recall_events)}")
    for c_name, c_id in class_map.items():
        print(f"      • {c_name:12s}: {np.sum(recall_events[:, 2] == c_id)} trials")
    print(f"    - Box Blinking (Visual Cue) Trials: {len(blink_events)}")

    # ---------------------------------------------------------
    # Epoching: Recall / Imagery vs Blinking
    # ---------------------------------------------------------
    event_id_dict = {k: v for k, v in class_map.items()}

    print(f"\n[*] Epoching Song Recall Phase (t = {recall_tmin}s to {recall_tmax}s after Box Stop Blinking)...")
    epochs_recall = mne.Epochs(
        raw_filt,
        recall_events,
        event_id=event_id_dict,
        tmin=recall_tmin,
        tmax=recall_tmax,
        baseline=None,
        preload=True,
        verbose=False
    )

    print(f"[*] Epoching Visual Blinking Phase (t = {blink_tmin}s to {blink_tmax}s after Box Start Blinking)...")
    epochs_blink = mne.Epochs(
        raw_filt,
        blink_events,
        event_id=event_id_dict,
        tmin=blink_tmin,
        tmax=blink_tmax,
        baseline=None,
        preload=True,
        verbose=False
    )

    X_recall = epochs_recall.get_data()  # (n_epochs, n_channels, n_times)
    y_recall = epochs_recall.events[:, 2]

    X_blink = epochs_blink.get_data()
    y_blink = epochs_blink.events[:, 2]

    # Convert to uV for numerical stability across sklearn / pyriemann / pytorch
    X_recall_uv = X_recall * 1e6
    X_blink_uv = X_blink * 1e6

    # ---------------------------------------------------------
    # 1. Spectral Analysis & Power Spectral Densities
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print(" 1. Spectral Power Density (PSD) & Rhythms Analysis ".center(60, "="))
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    palette = {'FIRE': '#FF5722', 'WATER': '#2196F3', 'WIND': '#4CAF50', 'ELECTRICITY': '#FFC107'}

    for idx, (ep_obj, title, ax) in enumerate([(epochs_blink, "Visual Blinking Phase (SSVEP / Cue)", axes[0]),
                                              (epochs_recall, "Auditory Recall / Song Imagery Phase", axes[1])]):
        for c_name in class_names:
            sub_ep = ep_obj[c_name]
            psd_obj = sub_ep.compute_psd(fmin=2.0, fmax=45.0, verbose=False)
            psds, freqs = psd_obj.get_data(return_freqs=True)
            mean_psd = np.mean(psds, axis=(0, 1)) * 1e12  # scale uV^2 / Hz
            ax.plot(freqs, mean_psd, label=c_name, color=palette[c_name], linewidth=2.2)

        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Frequency (Hz)', fontsize=11)
        ax.set_ylabel('Power Spectral Density (µV²/Hz)', fontsize=11)
        ax.axvspan(8.0, 12.0, color='gold', alpha=0.15, label='Alpha (8-12 Hz)' if idx == 0 else "")
        ax.axvspan(13.0, 30.0, color='cyan', alpha=0.10, label='Beta (13-30 Hz)' if idx == 0 else "")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right')

    plt.suptitle(f"Power Spectral Density Comparison: Blinking vs Song Recall (Sub-{sub_clean})", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    psd_plot_path = os.path.join(out_dir, "psd_spectral_comparison.png")
    plt.savefig(psd_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved PSD comparison plot: {psd_plot_path}")

    # ---------------------------------------------------------
    # 2. Multi-Model Decoding Suite on Song Recall
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print(" 2. Neural Decoding Benchmark Suite on Song Recall ".center(60, "="))
    print("=" * 60)

    models_results = {}
    confusion_matrices = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Model 1: Riemannian Covariances + Tangent Space Logistic Regression
    pipe_ts_lr = Pipeline([
        ('cov', Covariances(estimator='oas')),
        ('ts', TangentSpace(metric='riemann')),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, C=1.0))
    ])
    scores_ts_lr = cross_val_score(pipe_ts_lr, X_recall_uv, y_recall, cv=cv, scoring='accuracy')
    preds_ts_lr = cross_val_predict(pipe_ts_lr, X_recall_uv, y_recall, cv=cv)
    models_results['Riemannian TS + LogReg'] = {
        'accuracy_mean': float(np.mean(scores_ts_lr)),
        'accuracy_std': float(np.std(scores_ts_lr)),
        'balanced_acc': float(balanced_accuracy_score(y_recall, preds_ts_lr)),
        'f1_macro': float(f1_score(y_recall, preds_ts_lr, average='macro'))
    }
    confusion_matrices['Riemannian TS + LogReg'] = confusion_matrix(y_recall, preds_ts_lr)
    print(f"[+] [1/6] Riemannian Tangent Space (LogReg):     Acc = {np.mean(scores_ts_lr)*100:5.2f}% ± {np.std(scores_ts_lr)*100:4.2f}% | F1 = {f1_score(y_recall, preds_ts_lr, average='macro'):.3f}")

    # Model 2: Riemannian Covariances + Tangent Space SVM (RBF)
    pipe_ts_svm = Pipeline([
        ('cov', Covariances(estimator='oas')),
        ('ts', TangentSpace(metric='riemann')),
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', C=1.0))
    ])
    scores_ts_svm = cross_val_score(pipe_ts_svm, X_recall_uv, y_recall, cv=cv, scoring='accuracy')
    preds_ts_svm = cross_val_predict(pipe_ts_svm, X_recall_uv, y_recall, cv=cv)
    models_results['Riemannian TS + SVM (RBF)'] = {
        'accuracy_mean': float(np.mean(scores_ts_svm)),
        'accuracy_std': float(np.std(scores_ts_svm)),
        'balanced_acc': float(balanced_accuracy_score(y_recall, preds_ts_svm)),
        'f1_macro': float(f1_score(y_recall, preds_ts_svm, average='macro'))
    }
    confusion_matrices['Riemannian TS + SVM (RBF)'] = confusion_matrix(y_recall, preds_ts_svm)
    print(f"[+] [2/6] Riemannian Tangent Space (SVM RBF):    Acc = {np.mean(scores_ts_svm)*100:5.2f}% ± {np.std(scores_ts_svm)*100:4.2f}% | F1 = {f1_score(y_recall, preds_ts_svm, average='macro'):.3f}")

    # Model 3: Riemannian Minimum Distance to Mean (MDM)
    pipe_mdm = Pipeline([
        ('cov', Covariances(estimator='oas')),
        ('clf', MDM(metric='riemann'))
    ])
    scores_mdm = cross_val_score(pipe_mdm, X_recall_uv, y_recall, cv=cv, scoring='accuracy')
    preds_mdm = cross_val_predict(pipe_mdm, X_recall_uv, y_recall, cv=cv)
    models_results['Riemannian MDM'] = {
        'accuracy_mean': float(np.mean(scores_mdm)),
        'accuracy_std': float(np.std(scores_mdm)),
        'balanced_acc': float(balanced_accuracy_score(y_recall, preds_mdm)),
        'f1_macro': float(f1_score(y_recall, preds_mdm, average='macro'))
    }
    confusion_matrices['Riemannian MDM'] = confusion_matrix(y_recall, preds_mdm)
    print(f"[+] [3/6] Riemannian MDM Classifier:             Acc = {np.mean(scores_mdm)*100:5.2f}% ± {np.std(scores_mdm)*100:4.2f}% | F1 = {f1_score(y_recall, preds_mdm, average='macro'):.3f}")

    # Model 4: Multi-Class One-vs-Rest CSP + Shrinkage LDA
    print("[*] Extracting Multi-Class CSP Log-Variance Features (8-30 Hz)...")
    raw_mubeta = raw.copy().filter(l_freq=8.0, h_freq=30.0, verbose=False)
    epochs_mubeta = mne.Epochs(raw_mubeta, recall_events, event_id=event_id_dict, tmin=recall_tmin, tmax=recall_tmax, baseline=None, preload=True, verbose=False)
    X_mubeta = epochs_mubeta.get_data() * 1e6

    X_csp_feats = extract_ovr_csp_features(X_mubeta, y_recall, n_components=4)
    clf_lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    scores_csp_lda = cross_val_score(clf_lda, X_csp_feats, y_recall, cv=cv, scoring='accuracy')
    preds_csp_lda = cross_val_predict(clf_lda, X_csp_feats, y_recall, cv=cv)
    models_results['One-vs-Rest CSP + LDA'] = {
        'accuracy_mean': float(np.mean(scores_csp_lda)),
        'accuracy_std': float(np.std(scores_csp_lda)),
        'balanced_acc': float(balanced_accuracy_score(y_recall, preds_csp_lda)),
        'f1_macro': float(f1_score(y_recall, preds_csp_lda, average='macro'))
    }
    confusion_matrices['One-vs-Rest CSP + LDA'] = confusion_matrix(y_recall, preds_csp_lda)
    print(f"[+] [4/6] One-vs-Rest CSP + Shrinkage LDA:       Acc = {np.mean(scores_csp_lda)*100:5.2f}% ± {np.std(scores_csp_lda)*100:4.2f}% | F1 = {f1_score(y_recall, preds_csp_lda, average='macro'):.3f}")

    # Model 5: Multi-Band PSD Features + Random Forest
    print("[*] Computing Multi-Band Spectral Features (Delta/Theta/Alpha/Beta/Gamma)...")
    X_bandpower = extract_bandpower_features(X_recall_uv, sfreq=sfreq)
    pipe_rf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=150, random_state=42))
    ])
    scores_rf = cross_val_score(pipe_rf, X_bandpower, y_recall, cv=cv, scoring='accuracy')
    preds_rf = cross_val_predict(pipe_rf, X_bandpower, y_recall, cv=cv)
    models_results['Bandpower + Random Forest'] = {
        'accuracy_mean': float(np.mean(scores_rf)),
        'accuracy_std': float(np.std(scores_rf)),
        'balanced_acc': float(balanced_accuracy_score(y_recall, preds_rf)),
        'f1_macro': float(f1_score(y_recall, preds_rf, average='macro'))
    }
    confusion_matrices['Bandpower + Random Forest'] = confusion_matrix(y_recall, preds_rf)
    print(f"[+] [5/6] Multi-Band Power + Random Forest:      Acc = {np.mean(scores_rf)*100:5.2f}% ± {np.std(scores_rf)*100:4.2f}% | F1 = {f1_score(y_recall, preds_rf, average='macro'):.3f}")

    # Model 6: PyTorch Deep Learning (EEGNet)
    print("[*] Training PyTorch EEGNet Deep Neural Network...")
    scores_eegnet, preds_eegnet = train_eval_eegnet(X_recall_uv, y_recall, n_classes=4, n_splits=5, epochs=35)
    models_results['PyTorch EEGNet DL'] = {
        'accuracy_mean': float(np.mean(scores_eegnet)),
        'accuracy_std': float(np.std(scores_eegnet)),
        'balanced_acc': float(balanced_accuracy_score(y_recall, preds_eegnet)),
        'f1_macro': float(f1_score(y_recall, preds_eegnet, average='macro'))
    }
    confusion_matrices['PyTorch EEGNet DL'] = confusion_matrix(y_recall, preds_eegnet)
    print(f"[+] [6/6] PyTorch EEGNet Deep Learning:          Acc = {np.mean(scores_eegnet)*100:5.2f}% ± {np.std(scores_eegnet)*100:4.2f}% | F1 = {f1_score(y_recall, preds_eegnet, average='macro'):.3f}")

    # ---------------------------------------------------------
    # 3. Benchmark Visualization & Export
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print(" 3. Visualizing Performance & Confusion Matrices ".center(60, "="))
    print("=" * 60)

    # Plot 1: Decoding Accuracy Comparison Bar Chart
    df_res = pd.DataFrame(models_results).T.reset_index()
    df_res.rename(columns={'index': 'Model'}, inplace=True)
    df_res['accuracy_pct'] = df_res['accuracy_mean'] * 100
    df_res['std_pct'] = df_res['accuracy_std'] * 100

    plt.figure(figsize=(12, 6))
    colors = ['#3498db', '#2ecc71', '#9b59b6', '#e67e22', '#1abc9c', '#e74c3c']
    bars = plt.bar(df_res['Model'], df_res['accuracy_pct'], yerr=df_res['std_pct'], capsize=6, color=colors, alpha=0.85, edgecolor='black')
    plt.axhline(25.0, color='red', linestyle='--', linewidth=2, label='Chance Level (25.0%)')

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 1.5, f'{h:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.ylabel('Classification Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title(f'4-Class Song Recall Neural Decoding Accuracy (sub-{sub_clean})', fontsize=14, fontweight='bold', pad=15)
    plt.xticks(rotation=20, ha='right', fontsize=11)
    plt.ylim(0, max(60.0, df_res['accuracy_pct'].max() + 15.0))
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    decoding_plot_path = os.path.join(out_dir, "decoding_benchmark_accuracy.png")
    plt.savefig(decoding_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved Decoding Accuracy benchmark: {decoding_plot_path}")

    # Plot 2: Confusion Matrices
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    for idx, (m_name, cm) in enumerate(confusion_matrices.items()):
        ax = axes[idx]
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, fontsize=10)
        ax.set_yticklabels(class_names, fontsize=10)
        for row in range(len(class_names)):
            for col in range(len(class_names)):
                val = cm_norm[row, col]
                color = 'white' if val > 0.5 else 'black'
                ax.text(col, row, f'{val:.2f}', ha='center', va='center', color=color, fontweight='bold', fontsize=11)

        ax.set_title(f"{m_name}\nAcc: {models_results[m_name]['accuracy_mean']*100:.1f}% | F1: {models_results[m_name]['f1_macro']:.2f}", fontsize=11, fontweight='bold')
        ax.set_ylabel('True Song Class', fontsize=10)
        ax.set_xlabel('Predicted Song Class', fontsize=10)

    plt.suptitle("4-Class Song Recall Confusion Matrices across Neural Decoders", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    cm_plot_path = os.path.join(out_dir, "confusion_matrices_grid.png")
    plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved Confusion Matrices Grid: {cm_plot_path}")

    # Export Summary JSON & CSV
    summary_data = {
        'dataset': 'bids_tower_defense',
        'subject': sub_clean,
        'session': ses_clean,
        'task': 'recall',
        'total_trials': len(recall_events),
        'class_distribution': {c_name: int(np.sum(recall_events[:, 2] == c_id)) for c_name, c_id in class_map.items()},
        'recall_epoch_window': [recall_tmin, recall_tmax],
        'blinking_epoch_window': [blink_tmin, blink_tmax],
        'spatial_filter_applied': spatial_filter,
        'models_benchmark': models_results
    }

    json_path = os.path.join(out_dir, "recall_decoding_summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary_data, f, indent=4)

    csv_path = os.path.join(out_dir, "models_benchmark_metrics.csv")
    df_res.to_csv(csv_path, index=False)

    print(f"[+] Exported results JSON: {json_path}")
    print(f"[+] Exported metrics CSV:  {csv_path}")

    print("\n" + "=" * 80)
    print(" ANALYSIS COMPLETE ".center(80, "="))
    print("=" * 80)
    return summary_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze BIDS Tower Defense Recall & Blinking Dataset")
    parser.add_argument("--bids-root", type=str, default="scripts/bids_tower_defense", help="Path to bids_tower_defense folder")
    parser.add_argument("--sub", type=str, default="01", help="Subject ID (e.g. 01)")
    parser.add_argument("--ses", type=str, default="01", help="Session ID (e.g. 01)")
    parser.add_argument("--out-dir", type=str, default="scripts/analysis_results/tower_defense_recall", help="Output directory")
    parser.add_argument("--recall-tmin", type=float, default=0.5, help="Recall epoch start relative to box stop blinking (s)")
    parser.add_argument("--recall-tmax", type=float, default=4.5, help="Recall epoch end relative to box stop blinking (s)")
    parser.add_argument("--blink-tmin", type=float, default=0.5, help="Blink epoch start relative to box start blinking (s)")
    parser.add_argument("--blink-tmax", type=float, default=4.5, help="Blink epoch end relative to box start blinking (s)")
    parser.add_argument("--spatial-filter", type=str, default="robust_car", choices=["none", "robust_car", "car", "laplacian"], help="Spatial filter mode")

    args = parser.parse_args()

    run_tower_defense_recall_analysis(
        bids_root=args.bids_root,
        subject_id=args.sub,
        session_id=args.ses,
        out_dir=args.out_dir,
        recall_tmin=args.recall_tmin,
        recall_tmax=args.recall_tmax,
        blink_tmin=args.blink_tmin,
        blink_tmax=args.blink_tmax,
        spatial_filter=args.spatial_filter
    )
