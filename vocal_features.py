

import librosa
import numpy as np
import scipy.stats
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import pandas as pd
import os
import math

import seaborn as sns


def extract_audio_features(wav_path):

    features = {}
    y, sr = librosa.load(wav_path, sr=16000)

    #Pitch
    pitches, mags = librosa.piptrack(y=y, sr=sr)
    pitch_vals = pitches[mags > np.median(mags)]
    features['pitch_mean'] = np.mean(pitch_vals) if len(pitch_vals) else 0
    features['pitch_std'] = np.std(pitch_vals) if len(pitch_vals) else 0

    #Energy
    energy = librosa.feature.rms(y=y)[0]
    features['energy_mean'] = np.mean(energy)
    features['energy_entropy'] = scipy.stats.entropy(energy + 1e-6)

    #MFCC first,third
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=3)
    features['mfcc_1'] = np.mean(mfccs[0])

    features['mfcc_3'] = np.mean(mfccs[2])

    return features





#Violin plot for audio features
# def plot_violin_grid(df, features, label_col='label', plots_per_row=3):
#     num_features = len(features)
#     num_rows = math.ceil(num_features / plots_per_row)
#
#     plt.figure(figsize=(plots_per_row * 5, num_rows * 4))
#
#     for i, feat in enumerate(features):
#         plt.subplot(num_rows, plots_per_row, i + 1)
#         sns.violinplot(x=label_col, y=feat, data=df, inner="quartile")
#         plt.title(f'{feat}')
#         plt.tight_layout()
#
#     plt.show()
#
# audio_folder = "audio_files"
# all_features = []
#
# for fname in os.listdir(audio_folder):
#     if not fname.lower().endswith(".wav"):
#         continue
#
#     wav_path = os.path.join(audio_folder, fname)
#
#     feats = extract_minimal_audio_features(wav_path)
#
#     #Label assignment
#     label = "deepfake" if "audio" in fname.lower() else "real"
#     feats['label'] = label
#     feats['filename'] = fname
#
#     all_features.append(feats)
#
#     print(f"Processed {fname} ({label})")
#
# df = pd.DataFrame(all_features)
#
# features_to_plot = ['pitch_mean', 'pitch_std', 'energy_mean', 'energy_entropy',
#                     'mfcc_1', 'mfcc_2', 'mfcc_3', 'zcr', 'silence_ratio']
#
# plot_violin_grid(df, features_to_plot, plots_per_row=3)
