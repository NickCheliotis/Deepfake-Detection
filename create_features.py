

import matplotlib.pyplot as plt
import pandas as pd
import os
import math
import seaborn as sns
from blink_rate_feature import extract_eye_blink_rate
from expression_entropy import extract_expression_entropy
from heam_motion_var import extract_head_motion_var_yaw_pitch
from mouth_open_ratio import extract_mouth_open_ratio
from vocal_features import extract_audio_features


audio_dir = "audio_files"
video_dir = "video_files"
all_data = []

def extract_visual_features(video_path):

    features = {}

    features['blink_rate'] = extract_eye_blink_rate(video_path)
    features['mouth_open_ratio'] = extract_mouth_open_ratio(video_path)

    yaw_var, pitch_var = extract_head_motion_var_yaw_pitch(video_path)
    features['yaw_var'] = yaw_var
    features['pitch_var'] = pitch_var

    features['expression_entropy'] = extract_expression_entropy(video_path)

    return features



def extract_all_features():

    for fname in os.listdir(audio_dir):
        if not fname.endswith(".wav"):
            continue

        name = os.path.splitext(fname)[0]
        audio_path = os.path.join(audio_dir, fname)
        video_path = os.path.join(video_dir, name + ".mp4")  # or .avi etc

        # Check video exists
        if not os.path.exists(video_path):
            print(f"Missing video for {fname}")
            continue

        # Get features
        audio_features = extract_audio_features(audio_path)
        visual_features = extract_visual_features(video_path)

        # Combine
        sample = {}
        sample.update(audio_features)
        sample.update(visual_features)
        sample['filename'] = name
        sample['label'] = 1 if 'deepfake' in name.lower() else 0

        all_data.append(sample)

    df = pd.DataFrame(all_data)
    df.to_csv("final_feature_dataset.csv", index=False)


extract_all_features()





# def plot_violin_grid(df, features, label_col='label', plots_per_row=3):
#     num_features = len(features)
#     num_rows = math.ceil(num_features / plots_per_row)
#     plt.figure(figsize=(plots_per_row * 5, num_rows * 4))
#
#     for i, feat in enumerate(features):
#         plt.subplot(num_rows, plots_per_row, i + 1)
#         sns.violinplot(x=label_col, y=feat, data=df, inner="quartile")
#         plt.title(feat)
#         plt.tight_layout()
#
#     plt.show()
#
#
#
# video_folder = "video_files"
# all_features = []
#
# for fname in os.listdir(video_folder):
#     if not fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
#         continue
#
#     video_path = os.path.join(video_folder, fname)
#
#     feats = extract_visual_features(video_path)
#
#     # Label based on filename: deepfake if 'audio' in name else real
#     label = "deepfake" if "video" in fname.lower() else "real"
#     feats['label'] = label
#     feats['filename'] = fname
#
#     all_features.append(feats)
#
#     print(f"Processed {fname} ({label})")
#
# df = pd.DataFrame(all_features)
#
# features_to_plot = ['blink_rate', 'mouth_open_ratio', 'yaw_var', 'pitch_var', 'expression_entropy']
#
# plot_violin_grid(df, features_to_plot, label_col='label', plots_per_row=3)


