import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Resample a sequence (frames x 66) to target number of frames
def resample_sequence(df, target_frames):
    original_frames = df.shape[0]
    x_old = np.linspace(0, 1, original_frames)
    x_new = np.linspace(0, 1, target_frames)

    resampled = []
    for col in df.columns:
        f = interp1d(x_old, df[col].values, kind='linear', fill_value="extrapolate")
        resampled.append(f(x_new))
    return pd.DataFrame(np.array(resampled).T, columns=df.columns)

base_input_dirs = {
    "gpt": "/Users/hariiyer/Desktop/G-AI-HMS data/gpt_scale_norm",
    "human": "/Users/hariiyer/Desktop/G-AI-HMS data/human_scale_norm",
    "mediapipe": "/Users/hariiyer/Desktop/G-AI-HMS data/mediapipe_scale_norm"
}

base_output_dirs = {
    "gpt": "/Users/hariiyer/Desktop/G-AI-HMS data/gpt_scale_norm_resampled_min",
    "human": "/Users/hariiyer/Desktop/G-AI-HMS data/human_scale_norm_resampled_min",
    "mediapipe": "/Users/hariiyer/Desktop/G-AI-HMS data/mediapipe_scale_norm_resampled_min"
}

# Create output directories
for out_dir in base_output_dirs.values():
    os.makedirs(out_dir, exist_ok=True)

# Resample each task
for i in range(1, 9):
    dfs = {}
    frame_counts = {}

    # Load all 3 versions
    for key in base_input_dirs:
        path = os.path.join(base_input_dirs[key], f"task_{i}.csv")
        df = pd.read_csv(path)
        dfs[key] = df
        frame_counts[key] = df.shape[0]

    max_frames = min(frame_counts.values())

    # Resample and save
    for key in dfs:
        resampled_df = resample_sequence(dfs[key], max_frames)
        out_path = os.path.join(base_output_dirs[key], f"task_{i}.csv")
        resampled_df.to_csv(out_path, index=False)
        print(f"Saved resampled: {out_path} to {max_frames} frames")
