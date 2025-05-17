import os
import numpy as np
from filters import apply_ultra_aggressive_filter
from metrics import *
from utils import load_sequence

base_dirs = {
    "mediapipe": "/Users/hariiyer/Desktop/Hari Iyer/final final G-AI-HMS/data/mediapipe_scale_norm_resampled_min/",
    "human": "/Users/hariiyer/Desktop/Hari Iyer/final final G-AI-HMS/data/human_scale_norm_resampled_min/",
    "gpt": "/Users/hariiyer/Desktop/Hari Iyer/final final G-AI-HMS/data/gpt_scale_norm_resampled_min/"
}

joint_mpjpe_gpt, joint_mpjpe_human = [], []
joint_pa_mpjpe_gpt, joint_pa_mpjpe_human = [], []
joint_dtw_gpt, joint_dtw_human = [], []

print(f"{'Task':<8} {'Metric':<20} {'GPT vs Mediapipe':<20} {'Human vs Mediapipe':<20}")
print("-" * 70)

for i in range(1, 9):
    m = load_sequence(os.path.join(base_dirs["mediapipe"], f"task_{i}.csv"))
    g = apply_ultra_aggressive_filter(load_sequence(os.path.join(base_dirs["gpt"], f"task_{i}.csv")))
    h = apply_ultra_aggressive_filter(load_sequence(os.path.join(base_dirs["human"], f"task_{i}.csv")))

    results = {
        "MPJPE": (compute_mpjpe(g, m), compute_mpjpe(h, m)),
        "PA-MPJPE": (compute_pa_mpjpe(g, m), compute_pa_mpjpe(h, m)),
        "DTW": (compute_dtw(g, m), compute_dtw(h, m))
    }

    for metric, (gv, hv) in results.items():
        print(f"task_{i:<6} {metric:<20} {gv:<20.4f} {hv:<20.4f}")

    joint_mpjpe_gpt.append(compute_mpjpe_per_joint(g, m))
    joint_mpjpe_human.append(compute_mpjpe_per_joint(h, m))
    joint_pa_mpjpe_gpt.append(compute_pa_mpjpe_per_joint(g, m))
    joint_pa_mpjpe_human.append(compute_pa_mpjpe_per_joint(h, m))
    joint_dtw_gpt.append(compute_dtw_per_joint(g, m))
    joint_dtw_human.append(compute_dtw_per_joint(h, m))

    print(f"\nJoint-wise metrics for task_{i}:")
    print(f"{'Joint':<6} {'MPJPE (G)':<12} {'MPJPE (H)':<12} {'PA-MPJPE (G)':<14} {'PA-MPJPE (H)':<14} {'DTW (G)':<12} {'DTW (H)':<12}")
    for j in range(22):
        print(f"{j:<6} {joint_mpjpe_gpt[-1][j]:<12.4f} {joint_mpjpe_human[-1][j]:<12.4f} "
              f"{joint_pa_mpjpe_gpt[-1][j]:<14.4f} {joint_pa_mpjpe_human[-1][j]:<14.4f} "
              f"{joint_dtw_gpt[-1][j]:<12.2f} {joint_dtw_human[-1][j]:<12.2f}")

# average across tasks
avg = lambda lst: np.mean(np.vstack(lst), axis=0)
print("\nJoint-wise Averages Across All Tasks:")
print(f"{'Joint':<6} {'Avg MPJPE (G)':<15} {'Avg MPJPE (H)':<15} {'Avg PA-MPJPE (G)':<18} {'Avg PA-MPJPE (H)':<18} {'Avg DTW (G)':<15} {'Avg DTW (H)':<15}")
for j in range(22):
    print(f"{j:<6} {avg(joint_mpjpe_gpt)[j]:<15.4f} {avg(joint_mpjpe_human)[j]:<15.4f} "
          f"{avg(joint_pa_mpjpe_gpt)[j]:<18.4f} {avg(joint_pa_mpjpe_human)[j]:<18.4f} "
          f"{avg(joint_dtw_gpt)[j]:<15.2f} {avg(joint_dtw_human)[j]:<15.2f}")
