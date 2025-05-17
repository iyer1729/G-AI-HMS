import os
import pandas as pd
import numpy as np

# Path to input and output folders
input_dir = "/Users/hariiyer/Desktop/G-AI-HMS data/mediapipe_raw/CSV"
output_dir = "/Users/hariiyer/Desktop/G-AI-HMS data/mediapipe_converted_22"
os.makedirs(output_dir, exist_ok=True)

# Define conversion function (reuse from previous message)
def convert_mediapipe_33_to_22(df_33):
    def get_xyz(df, index):
        return df[[f"x_{index}", f"y_{index}", f"z_{index}"]].values

    root = (get_xyz(df_33, 23) + get_xyz(df_33, 24)) / 2
    RH = get_xyz(df_33, 24)
    LH = get_xyz(df_33, 23)
    BP = get_xyz(df_33, 24)
    RK = get_xyz(df_33, 26)
    LK = get_xyz(df_33, 25)
    BT = get_xyz(df_33, 12)
    RMrot = get_xyz(df_33, 30)
    LMrot = get_xyz(df_33, 29)
    BLN = get_xyz(df_33, 12)
    RF = get_xyz(df_33, 28)
    LF = get_xyz(df_33, 27)
    BMN = get_xyz(df_33, 12)
    RSI = get_xyz(df_33, 14)
    LSI = get_xyz(df_33, 13)
    BUN = get_xyz(df_33, 0)
    RS = get_xyz(df_33, 12)
    LS = get_xyz(df_33, 11)
    RE = get_xyz(df_33, 14)
    LE = get_xyz(df_33, 13)
    RW = get_xyz(df_33, 16)
    LW = get_xyz(df_33, 15)

    all_joints = np.hstack([
        root, RH, LH, BP, RK, LK, BT, RMrot, LMrot, BLN,
        RF, LF, BMN, RSI, LSI, BUN, RS, LS, RE, LE, RW, LW
    ])

    headers = [f"{axis}_{i}" for i in range(22) for axis in ["x", "y", "z"]]
    df_22 = pd.DataFrame(all_joints, columns=headers)

    if "image_name" in df_33.columns:
        df_22.insert(0, "image_name", df_33["image_name"].values)

    return df_22

# Process each file
for filename in os.listdir(input_dir):
    if filename.endswith(".csv") and filename.startswith("task_"):
        file_path = os.path.join(input_dir, filename)
        df_33 = pd.read_csv(file_path)
        df_22 = convert_mediapipe_33_to_22(df_33)
        output_path = os.path.join(output_dir, filename)
        df_22.to_csv(output_path, index=False)
        print(f"Converted {filename} → {output_path}")
