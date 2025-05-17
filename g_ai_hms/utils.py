import pandas as pd
import numpy as np

# load csv and reshape to (frames, 22, 3)
def load_sequence(file_path):
    df = pd.read_csv(file_path)
    return df.values.reshape(-1, 22, 3)
