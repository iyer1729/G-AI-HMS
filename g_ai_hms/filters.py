import numpy as np
from scipy.signal import butter, filtfilt, medfilt

# apply median and low-pass filter to each joint dimension
def apply_ultra_aggressive_filter(seq, median_kernel=11, butter_order=4, butter_cutoff=0.05):
    T, J, D = seq.shape
    filtered = np.zeros_like(seq)
    b, a = butter(butter_order, butter_cutoff, btype='low')
    for j in range(J):
        for d in range(D):
            med = medfilt(seq[:, j, d], kernel_size=median_kernel)
            filtered[:, j, d] = filtfilt(b, a, med)
    return filtered
