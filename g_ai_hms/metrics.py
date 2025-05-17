import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import cosine

# mean per-joint error across all frames
def compute_mpjpe(seq1, seq2):
    return np.mean(np.linalg.norm(seq1 - seq2, axis=-1))

# rigid-aligned mean per-joint error
def compute_pa_mpjpe(seq1, seq2):
    aligned = []
    for i in range(seq1.shape[0]):
        A, B = seq1[i], seq2[i]
        mu_A, mu_B = A.mean(axis=0), B.mean(axis=0)
        A_hat, B_hat = A - mu_A, B - mu_B
        U, _, Vt = np.linalg.svd(B_hat.T @ A_hat)
        R = U @ Vt
        B_aligned = (B_hat @ R.T) + mu_A
        aligned.append(np.mean(np.linalg.norm(A - B_aligned, axis=1)))
    return np.mean(aligned)

# DTW distance per joint
def compute_dtw(seq1, seq2):
    return np.mean([fastdtw(seq1[:, j, :], seq2[:, j, :])[0] for j in range(22)])

# cosine similarity over entire sequence
def compute_cosine_similarity(seq1, seq2):
    return 1 - cosine(seq1.flatten(), seq2.flatten())

def compute_mpjpe_per_joint(seq1, seq2):
    return np.mean(np.linalg.norm(seq1 - seq2, axis=-1), axis=0)

def compute_pa_mpjpe_per_joint(seq1, seq2):
    joint_errors = np.zeros(22)
    for i in range(seq1.shape[0]):
        A, B = seq1[i], seq2[i]
        mu_A, mu_B = A.mean(axis=0), B.mean(axis=0)
        A_hat, B_hat = A - mu_A, B - mu_B
        U, _, Vt = np.linalg.svd(B_hat.T @ A_hat)
        R = U @ Vt
        B_aligned = (B_hat @ R.T) + mu_A
        joint_errors += np.linalg.norm(A - B_aligned, axis=1)
    return joint_errors / seq1.shape[0]

def compute_dtw_per_joint(seq1, seq2):
    return [fastdtw(seq1[:, j, :], seq2[:, j, :])[0] for j in range(22)]
