import numpy as np
import matplotlib.pyplot as plt
import os
import wfdb

DATA_DIR = "/Users/adamtheatlantean/Desktop/SeniorProjectHRV/data/mit-bih-arrhythmia-database-1.0.0"
RECORD = "100"

def load_rr_from_record(data_dir: str, record: str, rr_min=0.30, rr_max=2.00):
    """
    Returns:
      rr: RR intervals in seconds (np.ndarray)
      rr_t: time (seconds) at which each RR interval ends (np.ndarray)
      fs: sampling frequency (float)
    """
    record_path = os.path.join(data_dir, record)
    header = wfdb.rdheader(record_path)
    fs = float(header.fs)

    ann = wfdb.rdann(record_path, "atr")
    samples = np.asarray(ann.sample, dtype=np.int64)

    rr = np.diff(samples) / fs
    rr_t = samples[1:] / fs

    # Simple physiologic filter (good default for visualization)
    keep = (rr >= rr_min) & (rr <= rr_max)
    rr = rr[keep]
    rr_t = rr_t[keep]

    return rr, rr_t, fs

rr, rr_t, fs = load_rr_from_record(DATA_DIR, RECORD)

# Use a manageable segment
rr_seg = rr[:2000]  # adjust
rr_seg = (rr_seg - rr_seg.mean()) / (rr_seg.std() + 1e-8)

# Distance matrix
D = np.abs(rr_seg[:, None] - rr_seg[None, :])

# Threshold: choose percentile of distances (controls density)
eps = np.percentile(D, 10)
R = (D <= eps).astype(float)

plt.figure(figsize=(5.0, 5.0), dpi=300)
plt.imshow(R, origin="lower", aspect="equal")
plt.title("Recurrence Plot (RR)")
plt.xlabel("n")
plt.ylabel("m")
plt.tight_layout()
plt.show()
