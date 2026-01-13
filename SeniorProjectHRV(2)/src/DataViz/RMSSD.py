import os
import numpy as np
import wfdb
import matplotlib.pyplot as plt

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

DATA_DIR = "/data/mit-bih-arrhythmia-database-1.0.0"
RECORD = "100"

rr, rr_t, fs = load_rr_from_record(DATA_DIR, RECORD)

diff = np.diff(rr)
sq = diff**2

# Rolling window in beats (e.g., 60 beats ≈ ~1 minute depending on HR)
W = 60
rmssd = np.sqrt(np.convolve(sq, np.ones(W)/W, mode="valid"))

# Align time: rmssd starts at rr_t[W] approximately
t_rmssd = rr_t[1:][W-1:] / 60  # minutes

plt.figure(figsize=(7.0, 3.2), dpi=300)
plt.plot(t_rmssd, rmssd, linewidth=1.0)
plt.xlabel("Time (minutes)")
plt.ylabel("RMSSD (s)")
plt.title(f"Rolling RMSSD (window={W} beats) – Record {RECORD}")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()
