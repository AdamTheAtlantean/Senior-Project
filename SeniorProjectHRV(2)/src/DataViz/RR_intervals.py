import matplotlib.pyplot as plt
import os
import numpy as np
import wfdb

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

    # Simple physiologic filter 
    keep = (rr >= rr_min) & (rr <= rr_max)
    rr = rr[keep]
    rr_t = rr_t[keep]

    return rr, rr_t, fs

DATA_DIR = "/Users/adamtheatlantean/Desktop/SeniorProjectHRV/data/mit-bih-arrhythmia-database-1.0.0"
RECORD = "100"

rr, rr_t, fs = load_rr_from_record(DATA_DIR, RECORD)

minutes = rr_t / 60
mask = minutes <= 30  # first 30 minutes

plt.figure(figsize=(7.0, 3.2), dpi=300)
plt.plot(minutes[mask], rr[mask], linewidth=0.8)
plt.scatter(minutes[mask], rr[mask], s=6)
plt.xlabel("Time (minutes)")
plt.ylabel("RR interval (s)")
plt.title(f"RR Interval Time Series (Record {RECORD})")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()
