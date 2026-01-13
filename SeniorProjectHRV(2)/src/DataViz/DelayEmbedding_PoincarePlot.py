import matplotlib.pyplot as plt
import numpy as np
import os
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

DATA_DIR = "/data/mit-bih-arrhythmia-database-1.0.0"
RECORD = "100"

rr, rr_t, fs = load_rr_from_record(DATA_DIR, RECORD)

#trim extreme outliers so geometry is readable
lo, hi = np.percentile(rr, [0.5, 99.5])
rr_use = rr[(rr >= lo) & (rr <= hi)]

x = rr_use[:-1]   # RR_{n-1}
y = rr_use[1:]    # RR_n

plt.figure(figsize=(4.8, 4.8), dpi=300)
plt.scatter(x, y, s=6)
plt.xlabel(r"$RR_{n-1}$ (s)")
plt.ylabel(r"$RR_n$ (s)")
plt.title(rf"Delay Embedding / Poincaré Plot (Record: {RECORD})")
plt.grid(alpha=0.25)
plt.gca().set_aspect("equal", adjustable="box")
plt.tight_layout()
plt.show()
