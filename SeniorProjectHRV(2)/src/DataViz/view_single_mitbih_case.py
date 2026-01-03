"""
Plot channel MLII from MIT-BIH record 100 with a classic PQRST-looking window.
Requires: wfdb, numpy, matplotlib
Run: python3.12 plot_pqrst_record100.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import wfdb

# Settings
ekg_folder = "/Users/adamtheatlantean/Desktop/SeniorProjectHRV/data/mit-bih-arrhythmia-database-1.0.0"  # folder containing 100.hea/.dat/.atr
record = "100"

START_SEC = 0.0       # 0, 60, 120, etc.
DURATION_SEC = 2.5    # 1.5–3.0 seconds is usually perfect
LEAD_NAME = "MLII"    # record 100 has MLII and V5

#--------------------------------------------------

record_path = os.path.join(ekg_folder, record)

# Load header first to get sampling frequency
header = wfdb.rdheader(record_path)
fs = header.fs

sampfrom = int(START_SEC * fs)
sampto = int((START_SEC + DURATION_SEC) * fs)

# Read only the window needed
rec = wfdb.rdrecord(record_path, sampfrom=sampfrom, sampto=sampto)

# Find which channel index corresponds to MLII
sig_names = rec.sig_name
if LEAD_NAME not in sig_names:
    raise ValueError(f"{LEAD_NAME} not found. Available leads: {sig_names}")

lead_idx = sig_names.index(LEAD_NAME)
y = rec.p_signal[:, lead_idx]
t = np.arange(len(y)) / fs + START_SEC

plt.figure(figsize=(9, 3))
plt.plot(t, y, linewidth=1.0)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")
plt.title(f"MIT-BIH record {record} – {LEAD_NAME} ({START_SEC:.1f}–{START_SEC + DURATION_SEC:.1f}s)")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()
