"""
config.py
---------
Central configuration for Experiment 12 (IIR filter design).

Every tunable constant lives here so that the processing, plotting and GUI
modules never hard-code values.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

# --------------------------------------------------------------------------
# Dataset: PhysioNet EEG Motor Movement/Imagery Database, subject S001
# --------------------------------------------------------------------------
BASE = "https://physionet.org/files/eegmmidb/1.0.0/S001/"
RUNS = {"open": "S001R01.edf", "closed": "S001R02.edf"}   # 1-minute baselines, eyes open / eyes closed
CHANNEL = "O1"                       # left occipital electrode, over the visual cortex
FS = 160.0                           # Hz
MAINS_HZ = 60.0                      # recorded in the USA

# --------------------------------------------------------------------------
# Filter specifications
# --------------------------------------------------------------------------
ALPHA_BAND = (8.0, 13.0)             # Hz
ORDER = 4                            # prototype order (band-pass order is 2 x 4 = 8)
CHEBY_RIPPLE_DB = 1.0
ELLIP_STOP_DB = 40.0
NOTCH_Q = 30.0                       # notch quality factor: bandwidth = f0 / Q = 2 Hz
HIGHPASS_HZ = 0.5                    # removes electrode drift
FAMILIES = ["butterworth", "chebyshev1", "elliptic"]

# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
WELCH_SECONDS = 4                    # PSD segment length
EPOCH_SECONDS = 2                    # epochs for the alpha-power comparison
