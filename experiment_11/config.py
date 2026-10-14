"""
config.py
---------
Central configuration for Experiment 11 (FIR filter design by windowing).

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
# Dataset: BIDMC PPG and Respiration Dataset (PhysioNet), record bidmc01
# --------------------------------------------------------------------------
RECORD = "bidmc01"
BASE = "https://physionet.org/files/bidmc/1.0.0/"
FILES = {
    "dat": f"{RECORD}.dat",                        # 5 signals, WFDB format 16 (int16, interleaved)
    "hea": f"{RECORD}.hea",                        # header: gains, baselines, signal names
    "numerics": "bidmc_csv/bidmc_01_Numerics.csv",  # monitor HR, PULSE, RESP, SpO2 every second
    "breaths": "bidmc_csv/bidmc_01_Breaths.csv",    # breaths annotated manually by two annotators
}
FS = 125.0                                          # Hz
PPG_INDEX = 1                                       # PLETH is the second signal in the record

# --------------------------------------------------------------------------
# Filter specifications
# --------------------------------------------------------------------------
PULSE_BAND = (0.5, 4.0)        # Hz: 30 to 240 beats per minute
RESP_CUTOFF = 0.5              # Hz: up to 30 breaths per minute
N_PULSE = 251                  # taps of the band-pass filter (group delay 1 s)
N_RESP = 501                   # taps of the low-pass filter (group delay 2 s)
WINDOWS = ["rectangular", "hann", "hamming", "blackman"]
DEFAULT_WINDOW = "hamming"
KAISER_SPEC = {"atten_db": 60.0, "transition_hz": 0.3}   # Kaiser design example

# --------------------------------------------------------------------------
# Rate estimation
# --------------------------------------------------------------------------
HR_WINDOW_S = 16               # beat-to-beat intervals averaged over 16 s
RR_WINDOW_S = 60               # breaths counted over 60 s
MIN_BEAT_S = 0.33              # refractory distance between beats (max 180 bpm)
MIN_BREATH_S = 1.5             # refractory distance between breaths (max 40 per min)
