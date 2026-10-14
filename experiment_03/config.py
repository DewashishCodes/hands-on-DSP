"""
config.py
---------
Central configuration for Experiment 3 (Moving-average filter; sampling and
reconstruction of a signal).

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
# Dataset: one 5-second field recording from the ESC-50 environmental-sound dataset
# --------------------------------------------------------------------------
CLIP_ID = "1-34495-A-14"                     # category 14: chirping birds
CLIP_URL = f"https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/{CLIP_ID}.wav"
CLIP_FILE = DATA_DIR / f"{CLIP_ID}.wav"

FS_ORIGINAL = 44_100                         # Hz, rate of the published clip
FS_REF = 48_000                              # Hz, "analog" reference grid used in the experiment
RESAMPLE_UP, RESAMPLE_DOWN = 160, 147        # 44.1 kHz * 160/147 = 48 kHz

# --------------------------------------------------------------------------
# Sampling experiment
# --------------------------------------------------------------------------
RATES = [16_000, 8_000, 4_000]               # Hz: wideband speech, telephone, very low-power
PREFILTERS = ["None", "Moving average", "FIR (ideal-like)"]
FIR_TAPS = 255                               # length of the high-quality anti-alias filter
FIR_EDGE = 0.90                              # passband edge as a fraction of the new Nyquist

# --------------------------------------------------------------------------
# Reconstruction
# --------------------------------------------------------------------------
RECON_METHODS = ["Zero-order hold", "Linear", "Sinc (band-limited)"]
SINC_HALF_WIDTH = 32                         # sinc kernel spans ±32 input samples
SINC_KAISER_BETA = 8.0

# --------------------------------------------------------------------------
# Moving-average filter study
# --------------------------------------------------------------------------
MA_LENGTHS = [3, 6, 12]                      # samples at 48 kHz (= D for 16, 8, 4 kHz)
BANDWIDTH_FRACTION = 0.99                    # energy fraction that defines the signal bandwidth
BITS_PER_SAMPLE = 16
