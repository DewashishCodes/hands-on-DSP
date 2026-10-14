"""
config.py
---------
Central configuration for Experiment 9 (Linear convolution using the DFT).

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
# Dataset: dry speech, a measured room impulse response, and the published
# convolution of the two (University of Rostock acoustics exercises, CC0)
# --------------------------------------------------------------------------
BASE_URL = "https://raw.githubusercontent.com/spatialaudio/communication-acoustics-exercises/master/data/"
FILES = {
    "dry": "xmas.wav",            # "Wishing you a Merry Christmas..." recorded dry (Freesound 136777)
    "rir": "rir_clap.wav",        # seminar room, measured by clapping two wooden boards
    "wet": "xmas_clap.wav",       # published result: xmas.wav convolved with rir_clap.wav
}

# --------------------------------------------------------------------------
# Convolution study
# --------------------------------------------------------------------------
OLA_BLOCKS = [256, 1024, 4096, 16384]     # overlap-add block lengths (samples)
DEFAULT_BLOCK = 4096
DIRECT_TIMING_SECONDS = 0.25              # dry-signal length used to time direct convolution
DECONV_EPS = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]   # regularisation of the inverse filter (fraction of max |H|^2)
NOISE_SNR_DB = 40.0                       # recording noise added in the noisy deconvolution case
SEED = 9
