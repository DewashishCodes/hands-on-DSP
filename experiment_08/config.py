"""
config.py
---------
Central configuration for Experiment 8 (DFT and IDFT).

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
# Dataset: monthly mean CO2 at Mauna Loa Observatory (NOAA GML, the Keeling Curve)
# --------------------------------------------------------------------------
DATA_URL = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv"
DATA_FILE = DATA_DIR / "co2_mm_mlo.csv"

# --------------------------------------------------------------------------
# DFT study
# --------------------------------------------------------------------------
SAMPLES_PER_YEAR = 12          # fs = 12 samples per year (monthly)
TREND_ORDER = 3                # polynomial removed before the DFT
HARMONICS = [1, 2]             # seasonal harmonics (cycles per year) kept for the IDFT model
LEAK_CUT = 5                   # months removed to show leakage (record no longer whole years)
ZERO_PAD = 8                   # zero-padding factor for the interpolated spectrum
DECADE = 120                   # months per window in the amplitude-change study
TIMING_N = [64, 128, 256, 512, 804, 1024, 2048]
