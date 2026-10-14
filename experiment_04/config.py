"""
config.py
---------
Central configuration for Experiment 4 (Circular convolution).

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
# Dataset: hourly METAR reports, Mumbai airport, from the Iowa Environmental Mesonet
# --------------------------------------------------------------------------
STATION = "VABB"
STATION_NAME = "Chhatrapati Shivaji Maharaj International Airport, Mumbai"
YEAR = 2024
API_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
DATA_FILE = DATA_DIR / f"{STATION.lower()}_metar_wind_{YEAR}.csv"

RUNWAYS = {"09/27": 90.0, "14/32": 140.0}      # nominal headings of the two runways (degrees)

# --------------------------------------------------------------------------
# Circular signal: wind-direction histogram
# --------------------------------------------------------------------------
N_BINS = 36                  # METAR directions are reported to the nearest 10 degrees
BIN_DEG = 360 // N_BINS
CALM_KT = 1.0                # speeds below this are "calm" (no direction)
SEASONS = {
    "Annual": list(range(1, 13)),
    "Winter (Dec to Feb)": [12, 1, 2],
    "Monsoon (Jun to Sep)": [6, 7, 8, 9],
}

# Smoothing kernel (circular, centred): h[0] = 1/2, h[±1] = 1/4
SMOOTH_KERNEL = [0.25, 0.5, 0.25]

# --------------------------------------------------------------------------
# Runway wind-coverage analysis (ICAO Annex 14, 3.1.3)
# --------------------------------------------------------------------------
CROSSWIND_LIMITS_KT = [10.0, 13.0, 20.0]       # by runway reference field length
DEFAULT_LIMIT_KT = 10.0
ICAO_USABILITY = 0.95                          # runway must be usable >= 95 % of the time
MAX_SPEED_KT = 40
