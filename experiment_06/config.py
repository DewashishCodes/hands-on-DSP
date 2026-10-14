"""
config.py
---------
Central configuration for Experiment 6 (Cross-correlation).

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
# Dataset: hourly water level at five NOAA tide gauges along Chesapeake Bay
# --------------------------------------------------------------------------
API_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
BEGIN, END = "20240101", "20240131"
# id: (name, latitude, longitude) from the NOAA CO-OPS metadata API, ordered from the bay mouth
STATIONS = {
    "8638610": ("Sewells Point", 36.94278, -76.32861),
    "8635750": ("Lewisetta", 37.99610, -76.46440),
    "8577330": ("Solomons Island", 38.31670, -76.45170),
    "8575512": ("Annapolis", 38.98388, -76.48003),
    "8574680": ("Baltimore", 39.26669, -76.57831),
}
REFERENCE = "8638610"                 # gauge nearest the ocean entrance
DATA_FILE = DATA_DIR / f"chesapeake_hourly_{BEGIN}_{END}.csv"

TS_HOURS = 1.0                        # sampling interval
M2_PERIOD_H = 12.4206                 # principal lunar semidiurnal tide
G = 9.81                              # m/s^2
PUBLISHED_MEAN_DEPTH_M = 6.4          # Chesapeake Bay mean depth (Chesapeake Bay Program)

# --------------------------------------------------------------------------
# Cross-correlation study
# --------------------------------------------------------------------------
MAX_LAG_H = 36                        # lags shown in the figures
SEARCH_LAG_H = (-3, 30)               # window in which the delay peak is searched
SUBTIDAL_WINDOW_H = 25                # moving average that removes the tides (exp. 3 filter)
