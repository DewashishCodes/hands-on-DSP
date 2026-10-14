"""
config.py
---------
Central configuration for Experiment 2 (Testing systems for linearity and time
invariance; time response of LTI systems).

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
# Dataset: hourly CAMS air-quality reanalysis for Pune (Open-Meteo API)
# --------------------------------------------------------------------------
LOCATION_NAME = "Pune, Maharashtra, India"
LATITUDE = 18.5204
LONGITUDE = 73.8567
TIMEZONE = "Asia/Kolkata"
START_DATE = "2025-10-14"          # two weeks around Diwali 2025 (20 to 21 October)
END_DATE = "2025-10-27"
FESTIVAL_DATE = "2025-10-20"

API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
HOURLY_VARIABLES = ["pm2_5", "pm10", "nitrogen_dioxide", "carbon_monoxide"]
DATA_FILE = DATA_DIR / f"pune_air_quality_{START_DATE}_{END_DATE}.csv"

COLUMN_MAP = {
    "pm2_5": "pm25",
    "pm10": "pm10",
    "nitrogen_dioxide": "no2",
    "carbon_monoxide": "co",
}
SIGNAL_INFO = {
    "pm25": ("PM2.5 concentration", "µg/m³"),
    "pm10": ("PM10 concentration", "µg/m³"),
    "no2":  ("Nitrogen dioxide", "µg/m³"),
    "co":   ("Carbon monoxide", "µg/m³"),
}

# --------------------------------------------------------------------------
# Systems under test (defaults; the GUI lets the user change all of them)
# --------------------------------------------------------------------------
MA_LENGTH_M = 24           # 24-hour average used by the Indian NAAQS for PM2.5
EWMA_ALPHA = 0.2           # smoothing factor of a low-cost sensor's display filter
DRIFT_BETA = 0.05 / 24     # sensor gain drift: +5 % per day

# Indian National Air Quality Index (CPCB), PM2.5 breakpoints: (C_lo, C_hi, I_lo, I_hi)
AQI_PM25_BREAKPOINTS = [
    (0, 30, 0, 50),
    (30, 60, 50, 100),
    (60, 90, 100, 200),
    (90, 120, 200, 300),
    (120, 250, 300, 400),
    (250, 500, 400, 500),
]
AQI_CATEGORIES = [(50, "Good"), (100, "Satisfactory"), (200, "Moderate"),
                  (300, "Poor"), (400, "Very poor"), (500, "Severe")]
NAAQS_PM25_24H = 60.0      # µg/m³, 24-hour standard

# --------------------------------------------------------------------------
# Test parameters
# --------------------------------------------------------------------------
TEST_A = 1.5               # superposition weights  a·x1 + b·x2
TEST_B = 0.5
TEST_SHIFT_K = 12          # samples (hours) used in the time-invariance test
RESPONSE_LENGTH = 48       # samples shown for impulse / step responses
TOLERANCE = 1e-9           # relative error below which a property "holds"
