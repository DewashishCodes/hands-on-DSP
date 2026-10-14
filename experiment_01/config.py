"""
config.py
---------
Central configuration for Experiment 1 (Basic Operations on Discrete-Time Signals).

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
# Dataset: hourly ERA5 reanalysis for Pune, served by the Open-Meteo archive API
# --------------------------------------------------------------------------
LOCATION_NAME = "Pune, Maharashtra, India"
LATITUDE = 18.5204
LONGITUDE = 73.8567
TIMEZONE = "Asia/Kolkata"
START_DATE = "2026-10-01"
END_DATE = "2026-10-07"

API_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARIABLES = [
    "temperature_2m",        # air temperature at 2 m          [deg C]
    "dew_point_2m",          # dew-point temperature at 2 m     [deg C]
    "relative_humidity_2m",  # relative humidity at 2 m         [%]
    "shortwave_radiation",   # global horizontal irradiance GHI [W/m^2]
    "direct_radiation",      # direct (beam) horizontal         [W/m^2]
    "diffuse_radiation",     # diffuse horizontal DHI           [W/m^2]
    "cloud_cover",           # total cloud cover                [%]
]
DATA_FILE = DATA_DIR / f"pune_hourly_{START_DATE}_{END_DATE}.csv"

# Short, readable column names used throughout the code
COLUMN_MAP = {
    "temperature_2m": "temp_c",
    "dew_point_2m": "dewpoint_c",
    "relative_humidity_2m": "rh_pct",
    "shortwave_radiation": "ghi_wm2",
    "direct_radiation": "direct_wm2",
    "diffuse_radiation": "diffuse_wm2",
    "cloud_cover": "cloud_pct",
}

# Human-readable labels/units for every signal (used by plots and the GUI)
SIGNAL_INFO = {
    "temp_c":      ("Air temperature", "°C"),
    "dewpoint_c":  ("Dew-point temperature", "°C"),
    "rh_pct":      ("Relative humidity", "%"),
    "ghi_wm2":     ("Global horizontal irradiance", "W/m²"),
    "direct_wm2":  ("Direct horizontal irradiance", "W/m²"),
    "diffuse_wm2": ("Diffuse horizontal irradiance", "W/m²"),
    "cloud_pct":   ("Cloud cover", "%"),
}

# --------------------------------------------------------------------------
# Signal-processing constants
# --------------------------------------------------------------------------
FS_PER_HOUR = 1.0          # sampling rate: 1 sample per hour (Ts = 1 h)
SAMPLES_PER_DAY = 24

# Experiment parameters (defaults; the GUI lets the user change all of them)
SHIFT_K = 24               # delay of one day -> persistence forecast
AMP_SCALE_A = 3.0          # 1 kWp -> 3 kWp rooftop array
DECIMATION_D = 3           # hourly -> 3-hourly synoptic reporting
INTERP_L = 3               # 3-hourly -> hourly reconstruction
MA_LENGTH_M = 24           # 24-hour moving average (daily-mean filter)
FOG_DPD_THRESHOLD = 2.0    # dew-point depression below which fog/mist is likely [deg C]

# Photovoltaic model (PVWatts-style, Dobos 2014)
PV_RATED_KW = 1.0          # rated DC power at STC
PV_GAMMA = -0.0037         # power temperature coefficient [1/deg C] (mono-Si)
PV_NOCT = 45.0             # nominal operating cell temperature [deg C]
STC_IRRADIANCE = 1000.0    # [W/m^2]
STC_TEMPERATURE = 25.0     # [deg C]
