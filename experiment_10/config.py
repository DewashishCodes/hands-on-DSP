"""
config.py
---------
Central configuration for Experiment 10 (Circular convolution using the DFT).

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
# Dataset: NOAA GHCN-Daily, New Delhi (Safdarjung), daily max / min temperature
# --------------------------------------------------------------------------
STATION = "IN022021900"
STATION_NAME = "New Delhi (Safdarjung)"
START, END = "1991-01-01", "2020-12-31"           # the WMO 1991 to 2020 climate-normal period
API_URL = ("https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries"
           f"&stations={STATION}&dataTypes=TMAX,TMIN&startDate={START}&endDate={END}"
           "&format=csv&units=metric")
DATA_FILE = DATA_DIR / f"ghcn_{STATION}_{START[:4]}_{END[:4]}.csv"

# --------------------------------------------------------------------------
# Circular smoothing of the 365-day climatology
# --------------------------------------------------------------------------
N_DAYS = 365                     # 29 February is dropped so every year has the same 365 calendar days
MA_DAYS = 31                     # moving-average (boxcar) kernel length
GAUSS_SIGMA = 7.0                # Gaussian kernel standard deviation (days)
HARMONICS = 3                    # number of annual harmonics kept by NOAA-style harmonic fitting
KERNEL_LENGTHS = [7, 15, 31, 61]
TIMING_SIGNALS = 1000            # number of 365-day signals smoothed in the speed test
