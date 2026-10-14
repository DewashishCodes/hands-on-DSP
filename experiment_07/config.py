"""
config.py
---------
Central configuration for Experiment 7 (Difference-equation based smoothing).

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
# Dataset: daily Indian rupee / US dollar exchange rate (FRED series DEXINUS)
# --------------------------------------------------------------------------
SERIES = "DEXINUS"
START, END = "2019-01-01", "2025-12-31"
DATA_URL = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={SERIES}&cosd={START}&coed={END}"
DATA_FILE = DATA_DIR / f"fred_{SERIES.lower()}_{START[:4]}_{END[:4]}.csv"

# --------------------------------------------------------------------------
# Smoothers (all expressed as difference equations)
# --------------------------------------------------------------------------
N_EMA = 20                        # "20-day" exponential moving average
ALPHA = 2.0 / (N_EMA + 1)         # the convention used in finance: alpha = 2 / (N + 1)
N_MA = 20                         # 20-day simple moving average
HOLT_ALPHA, HOLT_BETA = ALPHA, 0.05   # trend-corrected (second-order) smoother
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
TRADEOFF_LAGS = [1, 2, 3, 5, 7, 10, 15, 20]   # target lags (days) for the trade-off study
ZOOM = ("2022-01-01", "2022-12-31")            # year of the sharpest rupee fall
TRANSIENT_TOL = 0.01                           # 1 % initial-condition error
