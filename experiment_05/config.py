"""
config.py
---------
Central configuration for Experiment 5 (Autocorrelation).

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
# Dataset: monthly mean total sunspot number, SILSO (Royal Observatory of Belgium)
# --------------------------------------------------------------------------
DATA_URL = "https://www.sidc.be/SILSO/INFO/snmtotcsv.php"
DATA_FILE = DATA_DIR / "silso_monthly_sunspot_number.csv"
COLUMNS = ["year", "month", "decimal_year", "sn", "sn_std", "n_obs", "definitive"]
FIRST_YEAR = 1749

# --------------------------------------------------------------------------
# Autocorrelation study
# --------------------------------------------------------------------------
MAX_LAG = 600                 # months (50 years) shown in the main ACF figure
SMOOTH_MONTHS = 13            # the standard 13-month running mean used by solar physicists
CONF_Z = 1.96                 # 95 % white-noise bounds  ±1.96 / sqrt(N)
PEAK_SEARCH = (60, 200)       # months: window in which the first cycle peak is searched
WINDOWS = [(1749, 1818), (1818, 1887), (1887, 1956), (1956, 2026)]   # ~70-year epochs
SEED = 5                      # random seed for the noise-robustness study
NOISE_LEVELS = [0.0, 1.0, 3.0, 6.0]   # noise std as a multiple of the signal std
