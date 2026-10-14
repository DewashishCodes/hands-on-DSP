# Experiment 7: Difference-Equation Based Smoothing

## What the experiment is about
Smoothing filters written and solved as linear constant-coefficient difference equations: recursive (IIR) versus non-recursive (FIR), poles and stability, initial conditions (zero-input and zero-state response), noise gain, lag and computational cost.

## Real-world context
Forex desks, importers and the RBI track the rupee with moving averages. The exponential moving average (EMA) and MACD indicators are themselves difference equations. The report studies signal-processing properties only, not trading strategies.

## Data used
| Item | Detail |
|---|---|
| Source | FRED (Federal Reserve Bank of St. Louis), series DEXINUS (Federal Reserve H.10) |
| Signal | Indian rupees per US dollar, daily noon buying rate |
| Period | 2 January 2019 to 31 December 2025 (1,748 trading days; 78 holidays dropped) |
| Size / licence | 33 kB CSV, cached in `data/`; public domain |

## What was done
1. Wrote one iterative solver for the general difference equation, with initial conditions, and checked it against SciPy's `lfilter`.
2. Implemented five smoothers:
   * 20-day moving average, non-recursive and recursive
   * 20-day EMA
   * double EMA
   * Holt's trend-corrected second-order smoother
3. Analysed each one's impulse response, poles and zeros, stability, noise gain, lag and multiplications per sample, and measured lag on the 2022 fall of the rupee.
4. Studied the lag versus noise trade-off, initial-condition transients and the MACD indicator.
5. Verified the theory with 11 unit tests.

## Key results
* With α = 2/(N + 1), the EMA equals the moving average in lag (9.5 days) and noise gain (0.050), using 2 multiplications instead of 20.
* Every smoother of this kind follows noise gain = 1/(2L + 1).
* Holt's filter tracks the 2022 trend with 0.2 days of lag, against 7.6 days for the EMA.
* Starting the EMA at zero leaves a transient for 47 trading days, as the pole at z = 0.905 predicts.
* The recursive moving average has a pole on the unit circle, so it is only marginally stable.
* MACD is a band-pass filter peaking at about a 55-day period.

## Files and how to run
`python run_experiment.py`, `python gui.py` (difference-equation smoother designer, including custom b and a coefficients), `python -m unittest test_smoothing` and `python build_report.py` (10 pages).
