# Experiment 2: Testing Systems for Linearity and Time Invariance

## What the experiment is about
How to decide whether a discrete-time system is linear and time invariant (LTI) using black-box superposition and shift tests, and how to obtain and interpret the time response (impulse and step response) of LTI systems.

## Real-world context
Every published air-quality number passes through a chain of systems. Whether each one is LTI decides whether it can be analysed by its impulse response, and whether values may be averaged or added.

## Data used
| Item | Detail |
|---|---|
| Source | CAMS global atmospheric composition (Copernicus / ECMWF) via the Open-Meteo air-quality API |
| Location | Pune, Maharashtra |
| Period | 14 to 27 October 2025, hourly (N = 336), bracketing Diwali (20 October) |
| Signals | PM2.5 (x₁), NO₂ (x₂), plus PM10 and CO for context |
| Size | about 15 kB CSV, cached in `data/` |

## What was done
1. Implemented four real systems:
   * **S1:** the 24-hour moving average used by the Indian NAAQS.
   * **S2:** the exponential smoother used in low-cost sensor displays.
   * **S3:** the CPCB AQI sub-index.
   * **S4:** a sensor whose gain drifts with time.
2. Wrote black-box linearity (superposition) and time-invariance (shift) tests that work for any callable system, including any typed difference equation.
3. Computed impulse and step responses, checked y = x * h for the LTI systems, and applied all systems to the Diwali pollution episode.
4. Verified the theory with 13 unit tests, including the effect of a non-zero initial condition.

## Key results
* S1 and S2 are LTI (test errors about 1e-16).
* The AQI is nonlinear (superposition error 0.218). By hand, I(45) + I(30) = 125 but I(75) = 150, so AQI values must never be averaged.
* The drifting sensor is linear but time-varying (shift error 0.0198).
* PM2.5 peaked at 78.5 µg/m³ on 19 October. The 24-hour average exceeded the 60 µg/m³ standard for 15 hours and lagged the hourly peak by 3 h.
* The smoother's time constant matches theory (4.48 samples).

## Files and how to run
`python run_experiment.py`, `python gui.py` (the system tester, including custom difference equations), `python -m unittest test_systems` and `python build_report.py` (10 pages).
