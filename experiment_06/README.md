# Experiment 6: Cross-Correlation

## What the experiment is about
Cross-correlation, its properties (mirror symmetry when the signals are swapped, the Cauchy-Schwarz bound), the correlation coefficient, and time-delay estimation with sub-sample parabolic refinement, including the delay ambiguity of periodic signals.

## Real-world context
Timing the tide as it travels up an estuary, for port scheduling and storm-surge warnings. The same technique locates pipe leaks, ranges GPS signals and finds radar targets.

## Data used
| Item | Detail |
|---|---|
| Source | NOAA CO-OPS Tides and Currents data API (verified hourly heights, MSL datum) |
| Gauges | Sewells Point, Lewisetta, Solomons Island, Annapolis, Baltimore (Chesapeake Bay, 0 to 259 km) |
| Period | January 2024, hourly (744 samples per gauge) |
| Size | about 25 kB per gauge, merged into one cached CSV |

## What was done
1. Implemented cross-correlation three ways (definition, convolution with the folded reference, FFT) and verified its properties.
2. Separated each record into a tidal band and a subtidal (wind-driven) band with a 25-hour moving average.
3. Estimated the delay of each gauge relative to the bay mouth, and chained gauge-to-gauge delays to resolve the 12.42-hour ambiguity.
4. Fitted distance against delay to get the wave speed, and the mean depth from c = √(gh).
5. Verified the theory with 10 unit tests (delay in noise, fractional delay, periodic ambiguity).

## Key results
* Delays from the bay mouth: Lewisetta 4.36 h, Solomons 5.17 h, Annapolis 8.37 h, Baltimore 9.95 h.
* Wave speed is 7.49 m/s (R² = 0.979), giving a mean depth of 5.7 m against the published 6.4 m.
* Using the tidal band raises Baltimore's correlation from 0.58 to 0.79.
* The chained short delays sum to 9.58 h, which confirms 9.95 h and rules out −2.47 h.
* A storm on 9 to 10 January raised Baltimore about 1 m above normal.

## Files and how to run
`python run_experiment.py`, `python gui.py` (time-delay estimator with a manual-shift slider), `python -m unittest test_xcorr` and `python build_report.py` (10 pages).
