# Experiment 5: Autocorrelation

## What the experiment is about
The autocorrelation sequence and its properties (even symmetry, maximum at zero lag equal to the energy); three ways to compute it (definition, convolution with the folded signal, and Wiener-Khinchin via the power spectrum); biased versus unbiased estimators; and using autocorrelation to detect hidden periodicity and to test for white noise.

## Real-world context
Space-weather planning. Solar maximum brings more flares and geomagnetic storms, which affect satellite drag, GPS, HF radio and power grids, so operators need the period of the solar cycle.

## Data used
| Item | Detail |
|---|---|
| Source | SILSO World Data Center, Royal Observatory of Belgium |
| Series | International sunspot number v2.0, monthly mean total |
| Period | January 1749 to September 2026 (3,333 months) |
| Size / licence | 124 kB text file, cached in `data/`; CC BY-NC 4.0 |

## What was done
1. Implemented the three autocorrelation methods and verified the symmetry and maximum-at-zero properties, and that r[0] equals the energy.
2. Computed the normalised ACF up to 50 years of lag, and estimated the cycle period with parabolic peak refinement.
3. Tested the residual (after a 13-month mean) and the first difference for whiteness against the ±1.96/√N band.
4. Repeated the estimate on four 70-year epochs and with added white noise at 0, −10 and −16 dB SNR.
5. Verified the theory with 10 unit tests.

## Key results
* The solar-cycle period is 10.65 years (ρ = 0.579), with repeats at 21.8, 32.4 and 43.2 years. The shrinking peaks measure how irregular the cycle is.
* The series decorrelates quickly: ρ falls below 0.5 after 20 months and crosses zero at 36 months.
* The period has been stable at 10.7 to 11.0 years since 1818; 1749 to 1818 gives 9.3 years.
* The cycle is still found at −10 dB SNR but lost at −16 dB.
* The residual is not white: 21 of 36 lags lie outside the band.
* The three methods agree to 4e-16.

## Files and how to run
`python run_experiment.py`, `python gui.py` (periodicity detector with a noise slider), `python -m unittest test_autocorr` and `python build_report.py` (9 pages).
