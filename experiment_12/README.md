# Experiment 12: IIR Filter Design and Implementation

## What the experiment is about
IIR filters designed by direct pole-zero placement (a 60 Hz notch and a first-order high-pass) and by the bilinear transform with pre-warping (Butterworth, Chebyshev I and elliptic band-passes); implementation as recursive difference equations and as cascaded second-order sections; stability, numerical precision, group delay and zero-phase (forward-backward) filtering.

## Real-world context
Every EEG amplifier uses IIR filters: a drift high-pass, a mains notch (50 Hz in India, 60 Hz in the USA) and rhythm band-passes. The alpha rhythm (8 to 13 Hz) grows when the eyes close (the Berger effect), which is used for drowsiness detection and brain-computer interfaces.

## Data used
| Item | Detail |
|---|---|
| Source | PhysioNet EEG Motor Movement/Imagery Database (Schalk et al., BCI2000), ODC Attribution licence |
| Recordings | subject S001, run R01 (1 min, eyes open) and run R02 (1 min, eyes closed) |
| Channel | O1 (left occipital, over the visual cortex), 160 Hz, µV |
| Format / size | EDF, parsed directly; 2 × 1.25 MB, downloaded from PhysioNet's AWS open-data mirror and cached in `data/` |

## What was done
1. Implemented the notch, the high-pass, pre-warping, the bilinear zpk mapping, the recursive difference equation, the SOS cascade and zero-phase filtering. The bilinear designs match SciPy to 1e-14.
2. Compared the three order-8 alpha band-passes by attenuation, ripple, group delay and pole radius, and found the equivalent FIR length.
3. Showed what goes wrong without pre-warping, and how an order-16 design becomes unstable in direct form (but not as biquads).
4. Notched the 60 Hz interference, compared causal and zero-phase output, and measured alpha power in 2-second epochs, eyes open against eyes closed.
5. Verified the theory with 8 unit tests.

## Key results
* The notch is −290 dB deep at 60 Hz and 2.0 Hz wide. It pushes the mains line (0.617 → 0.0015 µV²/Hz) below the EEG background and leaves the alpha band unchanged.
* Each band-pass uses 17 coefficients, against 81 for an FIR filter of equal selectivity at 5 Hz.
* The elliptic design's in-band group delay varies from 0.14 to 0.87 s, so the causal output lags the zero-phase output by 194 ms.
* Without pre-warping, a 50 to 70 Hz band lands at 39.5 to 48.0 Hz.
* An order-16 direct-form filter gets a pole at radius 1.023 from coefficient rounding and blows up to about 10³⁷; the biquad cascade stays stable.
* Alpha power is 15.1 times higher with eyes closed (Welch t = 11.0), and 97 % of eyes-closed epochs exceed every eyes-open epoch.

## Files and how to run
`python run_experiment.py` (also writes `outputs/alpha_sos.csv`), `python gui.py` (IIR filter designer), `python -m unittest test_iir` and `python build_report.py` (11 pages).
