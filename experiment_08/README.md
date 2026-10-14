# Experiment 8: DFT and IDFT

## What the experiment is about
The discrete Fourier transform and its inverse implemented from the definition, as a matrix product and with the FFT; reading frequency, amplitude and phase from bins; Parseval's theorem and conjugate symmetry; spectral leakage, windowing and zero padding; and frequency-domain filtering by editing DFT coefficients.

## Real-world context
Separating the seasonal "breathing" of the planet's vegetation from the long-term rise in atmospheric CO₂. The same transforms run inside OFDM (4G, 5G, Wi-Fi), MRI and audio compression.

## Data used
| Item | Detail |
|---|---|
| Source | NOAA Global Monitoring Laboratory (with Scripps), Mauna Loa Observatory, Hawaii |
| Series | Monthly mean CO₂ (the Keeling Curve), plus NOAA's deseasonalized column for checking |
| Period | March 1958 to September 2026 (823 months); the DFT uses January 1959 to December 2025, N = 804 = 67 whole years |
| Size | 38 kB CSV, cached in `data/` |

## What was done
1. Detrended the record with a cubic, then computed the DFT by definition, by matrix and by FFT, and verified the inverse, Parseval's theorem and symmetry.
2. Read the seasonal lines from the amplitude spectrum.
3. Kept only the 1 and 2 cycle-per-year bins and used the IDFT to synthesise and remove the seasonal cycle, then compared the result with NOAA's deseasonalized series.
4. Showed leakage by cutting 5 months from the record, and corrected it with a Hann window and zero padding.
5. Tracked the seasonal swing by decade, and timed the matrix DFT against the FFT.
6. Verified the theory with 10 unit tests.

## Key results
* The three methods agree to 1e-13, and Parseval's theorem holds to 1e-16.
* The spectrum has two clean lines: 2.84 ppm at 1 cycle per year and 0.81 ppm at 2, against a noise floor of about 0.016 ppm.
* Four of the 804 coefficients explain 88 % of the detrended variance. CO₂ peaks in May and bottoms out in October (a 6.2 ppm swing).
* The deseasonalized series matches NOAA's own to 0.165 ppm.
* Leakage: the annual line drops from 2.84 to 2.20 ppm when 5 months are cut. A Hann window recovers 2.63 ppm, and Hann plus zero padding gives 2.88 ppm.
* The seasonal swing grew by 14 % since 1959.
* The FFT is about 1,160 times faster than the matrix DFT at N = 804.

## Files and how to run
`python run_experiment.py`, `python gui.py` (spectrum analyser and DFT calculator), `python -m unittest test_dft` and `python build_report.py` (10 pages).
