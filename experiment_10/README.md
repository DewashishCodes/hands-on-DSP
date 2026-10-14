# Experiment 10: Circular Convolution using the DFT

## What the experiment is about
N-point circular convolution computed as IDFT{X·H} without zero padding, the circular convolution theorem (the DFT diagonalises the circulant matrix), the cyclic prefix that turns linear convolution into circular convolution (the basis of OFDM), and reading smoothing kernels through their DFT gains.

## Real-world context
Daily climate normals: the expected temperature for every day of the year, used for heat-wave alerts (IMD), agriculture and cooling design. The calendar wraps (31 December is followed by 1 January), so smoothing it must be circular.

## Data used
| Item | Detail |
|---|---|
| Source | NOAA GHCN-Daily via the NCEI data service |
| Station | IN022021900, New Delhi (Safdarjung) |
| Period | 1991 to 2020 (the WMO normal period); 29 February removed |
| Signals | daily maximum and minimum temperature (9,668 / 9,244 observations); 365-day climatology built from them |
| Size | 417 kB CSV, cached in `data/` |

## What was done
1. Built the 365-day climatology, with circular interpolation of any missing days.
2. Smoothed it by circular convolution via the DFT, and checked the result against the direct sum, the DFT matrix and the cyclic-prefix construction.
3. Compared it with zero-padded linear smoothing for kernels of 7 to 61 days.
4. Compared a 31-day boxcar, a Gaussian and a 3-harmonic kernel by their DFT gains, and derived Delhi's daily normals and key dates.
5. Timed direct against FFT circular convolution for N = 64 to 4,096.
6. Verified the theory with 9 unit tests.

## Key results
* Circular convolution via the DFT equals the direct definition to 3e-14 °C. The cyclic-prefix construction matches to 3e-14 °C.
* Zero-padded linear smoothing is wrong by up to 10.3 °C on the 30 days nearest New Year.
* Delhi's normals: the hottest day is 31 May (40.3 °C), the coldest night is 8 January (6.9 °C), and the steepest monsoon cooling falls around 30 June.
* What the smoothing removed (0.44 °C) matches the sampling noise of a 30-year mean (0.53 °C).
* For N = 365 the FFT needs 14 times fewer multiplications, yet the direct matrix product is still faster in practice. The FFT wins from about N = 1,000 and is 46 times faster at N = 4,096.

## Files and how to run
`python run_experiment.py` (also writes `outputs/normals.csv`), `python gui.py` (climate-normals smoother and N-point calculator), `python -m unittest test_circdft` and `python build_report.py` (10 pages).
