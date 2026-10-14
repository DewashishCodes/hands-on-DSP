# Experiment 4: Circular Convolution

## What the experiment is about
N-point circular convolution computed three independent ways (direct modulo-N sum, circulant matrix, and linear convolution wrapped modulo N); its relation to linear convolution through time aliasing; and zero padding to N ≥ L + M − 1.

## Real-world context
Wind direction is circular: 350° is next to 0°. Smoothing a wind rose and checking runway orientation against the ICAO 95 % usability rule are circular operations, and ordinary convolution is wrong at the north seam.

## Data used
| Item | Detail |
|---|---|
| Source | Iowa Environmental Mesonet (Iowa State University) METAR archive |
| Station | VABB, Chhatrapati Shivaji Maharaj International Airport, Mumbai |
| Period | all of 2024, hourly reports (8,725 reports; 811 calm, 16 variable) |
| Signal | 36-bin wind-direction histogram (METAR directions are reported to 10°) |
| Size | 288 kB CSV, cached in `data/` |

## What was done
1. Implemented the three methods of circular convolution and checked that they agree.
2. Smoothed the annual and seasonal wind roses circularly and with ordinary (zero-padded) linear convolution.
3. Demonstrated time aliasing and its removal by zero padding to N = 38.
4. Expressed runway usability for every heading as a sum of circular convolutions of speed-binned histograms with crosswind acceptance kernels, and checked it against a brute-force count over all reports.
5. Verified the theory with 13 unit tests (DFT agreement, rotation, mass preservation).

## Key results
* The three methods agree exactly. Wrapping the linear result reproduces the circular one, and zero padding makes them equal.
* Linear smoothing loses 66 of 7,898 reports and errs by 45.5 at the 0°/350° seam. Circular smoothing keeps every report.
* Winter winds come from 40° (north-east) and monsoon winds from 240° (south-west).
* Runway usability at the 10 kt crosswind limit:
  * The best heading is 070° (99.53 %).
  * The real runway 09/27 gets 99.29 %.
  * The cross runway 14/32 gets 95.04 %, right at the ICAO minimum.
  * The convolution result matches the brute-force count exactly.

## Files and how to run
`python run_experiment.py`, `python gui.py` (wind-rose lab, runway planner and N-point calculator), `python -m unittest test_circular` and `python build_report.py` (10 pages).
