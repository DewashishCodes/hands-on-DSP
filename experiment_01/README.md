# Experiment 1: Basic Operations on Discrete-Time Signals

## What the experiment is about
The nine elementary operations on discrete-time signals: addition, subtraction, multiplication, amplitude scaling, time shifting, time reversal, down-sampling, up-sampling and linear convolution. Each signal is stored with its own index vector `n`, so shifting and folding move the support correctly instead of silently dropping samples.

## Real-world context
The first stage of a rooftop-solar and weather monitoring pipeline: combining irradiance components, warning of condensation, estimating PV power, sizing an array, forecasting, compressing data and extracting trends.

## Data used
| Item | Detail |
|---|---|
| Source | ERA5 / ERA5-Land reanalysis (ECMWF, Copernicus) via the Open-Meteo archive API |
| Location | Pune, Maharashtra (18.52° N, 73.86° E) |
| Period | 1 to 7 October 2026, hourly (N = 168) |
| Signals | air temperature, dew point, relative humidity, global / direct / diffuse irradiance, cloud cover |
| Size / licence | 8.7 kB CSV, cached in `data/`; CC BY 4.0 (Open-Meteo) |

## What was done
1. Built a `DTSignal(n, x)` container and implemented all nine operations in `signal_ops.py`.
2. Gave every operation a physical job:
   * **Addition:** direct + diffuse irradiance = global irradiance.
   * **Subtraction:** temperature minus dew point = dew-point depression (fog indicator).
   * **Multiplication:** irradiance × thermal derating = PV power.
   * **Amplitude scaling:** a 1 kWp array scaled to a 3 kWp system.
   * **Time shift:** a 24-hour delay used as a persistence forecast.
   * **Time reversal:** folding and even/odd decomposition.
   * **Time scaling:** hourly data compressed to 3-hourly synoptic data and reconstructed.
   * **Convolution:** 24-hour moving average.
3. Verified every identity from the theory with 20 unit tests, and worked the operations by hand on four real samples.

## Key results
* Addition reconstructs the reported global irradiance with a 0.0 W/m² residual.
* 11 fog-risk hours were found, all between 02:00 and 07:00.
* PV yield is 5.29 kWh/kWp/day, with 9.9 % lost to heat.
* Scaling energy by A² holds exactly (ratio 9.00 for A = 3).
* The persistence forecast has an RMSE of 0.91 °C (74 % skill over climatology).
* The 24-hour moving average cuts the standard deviation from 3.60 to 0.30 °C; the output length is 191 = 168 + 24 − 1.

## Files and how to run
`python run_experiment.py` regenerates the figures and `results.json`, `python gui.py` opens the interactive signal-operations lab, `python -m unittest test_signal_ops` runs the tests, and `python build_report.py` rebuilds `DSP_Experiment_01.docx` / `.pdf` (10 pages).
