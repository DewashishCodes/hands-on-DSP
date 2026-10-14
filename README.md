# DSP Laboratory: Twelve Experiments in Python

Digital Signal Processing lab practicals, Symbiosis Institute of Technology, Pune. Every experiment uses a real, freely available dataset, has modular Python code with an interactive Tkinter GUI and unit tests, and comes with an 8 to 12 page report (`DSP_Experiment_NN.docx` and `.pdf`) with the Symbiosis header on every page.

| No. | Experiment | Real-world application | Dataset | Pages |
|---|---|---|---|---|
| [1](experiment_01/README.md) | Basic operations on DT signals | Rooftop-solar and weather monitoring | ERA5 hourly weather and irradiance, Pune (Open-Meteo) | 10 |
| [2](experiment_02/README.md) | Linearity and time invariance | Air-quality averages, AQI, sensor drift | CAMS hourly PM2.5 / NO₂, Pune, Diwali 2025 (Open-Meteo) | 10 |
| [3](experiment_03/README.md) | Moving average, sampling and reconstruction | Bird-survey recorder sampling rate | ESC-50 bird-song clip (GitHub) | 10 |
| [4](experiment_04/README.md) | Circular convolution | Wind roses and runway orientation | METAR winds, Mumbai airport 2024 (Iowa Environmental Mesonet) | 10 |
| [5](experiment_05/README.md) | Autocorrelation | Solar-cycle period for space weather | Monthly sunspot number 1749 to 2026 (SILSO) | 9 |
| [6](experiment_06/README.md) | Cross-correlation | Tide travel time and wave speed | Five tide gauges, Chesapeake Bay, Jan 2024 (NOAA CO-OPS) | 10 |
| [7](experiment_07/README.md) | Difference-equation smoothing | EMA, Holt and MACD on the rupee | USD/INR daily rate 2019 to 2025 (FRED DEXINUS) | 10 |
| [8](experiment_08/README.md) | DFT and IDFT | Seasonal cycle of atmospheric CO₂ | Mauna Loa monthly CO₂ (NOAA GML) | 10 |
| [9](experiment_09/README.md) | Linear convolution using the DFT | Convolution reverb | Dry speech and seminar-room impulse response (Univ. of Rostock, CC0) | 11 |
| [10](experiment_10/README.md) | Circular convolution using the DFT | Daily climate normals, OFDM | Daily Tmax / Tmin, New Delhi 1991 to 2020 (NOAA GHCN-Daily) | 10 |
| [11](experiment_11/README.md) | FIR filter design by windowing | Heart and breathing rate from a PPG | ICU photoplethysmogram, record bidmc01 (PhysioNet BIDMC) | 10 |
| [12](experiment_12/README.md) | IIR filter design | EEG mains notch and alpha rhythm | EEG eyes open / closed, S001 (PhysioNet EEGMMIDB) | 11 |

## Folder layout (same in every experiment)
| File | Purpose |
|---|---|
| `config.py` | every constant, path and parameter |
| `data_loader.py` | downloads the dataset once and caches it in `data/` |
| core module (`signal_ops.py`, `fir.py`, `iir.py`, ...) | pure DSP functions, no I/O |
| `analysis.py` | applies the DSP to the data and computes every reported number |
| `plots.py` | report figures |
| `gui.py` | interactive Tkinter application |
| `run_experiment.py` | batch run: figures, `results.json` and any export files |
| `test_*.py` | unit tests that verify the theory |
| `build_report.py` | builds the Word and PDF report from the same results |

The shared report formatter is `tools/report_kit.py`.

## Running
Requirements: Python 3.10 with NumPy, SciPy, pandas and Matplotlib, plus `python-docx`, `latex2mathml` and Pygments for the reports. Microsoft Word is used for the PDF export.

```
cd experiment_NN
python run_experiment.py        # data, figures, results
python gui.py                   # interactive app
python -m unittest discover     # tests
python build_report.py          # .docx and .pdf
```

## Submission copies
`submission/experiment_NN/` holds only the final `DSP_Experiment_NN.docx` and `.pdf` for each experiment, ready to hand in.

## Datasets
Datasets are not stored in this repository. Each `data_loader.py` downloads its dataset from the original public source on the first run and caches it in `data/`. Credit and licence for every dataset are given in that experiment's README and report.

## Still to fill in
PRN and date are placeholders in the `STUDENT` dictionary at the top of each `build_report.py`.

## Licence
Code and reports are released under the MIT Licence (see `LICENSE`). The datasets remain under their own licences.
