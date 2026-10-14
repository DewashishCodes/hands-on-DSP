# Experiment 11: FIR Filter Design by Windowing

## What the experiment is about
Designing linear-phase FIR filters by the window method (ideal sinc × rectangular, Hann, Hamming, Blackman or Kaiser window), Kaiser's design formulas, implementation as a non-recursive difference equation, frequency response, constant group delay and delay compensation.

## Real-world context
Pulse oximeters and smartwatches extract heart rate and breathing rate from one optical fingertip sensor (PPG). FIR filters are chosen because linear phase keeps the pulse shape and beat timing intact.

## Data used
| Item | Detail |
|---|---|
| Source | PhysioNet BIDMC PPG and Respiration Dataset (Pimentel et al.), ODC Attribution licence |
| Record | `bidmc01`: an 88-year-old male in the medical ICU, 8 minutes at 125 Hz |
| Signal | PLETH (PPG), read directly from the WFDB format-16 binary |
| References | bedside monitor PULSE / RESP (1 per second) and manually annotated breaths |
| Size | about 600 kB in total, cached in `data/` |

## What was done
1. Implemented the windows, the ideal low-pass and band-pass responses, windowed designs, Kaiser's formulas and the FIR difference equation. They match SciPy's `firwin` exactly.
2. Compared the four classical windows on the same specification (ripple, attenuation, transition width) against the textbook table.
3. Designed a 251-tap 0.5 to 4 Hz band-pass for the pulse and a 501-tap 0.5 Hz low-pass for breathing, both Hamming, and compensated their delays.
4. Detected beats and breaths, estimated the rates and validated them against the monitor and the annotations. Then studied the effect of filter length (251 to 1,001 taps).
5. Verified the theory with 9 unit tests.

## Key results
* Measured stopband attenuation matches the textbook (theory in brackets):

| Window | Measured | Textbook |
|---|---|---|
| Rectangular | −21 dB | −21 dB |
| Hann | −44 dB | −44 dB |
| Hamming | −52 dB | −53 dB |
| Blackman | −75 dB | −74 dB |

* The Kaiser formulas sized a 60 dB design at 1,511 taps, which achieved 59.4 dB.
* Group delay is flat at 1 s and 2 s (linear phase), and is removed exactly.
* Heart rate is 92.2 bpm against the monitor's 91.0 (MAE 1.49 bpm).
* Breathing rate is 19.8 against 21.1 breaths/min from the annotations (MAE 1.33); 87 % of annotated breaths were detected.
* Longer filters improve breathing rejection (−12 → −58 dB) and cut the heart-rate MAE to 1.22 bpm. Motion-artefact spikes remain, because in-band noise cannot be removed by any linear filter.

## Files and how to run
`python run_experiment.py` (also writes `outputs/pulse_filter.csv`), `python gui.py` (FIR filter designer with live rate error), `python -m unittest test_fir` and `python build_report.py` (10 pages).
