# Experiment 3: Moving-Average Filter, Sampling and Reconstruction

## What the experiment is about
The sampling theorem and aliasing; the moving-average filter and its frequency response, including its use as a cheap anti-aliasing filter; and reconstruction of a signal from its samples by zero-order hold, linear and sinc (band-limited) interpolation.

## Real-world context
Choosing the sampling rate of a battery-powered bird-survey recorder: a higher rate costs memory and battery, a lower rate risks aliasing the song into false chirps. The same trade-off exists in telephony (8 kHz), VoIP (16 kHz) and IoT microphones.

## Data used
| Item | Detail |
|---|---|
| Source | ESC-50 environmental sound dataset (K. J. Piczak, 2015), GitHub |
| Clip | `1-34495-A-14` (class: chirping birds), 5 s, 44.1 kHz, 16-bit mono |
| Size / licence | 431 kB WAV, cached in `data/`; CC BY-NC (Freesound) |
| Processing | resampled once to a 48 kHz "analog" reference grid |

## What was done
1. Measured the clip's 99 % bandwidth and Nyquist rate.
2. Sampled it at 16, 8 and 4 kHz with three choices of prefilter: none, a moving average (M = D) and a 255-tap FIR anti-aliasing filter.
3. Reconstructed each sampled version by zero-order hold, linear and sinc interpolation, and scored SNR against the ideal band-limited signal.
4. Studied the moving average's frequency response: nulls at fs/M, the −3 dB point and the 13 dB sidelobe.
5. Verified the theory with 10 unit tests, including alias-frequency prediction.

## Key results
* The song's 99 % bandwidth is 6.63 kHz, so its Nyquist rate is about 13.3 kHz.
* 16 kHz with an FIR prefilter and sinc reconstruction gives 33.4 dB SNR; 8 kHz gives 28.6 dB.
* At 4 kHz, 88 % of the energy lies above Nyquist. The 2,102 Hz song peak aliases to 1,898 Hz.
* The moving average helps at 16 kHz and 4 kHz but hurts at 8 kHz (17.0 → 14.3 dB), because its passband droop removes song energy.
* For reconstruction, sinc beats linear by 12 to 17 dB and zero-order hold by 25 to 27 dB.

## Files and how to run
`python run_experiment.py`, `python gui.py` (the sampling lab, with Play buttons to hear aliasing), `python -m unittest test_sampling` and `python build_report.py` (10 pages).
