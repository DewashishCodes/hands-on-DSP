# Experiment 9: Linear Convolution using the DFT

## What the experiment is about
Computing linear convolution as IDFT{X·H} with zero padding to N ≥ L + M − 1; time aliasing when N is too small; FFT-friendly lengths; overlap-add block convolution; and the cost of direct versus FFT-based convolution.

## Real-world context
Convolution reverb: a dry recording convolved with a measured room impulse response sounds as if it had been performed in that room. It is used in music, film, games and VR, and acoustic consultants use it to let clients hear a hall before it is built.

## Data used
| Item | Detail |
|---|---|
| Source | Communication Acoustics exercises, University of Rostock (GitHub: spatialaudio/communication-acoustics-exercises), CC0 |
| Dry speech | `xmas.wav`: a male voice saying "Wishing you a Merry Christmas...", 114,400 samples |
| Room impulse response | `rir_clap.wav`: a seminar room measured by clapping two wooden boards, 44,100 samples (1 s) |
| Reference | `xmas_clap.wav`: the authors' own published convolution of the two |
| Format / size | 44.1 kHz, 16-bit mono, about 620 kB in total, cached in `data/` |

## What was done
1. Implemented direct convolution, DFT convolution with chosen N, an FFT-friendly length finder and overlap-add.
2. Compared the results with direct convolution and with the published reverberant file.
3. Reduced N step by step to show time aliasing; ran overlap-add with blocks of 256 to 16,384 samples; and timed every method.
4. Measured the room's reverberation time with Schroeder's method, and removed the room again by regularised spectral division.
5. Verified the theory with 8 unit tests.

## Key results
* The DFT result equals direct convolution to 8e-16, and matches the published file to 70.9 dB (16-bit rounding).
* Timing: about 48 s for direct convolution, 176 ms for an FFT of length 158,499, and 15 ms at the FFT-friendly 160,000 (2⁸ × 5⁴).
* With N = L the room tail wraps onto the start (error 5.7e-2); only N = L + M − 1 is exact.
* Overlap-add is exact for every block size. The trade-off is latency (6 ms) against cost (138 multiplications per sample).
* The room's RT60 is 0.63 s (T20).
* Deconvolution recovers the dry voice to 50 dB on clean data, but only 19 dB with 40 dB noise.

## Files and how to run
`python run_experiment.py` (also writes `outputs/reverberant.wav`), `python gui.py` (convolution reverb with playback), `python -m unittest test_fastconv` and `python build_report.py` (11 pages).
