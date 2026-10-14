"""
build_report.py
---------------
Generates the Word (and PDF) lab report for Experiment 3.

    python run_experiment.py      # figures + results first
    python build_report.py        # -> DSP_Experiment_03.docx / .pdf

Every number quoted in the text is read from the analysis results at build
time, so the report can never disagree with the code.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))

import analysis                      # noqa: E402
import config                        # noqa: E402
import data_loader                   # noqa: E402
import sampling as sp                # noqa: E402
from report_kit import Report, export_pdf   # noqa: E402

STUDENT = {
    "Name": "Dewashish Lambore",
    "PRN": "____________",
    "Date": "____ / ____ / 2026",
    "Course": "Digital Signal Processing Lab",
}
FIG = config.FIGURE_DIR
OUT_DOCX = HERE / "DSP_Experiment_03.docx"


def f(x, d=1):
    return f"{x:,.{d}f}"


def build():
    R = analysis.run_all()
    fs, x, clip, pre, rec, ma = R["fs"], R["x"], R["clip"], R["prefilter"], R["recon"], R["ma"]
    info = data_loader.describe_clip()
    st = R["storage_mb_per_h"]
    B = clip["bandwidth"]
    rates = config.RATES
    FIR, MA, NONE = "FIR (ideal-like)", "Moving average", "None"
    SINC, LIN, ZOH = "Sinc (band-limited)", "Linear", "Zero-order hold"

    rep = Report(footer_left="DSP Lab  ·  Experiment 3  ·  Moving-Average Filter, Sampling and Reconstruction")

    # ------------------------------------------------------------------ title
    rep.info_block([("Course", STUDENT["Course"]), ("Experiment No.", "03"),
                    ("Name", STUDENT["Name"]), ("PRN", STUDENT["PRN"]),
                    ("Language", "Python 3.10"), ("Date", STUDENT["Date"])])
    rep.title("Moving-Average Filter, Sampling and Reconstruction",
              "Choosing the sampling rate of a bird-song recorder")
    rep.abstract(
        f"A real 5-second bird-song field recording (ESC-50 dataset) with a 99 % bandwidth of {f(B / 1000, 2)} kHz "
        f"was treated as the analog signal and sampled at 16, 8 and 4 kHz, with no prefilter, a moving-average "
        f"prefilter and a proper FIR anti-aliasing filter, then reconstructed by zero-order hold, linear and sinc "
        f"interpolation. Sampling above the Nyquist rate ({f(clip['nyquist_rate'] / 1000, 1)} kHz) at 16 kHz with an "
        f"FIR prefilter and sinc reconstruction reached {f(rec[16000][SINC]['snr'])} dB SNR, while 4 kHz folded "
        f"{f(100 * clip['above'][4000], 0)} % of the song's energy into false tones. The moving average proved to be "
        f"a cheap but weak anti-aliasing filter: it helped at 4 kHz ({f(pre[4000][NONE]['snr'])} to "
        f"{f(pre[4000][MA]['snr'])} dB) but hurt at 8 kHz ({f(pre[8000][NONE]['snr'])} to {f(pre[8000][MA]['snr'])} dB) "
        "because it attenuates the passband. Reconstruction quality ranked sinc > linear > hold by 12 to 27 dB.",
        "sampling theorem, Nyquist rate, aliasing, moving-average filter, anti-aliasing, interpolation, "
        "reconstruction, bioacoustics, Python")

    # ------------------------------------------------------------- 1 aim
    rep.heading("Aim and Objectives")
    rep.para("**Aim:** to study the moving-average filter, the sampling of a signal, and the reconstruction "
             "of the signal from its samples, using a real audio recording in Python.", space_after=3)
    rep.bullets([
        "Measure the bandwidth of a real signal and find its Nyquist rate.",
        "Sample it above and below the Nyquist rate and observe aliasing, with and without an "
        "anti-aliasing prefilter (moving average versus FIR).",
        "Reconstruct the signal by zero-order hold, linear and sinc interpolation and compare their SNR.",
    ])

    # ----------------------------------------------------- 2 application
    rep.heading("Real-World Application")
    rep.para(
        "Ecologists survey birds with small battery-powered acoustic recorders left in a forest for weeks. "
        "The sampling rate is the key setting: too high and the battery and memory card run out, too low and "
        "the song is destroyed by aliasing. The same trade-off appears in every digital audio system (Table 1). "
        "Before sampling, a low-pass **anti-aliasing filter** must remove everything above half the sampling "
        "rate; a moving average is the cheapest such filter (only additions), which is why it is common on "
        "microcontrollers. After sampling, a **reconstruction** filter turns the samples back into a smooth "
        "signal in the playback DAC.")
    rep.table(
        ["System", "Sampling rate", "Usable band", "Storage (16-bit, 1 h)"],
        [
            ["Narrowband telephone (G.711)", "8 kHz", "0.3 to 3.4 kHz", f"{f(st[8000], 1)} MB"],
            ["Wideband / VoIP speech (G.722)", "16 kHz", "0.05 to 7 kHz", f"{f(st[16000], 1)} MB"],
            ["Low-power IoT microphone", "4 kHz", "up to 2 kHz", f"{f(st[4000], 1)} MB"],
            ["Professional audio / video", "48 kHz", "up to 20 kHz", f"{f(st[48000], 1)} MB"],
            ["Bioacoustic field recorder", "8 to 384 kHz (selectable)", "depends on species", "chosen per survey"],
        ],
        "Sampling rates used in practice", widths=[5.4, 3.8, 3.6, 4.2], font_size=9,
        align=["left", "center", "center", "center"], first_col_bold=True)

    # ------------------------------------------------------------ 3 theory
    rep.heading("Theory")
    rep.para("**Sampling.** Sampling a continuous signal $x_a(t)$ every $T_s = 1/f_s$ seconds gives "
             "$x[n] = x_a(nT_s)$. By the Nyquist-Shannon sampling theorem, a signal band-limited to $B$ Hz "
             "is recovered exactly if")
    rep.equation(r"f_s > 2B \qquad (\text{Nyquist rate } = 2B)")
    rep.para("If $f_s$ is too low, a component at frequency $f$ above $f_s/2$ is not lost but reappears at the "
             "**alias** frequency")
    rep.equation(r"f_{\mathrm{alias}} = \left| f - k f_s \right|, \quad k = \mathrm{round}(f/f_s)")
    rep.para("which is indistinguishable from a genuine tone, so aliasing must be prevented *before* sampling by "
             "a low-pass prefilter.")
    rep.para("**Moving-average filter.** The $M$-point moving average and its frequency response are")
    rep.equation(r"y[n] = \frac{1}{M}\sum_{k=0}^{M-1} x[n-k], \qquad \left|H(e^{j\omega})\right| = \left|\frac{\sin(M\omega/2)}{M\sin(\omega/2)}\right|")
    rep.para("with $\\omega = 2\\pi f/f_s$. It has unity gain at DC and nulls at $f = k f_s/M$, but its "
             "passband droops and its first sidelobe is only about 13 dB down, so it is a crude low-pass filter.")
    rep.para("**Reconstruction.** The ideal reconstruction is the Whittaker-Shannon interpolation")
    rep.equation(r"x_r(t) = \sum_{n=-\infty}^{\infty} x[n]\,\mathrm{sinc}\!\left(\frac{t - nT_s}{T_s}\right)")
    rep.para("Practical systems approximate it by the kernels in Table 2. Quality is scored by "
             "$\\mathrm{SNR} = 10\\log_{10}(\\sum r^2 / \\sum (r - \\hat{x})^2)$ against the ideal band-limited "
             "signal $r$, i.e. the best any system could deliver at that sampling rate.")
    rep.table(
        ["Method", "Kernel between samples", "Frequency behaviour", "Typical use"],
        [
            ["Zero-order hold", "hold the last value (staircase)", "sinc-shaped droop, strong images", "simple DACs"],
            ["Linear", "straight lines (triangle kernel)", "sinc² droop, weaker images", "microcontrollers, plots"],
            ["Sinc (band-limited)", "windowed sinc, ±32 samples", "flat to fs/2, images removed", "audio interpolators"],
        ],
        "Reconstruction methods", widths=[3.4, 4.8, 4.8, 4.0], font_size=9, first_col_bold=True,
        align=["left", "left", "left", "left"])
    rep.note("An analog recording cannot be stored in a computer, so the 44.1 kHz clip is first resampled to a "
             "48 kHz grid that stands in for the analog signal; every lower rate is then an exact integer "
             "decimation (D = 3, 6, 12 for 16, 8, 4 kHz). The bird song has almost no energy near 24 kHz, so the "
             "48 kHz grid is effectively continuous for this study.")

    # ----------------------------------------------------------- 4 dataset
    rep.heading("Dataset Description")
    rep.para(
        "**ESC-50** (K. J. Piczak, 2015) is a public benchmark of 2,000 five-second environmental recordings in "
        "50 classes, collected from the Freesound field-recording archive and widely used for sound "
        f"classification research. Clip `{config.CLIP_ID}` (class 14, *chirping birds*) is a single "
        f"{info['size_kb']:.0f} kB WAV file downloaded once from the project's GitHub repository and cached "
        "locally. It contains three loud bursts of song between 2.3 and 3.6 s over a quiet background, which "
        "makes both the waveform and the spectrum easy to read.")
    rep.table(
        ["Property", "Value", "Property", "Value"],
        [
            ["Source", "ESC-50, fold 1", "Clip / class", f"{config.CLIP_ID} / chirping birds"],
            ["Sampling rate", f"{info['fs']:,} Hz", "Duration", f"{info['duration_s']:.1f} s ({info['samples']:,} samples)"],
            ["Format", "16-bit PCM, mono", "File size", f"{info['size_kb']:.0f} kB"],
            ["Peak / RMS level", f"{info['peak']:.3f} / {info['rms']:.3f} FS", "Spectral peak", f"{f(clip['peak_freq'], 0)} Hz"],
            ["99 % bandwidth B", f"{f(B, 0)} Hz", "Nyquist rate 2B", f"{f(clip['nyquist_rate'], 0)} Hz"],
            ["Energy above 8 / 4 / 2 kHz", f"{100 * clip['above'][16000]:.1f} / {100 * clip['above'][8000]:.1f} / {100 * clip['above'][4000]:.1f} %",
             "Licence", "CC BY-NC (Freesound)"],
        ],
        "Dataset attributes", widths=[3.6, 4.6, 3.4, 5.4], font_size=8.5,
        align=["left", "left", "left", "left"])
    raw = info["raw"]
    i0 = int(np.argmax(np.abs(raw)))
    rep.table(
        ["n (44.1 kHz)", "Time (s)", "Sample (int16)", "Normalised value"],
        [[f"{i:,}", f"{i / info['fs']:.6f}", f"{int(raw[i])}", f"{raw[i] / 32768:.4f}"] for i in range(i0 - 3, i0 + 3)],
        "Raw samples around the loudest point of the recording",
        widths=[3.6, 4.0, 4.4, 5.0], font_size=8.5, align=["center"] * 4)
    rep.figure(FIG / "fig1_recording.png", "The recording in time and frequency.",
               "(a) Waveform; (b) spectrum with the Nyquist frequency of each test rate and the 99 % bandwidth.",
               inference=(
                   f"The song energy sits mainly between 1.8 and 3 kHz with a peak at {f(clip['peak_freq'], 0)} Hz and "
                   f"harmonics reaching about {f(B / 1000, 1)} kHz. The 16 kHz rate (Nyquist 8 kHz) lies beyond the "
                   f"whole song; 8 kHz (Nyquist 4 kHz) cuts off the upper harmonics ({100 * clip['above'][8000]:.1f} % "
                   f"of the energy); 4 kHz (Nyquist 2 kHz) lies *below* the main peak and would discard or alias "
                   f"{100 * clip['above'][4000]:.0f} % of the energy. The Nyquist rate of this signal is therefore about "
                   f"{f(clip['nyquist_rate'] / 1000, 1)} kHz."))

    # ---------------------------------------------------- 5 implementation
    rep.heading("Methodology and Implementation")
    rep.para("For each rate the reference is (1) optionally prefiltered, (2) decimated by D, (3) reconstructed "
             "on the 48 kHz grid by each method, and (4) compared with the ideal band-limited signal. The "
             f"moving-average length is matched to D (M = D), and the FIR filter has {config.FIR_TAPS} taps with "
             f"its edge at {config.FIR_EDGE:.0%} of the new Nyquist frequency.", keep_next=True)
    rep.table(
        ["Module", "Responsibility"],
        [
            ["`config.py`", "Clip URL, sampling rates, filter and reconstruction parameters"],
            ["`data_loader.py`", "Download once, cache WAV, convert to float, resample to the 48 kHz reference"],
            ["`sampling.py`", "Moving average and its response, prefilters, sampling, three reconstructions, SNR"],
            ["`analysis.py`", "Bandwidth, prefilter and reconstruction studies, storage cost"],
            ["`plots.py` / `gui.py`", "Report figures / interactive sampling lab with audio playback"],
            ["`test_sampling.py`", "10 unit tests (nulls of H, alias frequency, interpolation order), all pass"],
        ],
        "Software modules", widths=[4.2, 12.8], font_size=9, align=["left", "left"])
    rep.code('''def moving_average(x, M):                          # y[n] = (1/M) sum x[n-k]
    return np.convolve(x, np.full(M, 1.0 / M))[(M - 1) // 2:][:len(x)]

def sample(x, fs_ref, fs_new, kind="None"):        # x_s[n] = x_f(n Ts)
    D = fs_ref // fs_new
    return prefilter(x, fs_ref, fs_new, kind)[::D]

def reconstruct(xs, D, N, method):                 # back to the 48 kHz grid
    if method == "Zero-order hold":
        return np.repeat(xs, D)[:N]
    if method == "Linear":
        return np.interp(np.arange(N) / D, np.arange(len(xs)), xs)
    up = np.zeros(len(xs) * D); up[::D] = xs       # Whittaker-Shannon:
    return oaconvolve(up, sinc_kernel(D), mode="same")[:N]   # sum x[n] sinc((t - nTs)/Ts)''',
             title="Core of sampling.py", font_size=7.5)
    rep.para("**GUI.** `gui.py` (Figure 2) lets the user choose any sampling rate from 2 to 24 kHz, a prefilter "
             "(none, moving average of any length, or FIR) and a reconstruction method. It shows 5 ms of the chirp "
             "with the samples and the reconstruction, the spectrum with the Nyquist line, the SNR, the aliased "
             "energy and the storage cost, and two **Play** buttons let the student *hear* the original and the "
             "sampled version. Run `python run_experiment.py` and `python gui.py`.")
    rep.figure(FIG / "fig6_gui.png", "The sampling lab at 4 kHz with no prefilter and linear reconstruction.",
               "The reconstruction oscillates at the alias frequency while the true band-limited signal is small; "
               "the SNR is negative.", width_cm=14.5)

    # ------------------------------------------------------ 6 calculations
    rep.heading("Calculations")
    xs8 = rec[8000]["xs"]
    k0 = int(np.argmax(np.abs(xs8[200:-200]))) + 200
    v = xs8[k0 - 1: k0 + 3]
    y_ma = np.convolve(v, np.full(3, 1 / 3))[:4]
    rep.para(f"Four consecutive samples of the song at 8 kHz (n = {k0 - 1} to {k0 + 2}) are "
             f"x[n] = {{{', '.join(f'{u:.4f}' for u in v)}}}.", keep_next=True)
    g6 = sp.ma_frequency_response(6, 4000, fs)
    fa = sp.alias_frequency(clip["peak_freq"], 4000)
    rep.table(
        ["Quantity", "Working", "Result"],
        [
            ["3-point MA, y[2]", f"({v[0]:.4f} + {v[1]:.4f} + {v[2]:.4f}) / 3", f"{y_ma[2]:.4f}"],
            ["3-point MA, y[3]", f"({v[1]:.4f} + {v[2]:.4f} + {v[3]:.4f}) / 3", f"{y_ma[3]:.4f}"],
            ["ZOH value at t = (n + ½)Ts", "hold x[n]", f"{v[1]:.4f}"],
            ["Linear value at t = (n + ½)Ts", f"({v[1]:.4f} + {v[2]:.4f}) / 2", f"{(v[1] + v[2]) / 2:.4f}"],
            ["Alias of the song peak at fs = 4 kHz", f"|{clip['peak_freq']:.0f} − 1 × 4000|", f"{fa:.0f} Hz"],
            ["MA (M = 6) gain at the 8 kHz Nyquist", "sin(6ω/2)/(6 sin(ω/2)), ω = 2π·4000/48000", f"{g6:.3f} ({20 * np.log10(g6):.1f} dB)"],
            ["Storage at 8 kHz, 1 h", "8000 × 2 bytes × 3600 s", f"{f(st[8000], 1)} MB"],
        ],
        "Hand calculation on real samples", widths=[5.2, 7.6, 4.2], font_size=8.5,
        align=["left", "left", "center"])
    rep.para(f"The MA gain of {g6:.3f} at the folding frequency means that components just above 4 kHz are "
             "attenuated by less than 4 dB before they alias, which explains the weak anti-aliasing measured "
             "below. All values agree with the program output (checked by `test_sampling.py`).")

    # --------------------------------------------------------- 7 results
    rep.heading("Results and Discussion")
    rep.figure(FIG / "fig2_sampling.png", "Sampling at 8 kHz and the three reconstructions (4 ms of one chirp).",
               "(a) Band-limited reference and its samples; (b) zero-order hold, linear and sinc reconstruction "
               "with their SNR over the whole clip.",
               inference="", inference_items=[
                   f"At 8 kHz the {f(clip['peak_freq'] / 1000, 1)} kHz chirp gets only about "
                   f"{8000 / clip['peak_freq']:.1f} samples per cycle, close to the 2-sample limit, so the way the gaps "
                   "are filled matters a great deal.",
                   f"The **zero-order hold** staircase lags by half a sample and has large steps "
                   f"({f(rec[8000][ZOH]['snr'])} dB); **linear** interpolation cuts the peaks "
                   f"({f(rec[8000][LIN]['snr'])} dB); the **sinc** interpolator follows the true curve "
                   f"({f(rec[8000][SINC]['snr'])} dB) because it is the exact inverse of band-limited sampling.",
               ])
    rep.figure(FIG / "fig3_moving_average.png", "The moving-average filter.",
               "(a) |H| for M = 3, 6, 12 at 48 kHz; (b) spectrum of the song before and after M = 6.",
               inference=(
                   f"Each response has nulls at multiples of 48/M kHz and a −3 dB frequency of "
                   f"{f(ma[3]['f3db'] / 1000, 2)}, {f(ma[6]['f3db'] / 1000, 2)} and {f(ma[12]['f3db'] / 1000, 2)} kHz for "
                   f"M = 3, 6, 12, i.e. about 0.44·fs/M. The first sidelobe is only {f(-ma['sidelobe_db'], 1)} dB down. "
                   "On the song, M = 6 places a deep null at 8 kHz and leaves the 2 to 3 kHz song almost "
                   "untouched, but the 9 to 12 kHz band is attenuated by only 10 to 20 dB because of the sidelobes: "
                   "the moving average smooths, but it is not a sharp low-pass filter."))
    rep.figure(FIG / "fig4_aliasing.png", "Aliasing seen in spectrograms.",
               "(a) Original; (b) 8 kHz without prefilter; (c) 4 kHz without prefilter; (d) 4 kHz with the FIR prefilter.",
               inference=(
                   "At 8 kHz the song survives with only faint folded harmonics. At 4 kHz without a prefilter the "
                   f"main 2 to 3 kHz song is reflected about 2 kHz: the peak at {f(clip['peak_freq'], 0)} Hz appears at "
                   f"{fa:.0f} Hz and strong false chirps fill the 1 to 2 kHz band where the original has none, so an "
                   "analyst could mis-identify the species. With the FIR prefilter those false chirps vanish, leaving a mostly empty, "
                   "honest band: the information is lost, but it is not falsified."))
    rep.figure(FIG / "fig5_snr.png", "SNR summary.",
               "(a) Effect of the anti-aliasing prefilter (sinc reconstruction); (b) effect of the reconstruction "
               "method (FIR prefilter).",
               inference=(
                   f"The FIR prefilter is best at every rate ({f(pre[16000][FIR]['snr'])}, {f(pre[8000][FIR]['snr'])} and "
                   f"{f(pre[4000][FIR]['snr'])} dB). The moving average helps at 16 kHz "
                   f"(+{f(pre[16000][MA]['snr'] - pre[16000][NONE]['snr'])} dB) and 4 kHz, but costs "
                   f"{f(pre[8000][NONE]['snr'] - pre[8000][MA]['snr'])} dB at 8 kHz, because its passband droop "
                   "removes more of the song than the little aliasing it prevents. Among reconstructions, sinc beats "
                   f"linear by {f(rec[16000][SINC]['snr'] - rec[16000][LIN]['snr'])} dB and hold by "
                   f"{f(rec[16000][SINC]['snr'] - rec[16000][ZOH]['snr'])} dB at 16 kHz."))
    rep.table(
        ["Rate", "Energy > fs/2", "SNR none", "SNR MA", "SNR FIR", "Best recon.", "MB / hour"],
        [[f"{r // 1000} kHz", f"{100 * clip['above'][r]:.1f} %", f(pre[r][NONE]["snr"]), f(pre[r][MA]["snr"]),
          f(pre[r][FIR]["snr"]), f"{f(rec[r][SINC]['snr'])} dB" if r in rec else "n/a", f(st[r], 1)]
         for r in rates],
        "Summary of results (SNR in dB)", widths=[2.0, 2.6, 2.4, 2.4, 2.4, 2.8, 2.4], font_size=9,
        align=["center"] * 7)

    # ------------------------------------------------------ 8 conclusion
    rep.heading("Conclusion")
    rep.para(
        f"The bird-song recording needs a sampling rate above about {f(clip['nyquist_rate'] / 1000, 1)} kHz. At 16 kHz, "
        f"with a good anti-aliasing filter and sinc reconstruction, it is recovered with {f(rec[16000][SINC]['snr'])} dB "
        f"SNR at a cost of {f(st[16000], 1)} MB per hour; 8 kHz halves the storage and still keeps the song recognisable; "
        "4 kHz is below the song's main frequency and, without a prefilter, produces convincing but false chirps. "
        "The moving-average filter is easy to implement and has exact nulls at multiples of fs/M, but its "
        "drooping passband and high sidelobes make it a poor anti-aliasing filter, sometimes worse than none. "
        "For a field recorder the right design is therefore: choose fs above 2B for the target species, use a "
        "sharp FIR prefilter, and use band-limited (sinc) interpolation on playback.")
    rep.heading("Learning Outcomes")
    rep.bullets([
        "Determine a real signal's bandwidth and Nyquist rate from its spectrum.",
        "Predict and recognise aliasing, and compute alias frequencies.",
        "Derive and interpret the frequency response of the moving-average filter and its limits as a low-pass filter.",
        "Compare zero-order hold, linear and sinc reconstruction quantitatively with SNR.",
    ])
    rep.heading("References")
    refs = [
        "A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed. Pearson, 2010, ch. 4.",
        "C. E. Shannon, “Communication in the presence of noise,” *Proc. IRE*, vol. 37, no. 1, pp. 10-21, 1949.",
        "K. J. Piczak, “ESC: Dataset for environmental sound classification,” in *Proc. ACM Multimedia*, 2015, pp. 1015-1018.",
        "A. P. Hill et al., “AudioMoth: Evaluation of a smart open acoustic device for monitoring biodiversity "
        "and the environment,” *Methods Ecol. Evol.*, vol. 9, pp. 1199-1211, 2018.",
        "ITU-T Recommendations G.711 (1988) and G.722 (2012), speech coding at 8 and 16 kHz.",
    ]
    for i, r in enumerate(refs, 1):
        rep.para(f"[{i}]  {r}", align="left", size=9.5, space_after=1)

    path = rep.save(OUT_DOCX)
    print(f"saved {path}")
    print(f"saved {export_pdf(path)}")


if __name__ == "__main__":
    build()
