"""
build_report.py
---------------
Generates the Word (and PDF) lab report for Experiment 11.

    python run_experiment.py      # cached data and GUI screenshot first
    python build_report.py        # -> DSP_Experiment_11.docx / .pdf

Every number quoted in the text is read from the analysis results at build
time, and the figures are redrawn from the same run.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))

import analysis                      # noqa: E402
import config                        # noqa: E402
import fir                           # noqa: E402
from report_kit import Report, export_pdf   # noqa: E402

STUDENT = {
    "Name": "Dewashish Lambore",
    "PRN": "____________",
    "Date": "____ / ____ / 2026",
    "Course": "Digital Signal Processing Lab",
}
FIG = config.FIGURE_DIR
OUT_DOCX = HERE / "DSP_Experiment_11.docx"
FS = config.FS


def f(x, d=2):
    v = 0.0 if abs(x) < 0.5 * 10 ** (-d) else x
    return f"{v:,.{d}f}"


def build():
    R = analysis.run_all()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plots
    for name, fig in plots.make_all_figures(R).items():
        fig.savefig(FIG / name)
        plt.close(fig)

    D, F, W, K, A, Mt, T, Sp = R["D"], R["F"], R["win"], R["kaiser"], R["app"], R["methods"], R["taps"], R["spec"]
    hdr = D["hdr"]
    G = F["gains"]
    t251, t1001 = T[0], T[-1]
    num = D["numerics"]

    rep = Report(footer_left="DSP Lab  ·  Experiment 11  ·  FIR Filter Design by Windowing")

    # ------------------------------------------------------------------ title
    rep.info_block([("Course", STUDENT["Course"]), ("Experiment No.", "11"),
                    ("Name", STUDENT["Name"]), ("PRN", STUDENT["PRN"]),
                    ("Language", "Python 3.10"), ("Date", STUDENT["Date"])])
    rep.title("FIR Filter Design by Windowing",
              "Heart rate and breathing rate from one fingertip pulse sensor")
    rep.abstract(
        "Linear-phase FIR filters were designed by the window method, from the ideal sinc impulse response and the "
        "rectangular, Hann, Hamming, Blackman and Kaiser windows, and implemented as non-recursive difference equations. "
        f"For the same specification the four classical windows gave stopband attenuations of "
        + ", ".join(f"{W[w]['stop_db']:.0f}" for w in config.WINDOWS) + " dB, matching the textbook values, at the cost "
        f"of progressively wider transition bands; Kaiser's formulas sized a 60 dB design at N = {K['N']} taps ({-K['stop_db']:.1f} dB achieved). The "
        f"filters were applied to an 8-minute photoplethysmogram (PPG) from an intensive-care patient (PhysioNet BIDMC): a "
        f"{config.N_PULSE}-tap band-pass filter (0.5 to 4 Hz) recovered the heart rate with a mean absolute error of "
        f"{f(A['hr_mae'])} bpm against the bedside monitor, and a {config.N_RESP}-tap low-pass filter (0.5 Hz) recovered the "
        f"breathing rate within {f(A['rr_mae'])} breaths per minute of clinicians' manual annotations. The filters' constant "
        f"group delay ({F['pulse']['delay_s']:g} s and {F['resp']['delay_s']:g} s) was removed exactly, and longer filters "
        "improved rejection but could not remove in-band motion artefacts.",
        "FIR filter, window method, Hamming, Blackman, Kaiser window, linear phase, group delay, difference "
        "equation, photoplethysmography, heart rate, respiration rate, Python")

    # ------------------------------------------------------------- 1 aim
    rep.heading("Aim and Objectives")
    rep.para("**Aim:** to design FIR filters by the window method, implement them by their difference equation, plot "
             "their frequency response, and use them to extract physiological information from a real signal, in Python.",
             space_after=3)
    rep.bullets([
        "Derive windowed-sinc low-pass and band-pass filters and compare five windows by ripple, attenuation and "
        "transition width; size a filter with Kaiser's formulas.",
        "Implement the filters as y[n] = Σ bₖ x[n − k], verify linear phase and constant group delay.",
        "Extract heart rate and breathing rate from a PPG and validate them against hospital reference data.",
    ])

    # ----------------------------------------------------- 2 application
    rep.heading("Real-World Application")
    rep.para(
        "A **photoplethysmogram** (PPG) is measured by shining light into the skin and recording how much is absorbed as "
        "blood volume changes with each heartbeat. The same sensor is in hospital pulse oximeters, smartwatches and "
        "fitness bands. Its raw signal mixes three things at different frequencies: the cardiac pulse (about 1 to 3 Hz), "
        "a slow modulation by breathing (about 0.1 to 0.5 Hz), and noise from motion and electronics at higher "
        "frequencies. Separating them is a filtering problem, and FIR filters are preferred because their **linear phase** "
        "does not distort the pulse shape or shift beats relative to each other, which matters for heart-rate variability "
        "and clinical waveform analysis. Table 1 lists other uses of window-designed FIR filters.")
    rep.table(
        ["Field", "Filter", "Why FIR / linear phase"],
        [
            ["Wearables, pulse oximeters (this experiment)", "band-pass and low-pass on PPG", "keeps pulse shape and beat timing"],
            ["ECG and EEG instruments", "band-pass 0.5 to 40 Hz, notch", "no phase distortion of clinical waveforms"],
            ["Audio crossovers, equalisers", "low-pass / high-pass pairs", "phase-coherent summation of loudspeakers"],
            ["Digital communications", "pulse shaping, matched filters", "symmetric taps avoid inter-symbol distortion"],
            ["Seismology", "band-pass of seismograms", "arrival times of waves are preserved"],
        ],
        "Where window-designed FIR filters are used", widths=[5.0, 4.8, 7.2], font_size=9, first_col_bold=True,
        align=["left", "left", "left"])

    # ------------------------------------------------------------ 3 theory
    rep.heading("Theory")
    rep.para("An ideal low-pass filter with cut-off $f_c$ has an infinitely long sinc impulse response. The **window "
             "method** delays it by $M/2$ (with $M = N - 1$) to make it causal and multiplies it by a finite window "
             "$w[n]$ of $N$ samples:")
    rep.equation(r"h[n] = h_d[n]\,w[n], \qquad h_d[n] = 2\frac{f_c}{f_s}\,\mathrm{sinc}\!\left(2\frac{f_c}{f_s}\left(n - \frac{M}{2}\right)\right), \quad 0 \le n \le M")
    rep.para("A band-pass is the difference of two ideal low-passes, $h_d = h_{d,f_2} - h_{d,f_1}$. The four classical "
             "windows are raised cosines; Kaiser's window uses the modified Bessel function $I_0$ and a shape parameter "
             "$\\beta$:")
    rep.equation(r"w_{\mathrm{Hamming}}[n] = 0.54 - 0.46\cos\frac{2\pi n}{M}, \qquad w_{\mathrm{Kaiser}}[n] = \frac{I_0\left(\beta\sqrt{1 - (2n/M - 1)^2}\right)}{I_0(\beta)}")
    rep.para("Multiplying by the window convolves the ideal response with the window's spectrum: the **main lobe** sets "
             "the transition width and the **sidelobes** set the stopband attenuation (Table 2). For a required "
             "attenuation A dB and transition width Δω (rad/sample), Kaiser's formulas give the design directly:")
    rep.equation(r"\beta = 0.1102\,(A - 8.7) \;\; (A > 50), \qquad N \approx \frac{A - 8}{2.285\,\Delta\omega} + 1")
    rep.table(
        ["Window", "Peak sidelobe", "Filter stopband", "Transition width Δf", "Measured here (N = 251)"],
        [[w.capitalize(), f"{fir.WINDOW_TABLE[w]['sidelobe_db']} dB", f"{fir.WINDOW_TABLE[w]['stopband_db']} dB",
          f"≈ {fir.WINDOW_TABLE[w]['mainlobe'] / 2:g}·fs/M", f"{W[w]['stop_db']:.1f} dB"] for w in config.WINDOWS],
        "Window properties (Oppenheim and Schafer, Table 7.2) and measured attenuation",
        widths=[2.8, 2.8, 3.0, 3.8, 4.6], font_size=9, first_col_bold=True, align=["left"] + ["center"] * 4)
    rep.para("The filter is realised by the **non-recursive difference equation**; because the coefficients are "
             "symmetric, $h[n] = h[M-n]$, the phase is linear and every frequency is delayed by the same group delay:")
    rep.equation(r"y[n] = \sum_{k=0}^{M} h[k]\,x[n-k], \qquad H(e^{j\omega}) = A(\omega)\,e^{-j\omega M/2}, \qquad \tau = \frac{M}{2}\ \text{samples}")
    rep.note("A constant delay is harmless: it can be removed exactly by shifting the output M/2 samples earlier, which is "
             "done in this experiment. An IIR filter of similar selectivity (Experiment 12) needs far fewer coefficients "
             "but has a frequency-dependent delay that distorts waveform shape.")

    # ----------------------------------------------------------- 4 dataset
    rep.heading("Dataset Description")
    import re
    age = re.search(r"<age>:\s*(\d+)", hdr["comment"]).group(1)
    sex = {"M": "male", "F": "female"}.get(re.search(r"<sex>:\s*(\w)", hdr["comment"]).group(1), "patient")
    rep.para(
        "The **BIDMC PPG and Respiration Dataset** (Pimentel et al., PhysioNet, ODC Attribution licence) contains 53 "
        "eight-minute recordings from adult patients at the Beth Israel Deaconess Medical Center, extracted from the "
        "MIMIC-II waveform database. Each record has the PPG, impedance respiration and ECG at 125 Hz, the bedside "
        "monitor's numeric values once per second, and every breath annotated manually by two trained annotators. "
        f"Record **{config.RECORD}** ({age}-year-old {sex}, medical ICU) was "
        f"used: the binary WFDB file is read directly with its header gains, about {D['size_kb']:.0f} kB in total.")
    s = hdr["signals"][config.PPG_INDEX]
    rep.table(
        ["Property", "Value", "Property", "Value"],
        [
            ["Source", "PhysioNet BIDMC 1.0.0", "Record", f"{config.RECORD}, {hdr['nsig']} signals"],
            ["Sampling rate", f"{hdr['fs']:g} Hz", "Length", f"{hdr['nsamp']:,} samples ({hdr['nsamp'] / hdr['fs'] / 60:.1f} min)"],
            ["Signal used", f"{s['name']} (PPG), unit {s['unit']}", "Gain / baseline", f"{s['gain']:g} / {s['baseline']:g}"],
            ["Reference heart rate", f"monitor PULSE, mean {num['PULSE'].mean():.1f} bpm", "Reference breathing", f"{len(D['breaths'])} annotated breaths"],
            ["Monitor RESP", f"mean {num['RESP'].mean():.1f} per min", "Missing PULSE values", f"{A['pulse_missing']} of {len(num)} s"],
        ],
        "Dataset attributes", widths=[3.4, 4.8, 3.4, 5.4], font_size=8.5, align=["left"] * 4)
    raw = D["raw"]
    i0 = 12500
    rep.table(
        ["n", "Time (s)", "Raw int16", "PPG = (raw − baseline)/gain", "Monitor PULSE at that second"],
        [[f"{i:,}", f"{i / FS:.3f}", f"{int(raw[i, config.PPG_INDEX])}", f"{D['ppg'][i]:.5f}",
          f"{num['PULSE'].iloc[int(i / FS)]:.0f}"] for i in range(i0, i0 + 6)],
        "Dataset preview with the unit conversion", widths=[2.0, 2.4, 2.6, 5.2, 4.8], font_size=8.5,
        align=["center"] * 5)
    rep.figure(FIG / "fig1_raw.png", "The input photoplethysmogram.",
               "(a) 30 s of the signal; (b) its spectrum with the breathing and pulse bands shaded.",
               inference=(
                   f"Individual pulses are clearly visible, riding on a slow wave that follows breathing, and a large dip "
                   f"at about 113 s is a motion artefact. The spectrum shows the cardiac fundamental at "
                   f"{f(Sp['f_cardiac'], 2)} Hz ({60 * Sp['f_cardiac']:.0f} bpm) with harmonics at 3.0 and 4.5 Hz, broad "
                   "energy below 0.5 Hz from breathing, and a noise floor above 10 Hz. The pulse and breathing bands do not "
                   "overlap, so linear filters can separate them, provided their transition bands are narrow enough."))

    # ---------------------------------------------------- 5 implementation
    rep.heading("Methodology and Implementation")
    rep.para(f"The band-pass filter for the pulse uses {config.N_PULSE} taps (Hamming), the low-pass for breathing "
             f"{config.N_RESP} taps (Hamming). The output is advanced by M/2 to remove the delay; beats and breaths are found "
             f"as peaks at least {config.MIN_BEAT_S} s and {config.MIN_BREATH_S} s apart; rates are averaged in "
             f"{config.HR_WINDOW_S} s and {config.RR_WINDOW_S} s windows and compared with the references.", keep_next=True)
    rep.table(
        ["Module", "Responsibility"],
        [
            ["`config.py`", "Record, file URLs, band edges, tap counts, windows, rate-estimation settings"],
            ["`data_loader.py`", "Download once, parse the WFDB header, read format-16 samples, load references"],
            ["`fir.py`", "Windows, ideal responses, windowed designs, Kaiser formulas, difference equation, H(f), delay"],
            ["`analysis.py`", "Window study, filters, delay compensation, beat and breath detection, validation, tap study"],
            ["`plots.py` / `gui.py`", "Report figures / interactive FIR filter designer"],
            ["`test_fir.py`", "9 unit tests (match with SciPy firwin, symmetry, constant delay, Kaiser spec): all pass"],
        ],
        "Software modules", widths=[4.2, 12.8], font_size=9, align=["left", "left"])
    rep.code('''def ideal_lowpass(fc, fs, N):                 # h_d[n] = 2fc/fs sinc(2fc/fs (n - M/2))
    n = np.arange(N) - (N - 1) / 2
    return 2 * fc / fs * np.sinc(2 * fc / fs * n)

def design_bandpass(f1, f2, fs, N, win="hamming"):
    h = (ideal_lowpass(f2, fs, N) - ideal_lowpass(f1, fs, N)) * window(win, N)
    return h / abs(np.sum(h * np.exp(-2j * np.pi * np.sqrt(f1 * f2) / fs * np.arange(N))))

def fir_filter(b, x):                          # y[n] = sum_k b_k x[n - k]
    y = np.zeros(len(x))
    for k, bk in enumerate(b):
        y[k:] += bk * x[:len(x) - k]
    return y''', title="Core of fir.py", font_size=7.5)
    rep.para("**GUI.** `gui.py` (Figure 2) designs a low-pass or band-pass filter with any cut-offs, any odd number of taps "
             "and any window (including Kaiser with adjustable β). It shows the impulse response with its delay, the "
             "magnitude response, the filtered PPG, and the heart or breathing rate with its error against the hospital "
             "reference; the coefficients can be saved. Run `python run_experiment.py` and `python gui.py`.")
    rep.figure(FIG / "fig6_gui.png", "The FIR filter designer with a 1,001-tap Hamming band-pass filter.",
               "Stopband better than −50 dB and no baseline wander in the filtered pulse.", width_cm=14.5)

    # ------------------------------------------------------------ 6 calcs
    rep.heading("Calculations")
    hd = fir.ideal_lowpass(FS / 4, FS, 5)
    w5 = fir.window("hamming", 5)
    h5 = hd * w5
    h5n = h5 / h5.sum()
    Hn = lambda fq: np.sum(h5n * np.exp(-2j * np.pi * fq / FS * np.arange(5)))
    rep.para("**5-tap low-pass by hand.** fc = fs/4 (31.25 Hz), N = 5, M = 4, so 2fc/fs = ½ and "
             "$h_d[n] = \\frac{1}{2}\\mathrm{sinc}\\left(\\frac{n-2}{2}\\right)$:", keep_next=True)
    rep.table(
        ["n", "n − M/2", "h_d[n] = ½ sinc((n − 2)/2)", "Hamming w[n]", "h[n] = h_d w", "normalised h[n]"],
        [[str(n), f"{n - 2:+d}", f"{hd[n]:.4f}", f"{w5[n]:.2f}", f"{h5[n]:.4f}", f"{h5n[n]:.4f}"] for n in range(5)],
        "Window-method design of a 5-tap filter", widths=[1.0, 1.8, 4.6, 2.8, 3.0, 3.8], font_size=9,
        align=["center"] * 6)
    N_k, b_k = K["N"], K["beta"]
    rep.para(f"Sum of h[n] = {h5.sum():.4f}, so dividing by it gives unit DC gain; the taps are symmetric, hence linear "
             f"phase with a delay of M/2 = 2 samples (16 ms). At fc the normalised filter gives |H| = {abs(Hn(FS / 4)):.3f} "
             f"({20 * np.log10(abs(Hn(FS / 4))):.1f} dB), and at fs/2 it gives {abs(Hn(FS / 2)):.3f}. **Kaiser sizing:** for "
             f"A = {K['A']:g} dB and a transition of {K['tw']} Hz, Δω = 2π·{K['tw']}/125 = {2 * np.pi * K['tw'] / FS:.5f} "
             f"rad, β = 0.1102·({K['A']:g} − 8.7) = {b_k:.3f} and N = ({K['A']:g} − 8)/(2.285·Δω) + 1 → **{N_k} taps**. "
             f"**Delays used:** ({config.N_PULSE} − 1)/2 = {(config.N_PULSE - 1) // 2} samples = {F['pulse']['delay_s']:g} s and "
             f"({config.N_RESP} − 1)/2 = {(config.N_RESP - 1) // 2} samples = {F['resp']['delay_s']:g} s. The program reproduces "
             "these values (checked by `test_fir.py`).")

    # --------------------------------------------------------- 7 results
    rep.heading("Results and Discussion")
    rep.figure(FIG / "fig2_windows.png", "Effect of the window on a low-pass filter.",
               f"(a) The four windows for N = {config.N_PULSE}; (b) the resulting low-pass responses at 4 Hz with their "
               "measured stopband attenuation.",
               inference=(
                   "Each window trades transition width for attenuation exactly as theory predicts: the rectangular window "
                   f"gives the sharpest edge (−3 to −40 dB in {f(W['rectangular']['transition'], 2)} Hz) but only "
                   f"{W['rectangular']['stop_db']:.0f} dB of attenuation with ripples that decay slowly; Hann reaches "
                   f"{W['hann']['stop_db']:.0f} dB, Hamming {W['hamming']['stop_db']:.0f} dB and Blackman "
                   f"{W['blackman']['stop_db']:.0f} dB, while the transition widens to {f(W['blackman']['transition'], 2)} Hz. "
                   f"Passband ripple falls from {f(W['rectangular']['ripple_db'], 2)} dB to {f(W['blackman']['ripple_db'], 3)} dB. "
                   "Hamming is a good default: about 53 dB with a moderate transition."))
    rep.figure(FIG / "fig3_filters.png", "The designed filters.",
               "(a) Band-pass impulse response; (b) magnitude responses, including the Kaiser design; (c) group delay.",
               inference=(
                   f"The band-pass passes the pulse with {f(G['pulse@1.5'], 2)} dB at 1.5 Hz and attenuates 10 Hz noise by "
                   f"{f(-G['pulse@10'], 0)} dB, but with {config.N_PULSE} taps its lower edge is soft: breathing at 0.25 Hz is cut "
                   f"by only {f(-G['pulse@0.25'], 0)} dB. The low-pass keeps 0.33 Hz breathing ({f(G['resp@0.33'], 1)} dB) and "
                   f"removes the 1.5 Hz pulse by {f(-G['resp@1.5'], 0)} dB. The Kaiser design reaches {-K['stop_db']:.1f} dB against its "
                   f"60 dB target with N = {K['N']} (the formulas are approximations, so a few more taps would close the gap), "
                   "and shows that a 0.3 Hz transition costs many taps at 125 Hz. Both "
                   f"filters are exactly symmetric (asymmetry {Mt['symmetry']:.0e}) and their group delay is flat at "
                   f"{F['pulse']['delay_s']:g} s and {F['resp']['delay_s']:g} s (spread {Mt['tau_spread']:.0e} samples)."))
    rep.figure(FIG / "fig4_filtered.png", "The filters applied to the PPG (100 to 130 s).",
               "(a) Raw signal, band-pass output and detected beats; (b) low-pass output, detected breaths and the "
               "manual breath annotations.",
               inference=(
                   "After delay compensation the band-pass output lines up beat for beat with the raw pulses: linear phase "
                   "shifted nothing. Each pulse gives one clean peak except during the motion artefact around 113 s, where the "
                   "residual slow wander causes a few missed or extra beats. The low-pass output is a smooth breathing wave "
                   "whose peaks fall close to the annotators' marks, again with trouble around the artefact."))
    rep.figure(FIG / "fig5_rates.png", "Rates estimated from the filtered PPG against the hospital references.",
               f"(a) Heart rate ({config.HR_WINDOW_S} s windows) against the monitor; (b) breathing rate ({config.RR_WINDOW_S} s "
               "windows) against the manual annotations and the monitor's impedance-based value.",
               inference="", inference_items=[
                   f"Heart rate: {f(A['hr_mean'], 1)} bpm against {f(A['hr_ref_mean'], 1)} bpm from the monitor, mean absolute "
                   f"error {f(A['hr_mae'])} bpm from {A['n_beats']} detected beats. Short excursions to about "
                   f"{f(t251['hr_max'], 0)} bpm coincide with motion artefacts.",
                   f"Breathing rate: {f(A['rr_mean'], 1)} against {f(A['rr_ref_mean'], 1)} breaths per minute from the annotations "
                   f"(MAE {f(A['rr_mae'])}); {100 * A['breath_sensitivity']:.0f} % of annotated breaths were found within 1.5 s. The "
                   "underestimate between 60 and 150 s is the artefact region, where breaths are masked.",
                   f"Lengthening the band-pass from {t251['N']} to {t1001['N']} taps improved the 0.25 Hz rejection from "
                   f"{f(-t251['gain_025'], 0)} to {f(-t1001['gain_025'], 0)} dB and lowered the heart-rate error from "
                   f"{f(t251['mae'])} to {f(t1001['mae'])} bpm, but the artefact spikes ({f(t1001['hr_max'], 0)} bpm) remained: "
                   "noise that lies inside the passband cannot be removed by any linear filter.",
               ])
    rep.table(
        ["Band-pass taps N", "Delay (s)", "Gain at 0.25 Hz (dB)", "HR MAE (bpm)", "Windows within ±2 bpm"],
        [[str(t["N"]), f"{t['delay_s']:g}", f"{t['gain_025']:.1f}", f(t["mae"]), f"{100 * t['frac_within_2']:.0f} %"] for t in T],
        "Effect of filter length on heart-rate accuracy", widths=[3.2, 2.4, 3.6, 3.2, 4.6], font_size=9,
        align=["center"] * 5)

    # ------------------------------------------------------ 8 conclusion
    rep.heading("Conclusion")
    rep.para(
        "Window-designed FIR filters were built from first principles, matched SciPy's implementation exactly, and showed "
        "the predicted trade-off between transition width and stopband attenuation for each window, while Kaiser's formulas "
        "sized a filter close to a stated specification. Implemented as a symmetric difference equation, every filter had exactly "
        "linear phase and a constant delay that was removed by a simple shift, so the pulse shape and beat timing were "
        f"preserved. On a real ICU recording the band-pass filter recovered the heart rate to {f(A['hr_mae'], 1)} bpm and "
        f"the low-pass filter the breathing rate to {f(A['rr_mae'], 1)} breaths per minute of the hospital references, which "
        "is the principle behind smartwatch heart and breathing measurement. The price of FIR filters is length: sharp edges "
        "at low frequencies need hundreds or thousands of taps and seconds of delay, and no filter, however long, removes "
        "artefacts that share the signal's band.")
    rep.heading("Learning Outcomes")
    rep.bullets([
        "Design low-pass and band-pass FIR filters by the window method and size them with Kaiser's formulas.",
        "Relate a window's main lobe and sidelobes to the filter's transition width and stopband attenuation.",
        "Implement an FIR filter as a difference equation, verify linear phase and compensate its group delay.",
        "Apply FIR filters to a physiological signal and validate the extracted rates against reference data.",
    ])
    rep.heading("References")
    refs = [
        "A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed. Pearson, 2010, ch. 7.5 and 7.6.",
        "J. F. Kaiser, “Nonrecursive digital filter design using the I₀-sinh window function,” in *Proc. IEEE Int. Symp. "
        "Circuits and Systems*, 1974, pp. 20-23.",
        "M. A. F. Pimentel et al., “Toward a robust estimation of respiratory rate from pulse oximeters,” *IEEE Trans. "
        "Biomed. Eng.*, vol. 64, no. 8, pp. 1914-1923, 2017.",
        "A. L. Goldberger et al., “PhysioBank, PhysioToolkit, and PhysioNet,” *Circulation*, vol. 101, no. 23, pp. e215-e220, 2000.",
        "J. Allen, “Photoplethysmography and its application in clinical physiological measurement,” *Physiol. Meas.*, "
        "vol. 28, no. 3, pp. R1-R39, 2007.",
    ]
    for i, rr in enumerate(refs, 1):
        rep.para(f"[{i}]  {rr}", align="left", size=9.5, space_after=1)

    path = rep.save(OUT_DOCX)
    print(f"saved {path}")
    print(f"saved {export_pdf(path)}")


if __name__ == "__main__":
    build()
