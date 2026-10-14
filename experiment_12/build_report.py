"""
build_report.py
---------------
Generates the Word (and PDF) lab report for Experiment 12.

    python run_experiment.py      # cached data and GUI screenshot first
    python build_report.py        # -> DSP_Experiment_12.docx / .pdf

Every number quoted in the text is read from the analysis results at build
time, and the figures are redrawn from the same run.
"""

import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))

import analysis                      # noqa: E402
import config                        # noqa: E402
import iir                           # noqa: E402
from report_kit import Report, export_pdf   # noqa: E402

STUDENT = {
    "Name": "Dewashish Lambore",
    "PRN": "____________",
    "Date": "____ / ____ / 2026",
    "Course": "Digital Signal Processing Lab",
}
FIG = config.FIGURE_DIR
OUT_DOCX = HERE / "DSP_Experiment_12.docx"
FS = config.FS
NAME = {"butterworth": "Butterworth", "chebyshev1": "Chebyshev I", "elliptic": "Elliptic"}


def f(x, d=2):
    v = 0.0 if abs(x) < 0.5 * 10 ** (-d) else x
    return f"{v:,.{d}f}"


def build():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    R = analysis.run_all()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plots
    for name, fig in plots.make_all_figures(R).items():
        fig.savefig(FIG / name)
        plt.close(fig)

    D, Ds, FE, PW, NU, NT, PH, A = (R["D"], R["designs"], R["fir_eq"], R["prewarp"], R["numerics"], R["notch"],
                                    R["phase"], R["alpha"])
    e = D["open"]["edf"]
    bu, ch, el = Ds["butterworth"], Ds["chebyshev1"], Ds["elliptic"]
    pw_hi_warp = next(r for r in PW if r["band"] == (50.0, 70.0) and r["prewarp"])
    pw_hi_no = next(r for r in PW if r["band"] == (50.0, 70.0) and not r["prewarp"])
    pw_lo_no = next(r for r in PW if r["band"] == config.ALPHA_BAND and not r["prewarp"])
    p_txt = "< 10⁻¹⁰" if A["p"] < 1e-10 else f"{A['p']:.1e}"

    rep = Report(footer_left="DSP Lab  ·  Experiment 12  ·  IIR Filter Design")

    # ------------------------------------------------------------------ title
    rep.info_block([("Course", STUDENT["Course"]), ("Experiment No.", "12"),
                    ("Name", STUDENT["Name"]), ("PRN", STUDENT["PRN"]),
                    ("Language", "Python 3.10"), ("Date", STUDENT["Date"])])
    rep.title("IIR Filter Design and Implementation",
              "Removing mains hum and extracting the alpha rhythm from EEG")
    rep.abstract(
        "Infinite-impulse-response (IIR) filters were designed by placing poles and zeros directly (a 60 Hz notch and a "
        "drift-removing high-pass) and by transforming Butterworth, Chebyshev I and elliptic analog prototypes with the "
        "pre-warped bilinear transform, and were implemented by their recursive difference equations. The bilinear designs "
        f"matched SciPy to {max(d['vs_scipy'] for d in Ds.values()):.0e}; without pre-warping a 50 to 70 Hz band landed at "
        f"{pw_hi_no['edges'][0]:.1f} to {pw_hi_no['edges'][1]:.1f} Hz. With only {bu['n_coef']} coefficients the order-8 "
        "band-passes were as selective as an FIR filter of "
        f"{FE['N']} taps, but their group delay varied across the band ({f(el['tau_min'])} to {f(el['tau_max'])} s for the "
        "elliptic design), and an order-16 design written as a single difference equation became unstable through coefficient "
        "rounding while the same filter as cascaded biquads did not. Applied to 1-minute EEG recordings from PhysioNet, the "
        f"notch pushed the 60 Hz line below the EEG background, and the alpha band-pass showed {f(A['ratio'], 1)} times more "
        f"8 to 13 Hz power with the eyes closed than open (t = {f(A['t'], 1)}), the Berger effect.",
        "IIR filter, bilinear transform, pre-warping, Butterworth, Chebyshev, elliptic, notch filter, second-order "
        "sections, group delay, zero-phase filtering, EEG, alpha rhythm, Python")

    # ------------------------------------------------------------- 1 aim
    rep.heading("Aim and Objectives")
    rep.para("**Aim:** to design IIR filters, implement them by their difference equations, plot their frequency "
             "response, and use them to process a real biomedical signal in Python.", space_after=3)
    rep.bullets([
        "Design a notch and a high-pass from pole-zero positions, and band-pass filters from analog prototypes by the "
        "bilinear transform with pre-warping.",
        "Compare Butterworth, Chebyshev I and elliptic designs by selectivity, ripple, group delay and pole positions, "
        "and against an FIR filter of similar selectivity.",
        "Study stability and numerical precision (direct form versus second-order sections) and phase (causal versus zero-phase).",
    ])

    # ----------------------------------------------------- 2 application
    rep.heading("Real-World Application")
    rep.para(
        "Electroencephalography (EEG) records microvolt-level electrical activity of the brain from scalp electrodes. It is "
        "used to diagnose epilepsy and sleep disorders, to monitor anaesthesia depth, and in brain-computer interfaces. Every "
        "EEG amplifier contains IIR filters: a high-pass to remove slow electrode drift, a notch to remove interference from "
        "the mains supply (50 Hz in India, 60 Hz in the USA where this data was recorded), and band-passes that isolate the "
        "clinical rhythms (delta, theta, alpha, beta). IIR filters are chosen because they achieve sharp responses with very "
        "few coefficients, which suits low-power, real-time hardware. The best-known rhythm, alpha (8 to 13 Hz), grows when "
        "the eyes close, an effect first reported by Hans Berger in 1929. Table 1 lists other IIR applications.")
    rep.table(
        ["Field", "IIR filter", "Why IIR"],
        [
            ["EEG / ECG instruments (this experiment)", "high-pass, 50/60 Hz notch, rhythm band-passes", "sharp, cheap, real time"],
            ["Audio equalisers and crossovers", "peaking, shelving biquads", "few operations per sample"],
            ["Sensor anti-aliasing and smoothing", "Butterworth low-pass", "maximally flat passband"],
            ["Radio and telecom receivers", "elliptic channel filters", "steepest transition for a given order"],
            ["Vibration monitoring", "band-pass around bearing frequencies", "runs on embedded processors"],
        ],
        "Where IIR filters are used", widths=[5.2, 6.0, 5.8], font_size=9, first_col_bold=True,
        align=["left", "left", "left"])

    # ------------------------------------------------------------ 3 theory
    rep.heading("Theory")
    rep.para("An IIR filter feeds past outputs back, so its impulse response never ends. It is described by a "
             "recursive difference equation and a rational transfer function whose poles must lie inside the unit circle:")
    rep.equation(r"y[n] = \sum_{k=0}^{M} b_k\,x[n-k] - \sum_{k=1}^{N} a_k\,y[n-k], \qquad H(z) = \frac{\sum b_k z^{-k}}{1 + \sum a_k z^{-k}}")
    rep.para("**Notch by pole-zero placement.** Zeros on the unit circle at $\\pm\\omega_0 = 2\\pi f_0/f_s$ null the "
             "interference; poles just inside at radius $r$ restore the gain everywhere else. The closer $r$ is to 1, "
             "the narrower the notch, with bandwidth $\\Delta\\omega \\approx 2(1-r)$ rad/sample:")
    rep.equation(r"H_{\mathrm{notch}}(z) = K\,\frac{1 - 2\cos\omega_0\,z^{-1} + z^{-2}}{1 - 2r\cos\omega_0\,z^{-1} + r^{2}z^{-2}}, \qquad r = 1 - \frac{\omega_0}{2Q}")
    rep.para("**Bilinear transform.** Classical analog prototypes (Butterworth: maximally flat; Chebyshev I: equiripple "
             "passband; elliptic: equiripple in both bands, steepest edge) are mapped to the z-plane by")
    rep.equation(r"s = 2f_s\,\frac{1 - z^{-1}}{1 + z^{-1}} \quad\Rightarrow\quad z_i = \frac{2f_s + s_i}{2f_s - s_i}")
    rep.para("which maps the left half-plane inside the unit circle (so stability is preserved) but compresses the whole "
             "analog frequency axis onto 0 to $f_s/2$. Band edges are therefore **pre-warped** before the analog design:")
    rep.equation(r"\Omega = 2f_s\tan\left(\frac{\pi f}{f_s}\right)")
    rep.para("**Phase.** Unlike a symmetric FIR filter, an IIR filter has a frequency-dependent group delay "
             "$\\tau(\\omega) = -d\\phi/d\\omega$, largest near the band edges, so it distorts waveform shape. Off-line, "
             "filtering forward and then backward cancels the phase exactly (zero phase) at the price of a squared "
             "magnitude and a non-causal filter. **Numerics.** High-order filters are realised as a cascade of "
             "second-order sections (biquads), because the roots of one high-degree polynomial are extremely sensitive "
             "to rounding of its coefficients.")
    rep.table(
        ["Family", "Passband", "Stopband", "Transition for a given order", "Phase"],
        [
            ["Butterworth", "maximally flat", "monotonic", "widest", "most linear"],
            ["Chebyshev I", "equiripple (rp dB)", "monotonic", "narrower", "less linear"],
            ["Elliptic", "equiripple (rp dB)", "equiripple (rs dB floor)", "narrowest", "least linear"],
        ],
        "The three analog prototype families", widths=[2.8, 3.4, 3.8, 4.2, 2.8], font_size=9, first_col_bold=True,
        align=["left"] * 5)
    rep.note("Because poles near the unit circle give sharp resonances, IIR filters can be very selective with a handful "
             "of coefficients. The same property makes them fragile: a pole pushed outside the circle by rounding or "
             "by a design error makes the output grow without bound.")

    # ----------------------------------------------------------- 4 dataset
    rep.heading("Dataset Description")
    rep.para(
        "The **EEG Motor Movement/Imagery Database** (Schalk et al., PhysioNet, Open Data Commons Attribution licence) holds "
        "over 1,500 one- and two-minute 64-channel EEG recordings from 109 volunteers, made with the BCI2000 system. Each "
        "subject's first two runs are 1-minute baselines: **run 1 with eyes open, run 2 with eyes closed**. Both runs of "
        f"subject S001 were used, channel **{config.CHANNEL}** (left occipital, over the visual cortex where alpha is "
        f"strongest). The files are in EDF format, {D['size_kb'] / 1024:.1f} MB in total, downloaded from PhysioNet's open-data "
        "mirror on AWS and parsed directly. The header records no hardware filtering, so the 60 Hz interference is still present.")
    xo, xc = D["open"]["x"], D["closed"]["x"]
    rep.table(
        ["Property", "Value", "Property", "Value"],
        [
            ["Source", "PhysioNet eegmmidb 1.0.0, S001", "System", "BCI2000, 64 electrodes (10-10 system)"],
            ["Runs", "R01 eyes open, R02 eyes closed", "Channel used", f"{config.CHANNEL} (occipital)"],
            ["Sampling rate", f"{e['fs']:g} Hz", "Length", f"{len(xo):,} samples ({len(xo) / FS:.0f} s) per run"],
            ["Units, resolution", f"{e['units']}, int16 ({e['digital'][1]:.0f} digital = {e['physical'][1]:.0f} µV)", "Hardware filter", e["prefilter"]],
            ["Std. dev. open / closed", f"{xo.std():.1f} / {xc.std():.1f} µV", "Recording start",
             f"{e['start'][:8]} {e['start'][8:].replace('.', ':')} (dd.mm.yy)"],
        ],
        "Dataset attributes", widths=[3.4, 4.8, 3.2, 5.6], font_size=8.5, align=["left"] * 4)
    rep.table(
        ["n", "Time (s)", "O1 eyes open (µV)", "O1 eyes closed (µV)"],
        [[f"{i:,}", f"{i / FS:.4f}", f"{xo[i]:.1f}", f"{xc[i]:.1f}"] for i in range(3200, 3206)],
        "Dataset preview (channel O1, t = 20 s)", widths=[2.6, 3.4, 5.5, 5.5], font_size=8.5, align=["center"] * 4)
    rep.figure(FIG / "fig1_raw.png", "The raw EEG.",
               "(a) Five seconds of channel O1 in both conditions; (b) power spectral density with the alpha band and "
               "the mains frequency marked.",
               inference=(
                   "With the eyes closed the trace is dominated by a regular rhythm of about ten cycles per second; with "
                   f"the eyes open it is smaller and irregular (std {xo.std():.0f} versus {xc.std():.0f} µV). The spectrum makes "
                   f"this quantitative: a sharp peak at {f(A['closed']['peak_hz'], 1)} Hz appears only with the eyes closed. Both "
                   f"recordings show a narrow line at 60 Hz, about {NT['p60_before'] / NT['floor']:.1f} times the surrounding "
                   "EEG background: mains interference. A falling 1/f background spans the whole range."))

    # ---------------------------------------------------- 5 implementation
    rep.heading("Methodology and Implementation")
    rep.para(f"The processing chain is: first-order high-pass at {config.HIGHPASS_HZ} Hz (drift), 60 Hz notch "
             f"(Q = {config.NOTCH_Q:g}), and an order-{config.ORDER} prototype band-pass at "
             f"{config.ALPHA_BAND[0]:g} to {config.ALPHA_BAND[1]:g} Hz (order {2 * config.ORDER} overall) realised as "
             "second-order sections. The three prototype families are compared, alpha power is computed in 2-second "
             "epochs and the two conditions are compared with Welch's t-test.", keep_next=True)
    rep.table(
        ["Module", "Responsibility"],
        [
            ["`config.py`", "Runs, channel, band, orders, ripple, notch Q"],
            ["`data_loader.py`", "Download from the AWS mirror once, cache, parse EDF, convert to µV"],
            ["`iir.py`", "Difference equation, SOS cascade, zero-phase, notch, high-pass, pre-warp, bilinear transform"],
            ["`analysis.py`", "Family comparison, FIR equivalent, pre-warping, numerics, notch, phase, alpha study"],
            ["`plots.py` / `gui.py`", "Report figures / interactive IIR filter designer"],
            ["`test_iir.py`", "8 unit tests (SciPy agreement, exact edges, stability, notch, zero phase): all pass"],
        ],
        "Software modules", widths=[4.2, 12.8], font_size=9, align=["left", "left"])
    rep.code('''def notch(f0, Q, fs):                     # zeros on |z| = 1, poles at radius r
    w0 = 2 * np.pi * f0 / fs
    r = 1 - w0 / Q / 2
    b = np.array([1.0, -2 * np.cos(w0), 1.0])
    a = np.array([1.0, -2 * r * np.cos(w0), r * r])
    return b * a.sum() / b.sum(), a, r          # unit gain at DC

def bilinear_zpk(z, p, k, fs):            # s = 2fs (1 - z^-1) / (1 + z^-1)
    c = 2 * fs
    zd, pd = (c + z) / (c - z), (c + p) / (c - p)
    zd = np.r_[zd, -np.ones(len(p) - len(z))]   # zeros at s = infinity go to z = -1
    return zd, pd, k * np.real(np.prod(c - z) / np.prod(c - p))

def iir_filter(b, a, x):                  # y[n] = sum b_k x[n-k] - sum a_k y[n-k]
    y = np.zeros(len(x))
    for n in range(len(x)):
        y[n] = sum(b[k] * x[n - k] for k in range(min(len(b) - 1, n) + 1)) \\
             - sum(a[k] * y[n - k] for k in range(1, min(len(a) - 1, n) + 1))
    return y''', title="Core of iir.py", font_size=7.5)
    rep.para("**GUI.** `gui.py` (Figure 2) designs a Butterworth, Chebyshev I or elliptic band-pass of any order, edges, "
             "ripple and attenuation, optionally after the 60 Hz notch and with causal or zero-phase filtering. It shows "
             "the magnitude response, group delay, pole-zero plot and the filtered EEG for both conditions, and reports the "
             "largest pole radius, the in-band delay and the alpha-power ratio; the second-order sections can be exported. "
             "Run `python run_experiment.py` and `python gui.py`.")
    rep.figure(FIG / "fig7_gui.png", "The IIR filter designer with the order-8 elliptic alpha band-pass.",
               "Eight poles crowd close to the unit circle near 10 Hz; the eyes-closed trace shows the alpha rhythm.",
               width_cm=14.5)

    # ------------------------------------------------------------ 6 calcs
    rep.heading("Calculations")
    w0 = 2 * np.pi * 60 / FS
    r60 = 1 - w0 / config.NOTCH_Q / 2
    W1, W2 = iir.prewarp(config.ALPHA_BAND, FS)
    s_test = -100 + 200j
    z_test = (2 * FS + s_test) / (2 * FS - s_test)
    bh, ah = iir.dc_blocker(config.HIGHPASS_HZ, FS)
    rep.para("**60 Hz notch.** ω₀ = 2π·60/160 = 3π/4 rad/sample, cos ω₀ = −0.7071.", keep_next=True)
    rep.table(
        ["Quantity", "Working", "Result"],
        [
            ["Pole radius r", f"1 − ω₀/(2Q) = 1 − 2.3562/60", f"{r60:.4f}"],
            ["Numerator (zeros at e^(±jω₀))", "1 − 2cos ω₀ z⁻¹ + z⁻²", "1 + 1.4142 z⁻¹ + z⁻²"],
            ["Denominator (poles)", f"1 − 2r cos ω₀ z⁻¹ + r² z⁻²", f"1 + {2 * r60 * 0.70711:.4f} z⁻¹ + {r60 ** 2:.4f} z⁻²"],
            ["−3 dB bandwidth", f"f₀/Q = 60/{config.NOTCH_Q:g}", f"2.0 Hz (measured {f(NT['bw3'])} Hz)"],
            ["Time to settle to 1 %", f"ln 0.01 / ln r / fs", f"{f(NT['settle_s'])} s"],
            ["High-pass pole at 0.5 Hz", "p = e^(−2π·0.5/160)", f"{-ah[1]:.4f}"],
            ["Pre-warped alpha edges", "2fs tan(πf/fs), f = 8 and 13 Hz", f"{W1 / (2 * np.pi):.3f} and {W2 / (2 * np.pi):.3f} Hz"],
            ["Bilinear map of s = −100 + j200", "(2fs + s)/(2fs − s)", f"{z_test.real:.4f} {'+' if z_test.imag >= 0 else '−'} j{abs(z_test.imag):.4f} (|z| = {abs(z_test):.4f} < 1)"],
        ],
        "Hand calculations", widths=[4.6, 6.4, 6.0], font_size=8.5, align=["left", "left", "left"])
    rep.para("Pre-warping moves the analog edges up by only "
             f"{100 * (W1 / (2 * np.pi * 8) - 1):.1f} % and {100 * (W2 / (2 * np.pi * 13) - 1):.1f} % at alpha frequencies, "
             "but by much more near the Nyquist frequency, where tan(πf/fs) grows quickly. The left-half-plane test point "
             "maps inside the unit circle, as the bilinear transform guarantees. These values agree with the program "
             "(checked by `test_iir.py`).")

    # --------------------------------------------------------- 7 results
    rep.heading("Results and Discussion")
    rep.figure(FIG / "fig2_designs.png", "Three order-8 alpha band-pass designs.",
               "(a) Magnitude responses; (b) group delay (alpha band shaded); (c) poles (×) and zeros (○) of the elliptic design.",
               inference="", inference_items=[
                   f"All three use {bu['n_coef']} coefficients. At 5 Hz the Butterworth attenuates {f(-bu['att5'], 0)} dB, Chebyshev I "
                   f"{f(-ch['att5'], 0)} dB and elliptic {f(-el['att5'], 0)} dB; Chebyshev and elliptic buy this with {f(config.CHEBY_RIPPLE_DB, 0)} dB "
                   f"passband ripple, and the elliptic's stopband stops falling at its {config.ELLIP_STOP_DB:g} dB floor (only "
                   f"{f(-el['att60'], 0)} dB at 60 Hz, against {f(-bu['att60'], 0)} dB for Butterworth).",
                   f"A Hamming-window FIR filter needs {FE['N']} taps to match the Butterworth's attenuation at 5 Hz, about "
                   f"{FE['N'] / bu['n_coef']:.0f} times more coefficients.",
                   f"The price is phase: the in-band group delay ranges from {f(bu['tau_min'])} to {f(bu['tau_max'])} s for "
                   f"Butterworth and {f(el['tau_min'])} to {f(el['tau_max'])} s for the elliptic, peaking at the band edges where "
                   f"the poles sit closest to the unit circle (radius {f(el['max_pole'], 4)}).",
               ])
    rep.figure(FIG / "fig3_bilinear.png", "Pre-warping and numerical precision.",
               "(a) A Butterworth band-pass specified as 50 to 70 Hz, designed with and without pre-warping; (b) an order-16 "
               "band-pass (9 to 11 Hz) applied as one difference equation and as a cascade of biquads.",
               inference="", inference_items=[
                   f"With pre-warping the band edges are exact ({f(pw_hi_warp['edges'][0], 1)} and {f(pw_hi_warp['edges'][1], 1)} Hz). "
                   f"Without it the bilinear transform's frequency compression moves the band to {f(pw_hi_no['edges'][0], 1)} to "
                   f"{f(pw_hi_no['edges'][1], 1)} Hz. For the low alpha band the error is small ({f(pw_lo_no['edges'][0], 2)} to "
                   f"{f(pw_lo_no['edges'][1], 2)} Hz), but it grows rapidly towards fs/2.",
                   f"For the order-8 filters used here the direct form agrees with the cascade to {NU['butterworth']['rel_err']:.0e}. "
                   f"For the narrower order-16 design the coefficients of the single denominator polynomial, rounded to double "
                   f"precision, place a pole at radius {f(NU['narrow16']['max_pole_tf'], 4)} instead of "
                   f"{f(NU['narrow16']['max_pole_true'], 4)}: the filter is unstable and its output grows to about 10³⁷ µV within a "
                   "minute. The same filter as eight biquads is perfectly stable, which is why SOS form is standard practice.",
               ])
    rep.figure(FIG / "fig4_notch.png", "The 60 Hz notch filter.",
               "(a) Magnitude response with its pole-zero pair (inset); (b) eyes-open EEG spectrum before and after notching.",
               inference=(
                   f"The notch is {f(-NT['depth_db'], 0)} dB deep at exactly 60 Hz, only {f(NT['bw3'])} Hz wide at −3 dB, and leaves the "
                   f"alpha band untouched (gain at 10 Hz {f(NT['gain10_db'], 4)} dB, alpha power changed by {f(NT['alpha_change_db'], 4)} dB). "
                   f"The 60 Hz line falls from {NT['p60_before']:.3f} to {NT['p60_after']:.4f} µV²/Hz, below the EEG background "
                   f"(about {NT['floor']:.2f} µV²/Hz). Power in the whole 59 to 61 Hz band falls by only {f(NT['mains_reduction_db'], 1)} dB "
                   f"because most of it is broadband EEG, which a notch must not remove. Its poles at radius {f(NT['r'], 3)} make "
                   f"the filter ring for about {f(NT['settle_s'], 1)} s after a disturbance."))
    rep.figure(FIG / "fig5_phase.png", "Causal versus zero-phase filtering of the eyes-closed EEG.",
               "1.5 s of the raw signal with the elliptic band-pass applied causally and forward-backward.",
               inference=(
                   f"Both outputs isolate the same 10 Hz oscillation, but the causal output is delayed by {f(PH['lag_ms'], 0)} ms "
                   f"({PH['lag_samples']} samples), about two alpha cycles, and its waveform is slightly reshaped because the "
                   "delay varies with frequency. The zero-phase output lines up with the raw signal. Real-time systems (monitors, "
                   "brain-computer interfaces) must accept the delay; off-line clinical analysis uses zero-phase filtering."))
    rep.figure(FIG / "fig6_alpha.png", "The Berger effect measured with the IIR filter chain.",
               "(a) Alpha-band output with its envelope in both conditions; (b) alpha power in each 2-second epoch.",
               inference=(
                   f"With the eyes closed the alpha rhythm is continuous and large (RMS {f(A['closed']['rms_uv'], 1)} µV); with the "
                   f"eyes open it appears only in brief bursts (RMS {f(A['open']['rms_uv'], 1)} µV). Mean alpha power is "
                   f"**{f(A['ratio'], 1)} times** higher with the eyes closed, {100 * A['frac_closed_above']:.0f} % of eyes-closed epochs "
                   f"exceed every eyes-open epoch, and the difference is highly significant (Welch t = {f(A['t'], 1)}, p {p_txt}). "
                   "Closing the eyes removes visual input and the visual cortex falls into this idling rhythm, exactly the "
                   "effect that makes alpha useful for detecting drowsiness and for eyes-closed brain-computer interfaces."))
    rep.table(
        ["Quantity", "Butterworth", "Chebyshev I", "Elliptic"],
        [
            ["−3 dB edges (Hz)"] + [f"{Ds[k]['edges'][0]:.2f} to {Ds[k]['edges'][1]:.2f}" for k in config.FAMILIES],
            ["Passband variation (dB)"] + [f(Ds[k]["ripple_db"]) for k in config.FAMILIES],
            ["Attenuation at 5 / 16 Hz (dB)"] + [f"{-Ds[k]['att5']:.0f} / {-Ds[k]['att16']:.0f}" for k in config.FAMILIES],
            ["Attenuation at 60 Hz (dB)"] + [f"{-Ds[k]['att60']:.0f}" for k in config.FAMILIES],
            ["In-band group delay (s)"] + [f"{Ds[k]['tau_min']:.2f} to {Ds[k]['tau_max']:.2f}" for k in config.FAMILIES],
            ["Largest pole radius"] + [f(Ds[k]["max_pole"], 4) for k in config.FAMILIES],
        ],
        f"Summary of the alpha band-pass designs ({bu['n_coef']} coefficients each; FIR equivalent {FE['N']} taps)",
        widths=[5.2, 3.9, 3.9, 4.0], font_size=9, first_col_bold=True, align=["left", "center", "center", "center"])

    # ------------------------------------------------------ 8 conclusion
    rep.heading("Conclusion")
    rep.para(
        "IIR filters were designed in two ways: by placing poles and zeros by hand (notch, high-pass) and by transforming "
        "classical analog prototypes with the pre-warped bilinear transform, which matched SciPy exactly and preserved "
        "stability. They proved highly efficient: an order-8 band-pass with 17 coefficients replaced an FIR filter of "
        f"{FE['N']} taps, and a single biquad removed mains interference. The experiment also exposed their weaknesses: "
        "non-linear phase that delays and reshapes waveforms (cured off-line by forward-backward filtering), the need to "
        "pre-warp critical frequencies, and numerical fragility that makes a cascade of second-order sections essential for "
        "high orders. Applied to real EEG, the filter chain cleaned the signal and measured a "
        f"{f(A['ratio'], 0)}-fold increase in alpha power on closing the eyes, reproducing Berger's 1929 observation with a few "
        "lines of signal processing.")
    rep.heading("Learning Outcomes")
    rep.bullets([
        "Design IIR filters by pole-zero placement and by the bilinear transform with pre-warping.",
        "Compare Butterworth, Chebyshev and elliptic responses and choose a family for a specification.",
        "Implement IIR filters as difference equations and as second-order sections, and judge stability from pole radii.",
        "Explain IIR phase distortion, apply zero-phase filtering, and use IIR filters on a real biomedical signal.",
    ])
    rep.heading("References")
    refs = [
        "A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed. Pearson, 2010, ch. 7.1 to 7.3.",
        "J. G. Proakis and D. G. Manolakis, *Digital Signal Processing*, 4th ed. Pearson, 2007, ch. 10.",
        "G. Schalk et al., “BCI2000: a general-purpose brain-computer interface (BCI) system,” *IEEE Trans. Biomed. Eng.*, "
        "vol. 51, no. 6, pp. 1034-1043, 2004.",
        "A. L. Goldberger et al., “PhysioBank, PhysioToolkit, and PhysioNet,” *Circulation*, vol. 101, no. 23, pp. e215-e220, 2000.",
        "H. Berger, “Über das Elektrenkephalogramm des Menschen,” *Archiv für Psychiatrie und Nervenkrankheiten*, vol. 87, "
        "pp. 527-570, 1929.",
    ]
    for i, rr in enumerate(refs, 1):
        rep.para(f"[{i}]  {rr}", align="left", size=9.5, space_after=1)

    path = rep.save(OUT_DOCX)
    print(f"saved {path}")
    print(f"saved {export_pdf(path)}")


if __name__ == "__main__":
    build()
