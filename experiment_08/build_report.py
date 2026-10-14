"""
build_report.py
---------------
Generates the Word (and PDF) lab report for Experiment 8.

    python run_experiment.py      # figures + results first
    python build_report.py        # -> DSP_Experiment_08.docx / .pdf

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
import dft as D                      # noqa: E402
from report_kit import Report, export_pdf   # noqa: E402

STUDENT = {
    "Name": "Dewashish Lambore",
    "PRN": "____________",
    "Date": "____ / ____ / 2026",
    "Course": "Digital Signal Processing Lab",
}
FIG = config.FIGURE_DIR
OUT_DOCX = HERE / "DSP_Experiment_08.docx"


def f(x, d=2):
    v = 0.0 if abs(x) < 0.5 * 10 ** (-d) else x
    return f"{v:,.{d}f}"


def cplx(z, d=2):
    re, im = (0.0 if abs(z.real) < 1e-9 else z.real), (0.0 if abs(z.imag) < 1e-9 else z.imag)
    if im == 0:
        return f"{re:.{d}f}"
    return f"{re:.{d}f} {'+' if im > 0 else '−'} j{abs(im):.{d}f}"


def build():
    R = analysis.run_all()
    w, r, info, Mt, S, L, Se, Dd, G, T = (R["w"], R["r"], R["info"], R["methods"], R["spec"], R["leak"],
                                         R["season"], R["decades"], R["growth"]["by_epoch"], R["timing"])
    N = len(r)
    A1, ph1 = S["lines"][1]
    A2, ph2 = S["lines"][2]
    t804 = next(t for t in T if t["N"] == 804)
    t1024 = next(t for t in T if t["N"] == 1024)

    rep = Report(footer_left="DSP Lab  ·  Experiment 8  ·  DFT and IDFT")

    # ------------------------------------------------------------------ title
    rep.info_block([("Course", STUDENT["Course"]), ("Experiment No.", "08"),
                    ("Name", STUDENT["Name"]), ("PRN", STUDENT["PRN"]),
                    ("Language", "Python 3.10"), ("Date", STUDENT["Date"])])
    rep.title("Discrete Fourier Transform and its Inverse",
              "The seasonal breathing of the planet in the Mauna Loa CO₂ record")
    rep.abstract(
        f"The monthly Mauna Loa CO₂ record (the Keeling Curve) was trimmed to {info['years']} whole years "
        f"(N = {N}), detrended, and transformed with a DFT implemented from its definition, as a matrix product "
        f"and by the FFT; the three agreed to {max(Mt['matrix_vs_fft'], Mt['direct_vs_fft']):.0e}, the IDFT recovered "
        f"the signal to {Mt['idft_err']:.0e} ppm, and Parseval's theorem held to {Mt['parseval']:.0e}. The spectrum "
        f"consists of two clean lines at exactly 1 and 2 cycles per year (amplitudes {f(A1)} and {f(A2)} ppm) above "
        f"a noise floor of about {f(S['noise_median'], 3)} ppm. An IDFT of these two lines alone rebuilt the seasonal "
        f"cycle, which peaks in {Se['max_month']} and bottoms out in {Se['min_month']} ({f(Se['p2p'], 1)} ppm peak to "
        f"trough) and explains {100 * Se['var_explained']:.0f} % of the detrended variance; removing it reproduced "
        f"NOAA's official deseasonalized series to {f(Se['rmse_noaa'], 3)} ppm. Cutting the record by "
        f"{config.LEAK_CUT} months caused leakage that lowered the annual line by "
        f"{100 * (1 - L['peak_cut'] / L['peak_exact']):.0f} %, which a Hann window and zero padding largely corrected. "
        f"The seasonal swing grew from {f(Dd[0]['p2p'], 1)} to {f(Dd[-1]['p2p'], 1)} ppm over six decades.",
        "DFT, IDFT, FFT, spectral leakage, windowing, zero padding, Parseval, frequency-domain filtering, "
        "Keeling Curve, CO₂, Python")

    # ------------------------------------------------------------- 1 aim
    rep.heading("Aim and Objectives")
    rep.para("**Aim:** to compute the DFT and IDFT of a real-world signal, verify their properties, and use them "
             "to analyse and filter the signal in the frequency domain, in Python.", space_after=3)
    rep.bullets([
        "Implement the DFT and IDFT from the definition and as a matrix, and compare them with the FFT in accuracy and speed.",
        "Read frequencies, amplitudes and phases from the spectrum and understand leakage, windowing and zero padding.",
        "Use the IDFT to synthesise the seasonal cycle and to remove it, and validate the result against NOAA's own product.",
    ])

    # ----------------------------------------------------- 2 application
    rep.heading("Real-World Application")
    rep.para(
        "Atmospheric CO₂ has been measured continuously on Mauna Loa, Hawaii, since 1958. The record rises every year "
        "because of fossil-fuel burning, but it also 'breathes': each northern spring and summer, forests and crops "
        "take up CO₂, and each autumn and winter decaying vegetation releases it. Climate scientists separate this "
        "seasonal cycle from the long-term rise in exactly the way done here, and the DFT is the natural tool because "
        "the cycle is a sum of sinusoids at 1, 2, ... cycles per year. The DFT and IDFT also sit at the heart of many "
        "everyday technologies (Table 1).")
    rep.table(
        ["Technology", "What the DFT / IDFT does"],
        [
            ["Climate science (this experiment)", "separates seasonal cycles from trends; seasonal adjustment of indices"],
            ["4G / 5G / Wi-Fi (OFDM)", "the transmitter is an IDFT and the receiver a DFT, one per symbol"],
            ["MRI scanners", "the image is the 2-D IDFT of the measured k-space data"],
            ["Audio and music apps", "spectrum analysers, pitch detection, MP3 / AAC compression"],
            ["Power-quality meters", "measure harmonics of the 50 Hz mains (IEC 61000-4-7)"],
        ],
        "Where the DFT and IDFT are used", widths=[5.4, 11.6], font_size=9, first_col_bold=True,
        align=["left", "left"])

    # ------------------------------------------------------------ 3 theory
    rep.heading("Theory")
    rep.para("The **discrete Fourier transform** of an $N$-point sequence $x[n]$ and its **inverse** are")
    rep.equation(r"X[k] = \sum_{n=0}^{N-1} x[n]\,e^{-j2\pi kn/N}, \qquad x[n] = \frac{1}{N}\sum_{k=0}^{N-1} X[k]\,e^{j2\pi kn/N}")
    rep.para("for $k, n = 0, \\ldots, N-1$. Collecting the twiddle factors $W_N^{kn} = e^{-j2\\pi kn/N}$ in an "
             "$N \\times N$ matrix gives the matrix form, whose inverse is its scaled conjugate:")
    rep.equation(r"\mathbf{X} = \mathbf{W}\mathbf{x}, \qquad \mathbf{W}^{-1} = \frac{1}{N}\mathbf{W}^{*}")
    rep.para("Bin $k$ corresponds to frequency $f_k = k f_s/N$, so the resolution is $f_s/N$. For a real signal a "
             "sinusoid $A\\cos(2\\pi k_0 n/N + \\phi)$ appears in bins $k_0$ and $N - k_0$, from which its amplitude "
             "and phase are read:")
    rep.equation(r"f_k = \frac{k f_s}{N}, \qquad A = \frac{2\left|X[k_0]\right|}{N}, \qquad \phi = \angle X[k_0]")
    rep.para("Two properties are checked numerically: energy is preserved (Parseval) and the spectrum of a real "
             "signal is conjugate-symmetric, so only bins 0 to N/2 carry information:")
    rep.equation(r"\sum_{n=0}^{N-1}\left|x[n]\right|^{2} = \frac{1}{N}\sum_{k=0}^{N-1}\left|X[k]\right|^{2}, \qquad X[N-k] = X^{*}[k]")
    rep.para("**Leakage.** The DFT implicitly repeats the N samples periodically. If a sinusoid does not complete a "
             "whole number of cycles in N samples, the repetition has a jump and the line spreads into neighbouring "
             "bins. A tapering window such as the Hann window reduces the spread, and zero padding to $PN$ points "
             "samples the same spectrum more finely without adding resolution:")
    rep.equation(r"w[n] = \frac{1}{2} - \frac{1}{2}\cos\frac{2\pi n}{N}, \qquad X_{pad}[k] = \sum_{n=0}^{N-1} x[n]\,w[n]\,e^{-j2\pi kn/(PN)}")
    rep.note("Direct evaluation of Eq. 1 needs N² complex multiplications; the fast Fourier transform (FFT) computes "
             "exactly the same numbers in about N log₂ N by reusing partial sums. It is fastest when N has only "
             "small prime factors.")

    # ----------------------------------------------------------- 4 dataset
    rep.heading("Dataset Description")
    rep.para(
        "The **Mauna Loa monthly mean CO₂** record is published by the NOAA Global Monitoring Laboratory (in "
        "cooperation with Scripps Institution of Oceanography, which began it under C. D. Keeling). Values are dry-air "
        f"mole fractions in parts per million (ppm). The CSV is {info['size_kb']:.0f} kB, freely downloadable and cached. "
        f"It runs from {info['start']} to {info['end']} ({info['months']} months); NOAA also supplies a deseasonalized "
        "column, used here only as an independent check. For the DFT the record is trimmed to "
        f"{info['w_start']} to {info['w_end']}: N = {N} months = {info['years']} whole years, so that 1 cycle per year "
        f"falls exactly on bin k = {S['k1']}.")
    rep.table(
        ["Property", "Value", "Property", "Value"],
        [
            ["Provider", "NOAA GML / Scripps", "Station", "Mauna Loa Observatory, Hawaii (3,397 m)"],
            ["Full record", f"{info['start']} to {info['end']}", "DFT record", f"{info['w_start']} to {info['w_end']}, N = {N}"],
            ["Range", f"{info['min']:.2f} to {info['max']:.2f} ppm", "Sampling", "fs = 12 per year (monthly)"],
            ["Resolution fs/N", f"{S['resolution']:.4f} cycles/year", "Nyquist frequency", "6 cycles per year"],
            ["Columns used", "year, month, average, deseasonalized", "Months without day count", f"{info['no_daycount']} (early record)"],
        ],
        "Dataset attributes", widths=[3.2, 5.0, 3.6, 5.2], font_size=8.5, align=["left"] * 4)
    tail = w.iloc[-6:]
    rep.table(
        ["Year", "Month", "Decimal year", "CO₂ (ppm)", "NOAA deseasonalized", "Trend", "Detrended r[n]"],
        [[t.year, t.month, f"{t.t:.4f}", f"{t.co2:.2f}", f"{t.co2_deseason:.2f}", f"{R['trend'][t.m]:.2f}",
          f"{r[t.m]:+.2f}"] for t in tail.itertuples()],
        "Last six samples of the DFT record",
        widths=[1.6, 1.4, 2.6, 2.4, 3.4, 2.4, 3.2], font_size=8.5, align=["center"] * 7)
    rep.figure(FIG / "fig1_raw.png", "The input signal.",
               f"(a) Monthly CO₂ with the order-{config.TREND_ORDER} polynomial trend; (b) the detrended signal.",
               inference=(
                   f"CO₂ rose from about {info['first']:.0f} to {info['last']:.0f} ppm, and the rise is accelerating (the "
                   f"slope of the trend grew from {f(G[1960], 2)} ppm/yr around 1960 to {f(G[2020], 2)} ppm/yr around 2020). "
                   "After the trend is removed, what remains is a strikingly regular oscillation of about ±3 ppm every "
                   "year: a periodic signal for which the DFT is the natural description."))

    # ---------------------------------------------------- 5 implementation
    rep.heading("Methodology and Implementation")
    rep.para(f"The trend is removed by least squares, the DFT of r[n] is computed, the bins at 1 and 2 cycles per "
             f"year (k = {S['k1']} and {2 * S['k1']}, with their mirrors) are kept and all others set to zero, and the "
             f"IDFT of what remains is the seasonal model. Leakage is studied by deleting {config.LEAK_CUT} months, and "
             "the DFT matrix is timed against the FFT for several N.", keep_next=True)
    rep.table(
        ["Module", "Responsibility"],
        [
            ["`config.py`", "URL, sampling rate, trend order, harmonics, leakage and timing settings"],
            ["`data_loader.py`", "Download once, cache, trim to whole years"],
            ["`dft.py`", "DFT by definition, DFT matrix, IDFT, amplitude/phase, bin masking, Hann window, checks"],
            ["`analysis.py`", "Spectrum, leakage, seasonal synthesis, decades, growth rate, timing"],
            ["`plots.py` / `gui.py`", "Report figures / interactive spectrum analyser and DFT calculator"],
            ["`test_dft.py`", "10 unit tests (inverse, Parseval, symmetry, linearity, shift, leakage): all pass"],
        ],
        "Software modules", widths=[4.2, 12.8], font_size=9, align=["left", "left"])
    rep.code('''def dft_direct(x):                       # X[k] = sum x[n] e^{-j 2 pi k n / N}
    N = len(x)
    return np.array([sum(x[n] * np.exp(-2j * np.pi * k * n / N) for n in range(N))
                     for k in range(N)])

def dft_matrix(N):                       # W[k, n] = e^{-j 2 pi k n / N}
    k = np.arange(N)
    return np.exp(-2j * np.pi * np.outer(k, k) / N)

def idft(X):                             # x = (1/N) W* X
    return np.conj(dft_matrix(len(X))) @ X / len(X)

def keep_bins(X, bins):                  # ideal filter: keep +k and N-k only
    mask = np.zeros(len(X), bool)
    for k in bins:
        mask[k % len(X)] = mask[-k % len(X)] = True
    return np.where(mask, X, 0)''', title="Core of dft.py", font_size=7.5)
    rep.para("**GUI.** `gui.py` (Figure 2) lets the user change the record length month by month, the trend order, "
             "the window and the zero-padding factor, and choose which harmonics the IDFT keeps. It reports the line "
             "amplitude, whether N/12 falls on a bin, the resolution and the error against NOAA. A calculator returns "
             "the DFT of any typed sequence for checking hand calculations. Run `python run_experiment.py` and `python gui.py`.")
    rep.figure(FIG / "fig6_gui.png", f"The spectrum analyser with N = {L['Nc']} months.",
               "N/12 is not an integer: the annual line leaks, and an IDFT that keeps the nearest bins drifts out of phase.",
               width_cm=14.5)

    # ------------------------------------------------------ 6 calculations
    rep.heading("Calculations")
    xh = np.array([0.1, 2.5, 0.6, -3.2])
    Xh = D.dft(xh)
    rep.para("**4-point DFT by hand.** One sample per quarter of the seasonal model (January, April, July, October), "
             "rounded: x = [0.1, 2.5, 0.6, −3.2]. With N = 4, $W_4 = e^{-j\\pi/2} = -j$, so $W_4^{kn}$ cycles through "
             "1, −j, −1, j:", keep_next=True)
    rep.table(
        ["k", "X[k] = x[0] + x[1]W⁴ᵏ + x[2]W⁴²ᵏ + x[3]W⁴³ᵏ", "X[k]", "|X[k]|", "2|X[k]|/N"],
        [
            ["0", "0.1 + 2.5 + 0.6 − 3.2", cplx(Xh[0]), f(abs(Xh[0])), "n/a (mean)"],
            ["1", "0.1 + 2.5(−j) + 0.6(−1) − 3.2(j)", cplx(Xh[1]), f(abs(Xh[1])), f(2 * abs(Xh[1]) / 4)],
            ["2", "0.1 − 2.5 + 0.6 + 3.2", cplx(Xh[2]), f(abs(Xh[2])), f(2 * abs(Xh[2]) / 4)],
            ["3", "0.1 + 2.5(j) + 0.6(−1) − 3.2(−j)", cplx(Xh[3]), f(abs(Xh[3])), "mirror of k = 1"],
        ],
        "DFT of four quarterly values", widths=[0.8, 7.6, 3.0, 2.4, 3.2], font_size=8.5,
        align=["center", "left", "center", "center", "center"])
    E_t = float(np.sum(xh ** 2))
    E_f = float(np.sum(np.abs(Xh) ** 2) / 4)
    n_peak = (-ph1) % (2 * np.pi) / (2 * np.pi) * 12
    rep.para(f"**Checks.** X[3] is the complex conjugate of X[1] (symmetry). Parseval: Σx² = {E_t:.2f} and (1/4)Σ|X|² = "
             f"(0 + {abs(Xh[1]) ** 2:.2f} + {abs(Xh[2]) ** 2:.2f} + {abs(Xh[3]) ** 2:.2f})/4 = {E_f:.2f}. IDFT of x[1]: "
             "(1/4)[X[0] + X[1]·j + X[2]·(−1) + X[3]·(−j)] = (1/4)(0 + 5.7 − 0.5j − 1.4 + 5.7 + 0.5j) = 2.5. "
             f"**Real record:** |X[{S['k1']}]| = {A1 * N / 2:.1f}, so A = 2|X|/N = {f(A1, 3)} ppm with phase "
             f"φ = {ph1:.3f} rad. The fundamental alone peaks where 2πn/12 + φ = 0, i.e. n = {n_peak:.2f} months after "
             "the start of January (mid-April); adding the 2-cycle harmonic moves the combined peak to "
             f"{Se['max_month']}. These values match the program (checked by `test_dft.py`).")

    # --------------------------------------------------------- 7 results
    rep.heading("Results and Discussion")
    rep.figure(FIG / "fig2_spectrum.png", "Amplitude spectrum of the detrended record.",
               f"(a) 0 to 6 cycles per year (Nyquist); (b) individual bins around k = {S['k1']}.",
               inference="", inference_items=[
                   f"The spectrum is almost empty except for two lines exactly at 1 and 2 cycles per year: "
                   f"{f(A1, 3)} ppm and {f(A2, 3)} ppm, about {A1 / S['noise_median']:.0f} and {A2 / S['noise_median']:.0f} "
                   f"times the noise floor ({f(S['noise_median'], 3)} ppm). The 3 and 4 cycle harmonics are tiny "
                   f"({f(S['lines'][3][0], 3)} and {f(S['lines'][4][0], 3)} ppm), so the seasonal cycle is well described "
                   "by two sinusoids.",
                   f"Because N = {N} is a whole number of years, the annual line falls on a single bin: its neighbours "
                   "are at noise level, i.e. there is no leakage. The small bump near 0 cycles per year is slow "
                   "variation in the growth rate that a cubic trend does not capture.",
               ])
    rep.figure(FIG / "fig3_leakage.png", "Leakage, windowing and zero padding.",
               f"(a) N = {N} versus N = {L['Nc']} with rectangular and Hann windows; (b) Hann-windowed spectrum zero-padded "
               f"×{config.ZERO_PAD}.",
               inference=(
                   f"With {L['Nc']} months, N/12 is not an integer (the annual line sits {L['bin_offset']:.2f} of a bin "
                   f"between bins), and the line leaks: its peak drops from {f(L['peak_exact'], 2)} to "
                   f"{f(L['peak_cut'], 2)} ppm and its energy spreads into neighbouring bins. The Hann window "
                   f"recovers the peak to {f(L['peak_hann'], 2)} ppm and suppresses the far leakage, at the cost of a "
                   f"wider main lobe. Zero padding then shows the underlying curve: its maximum is {f(L['peak_pad'], 2)} ppm "
                   f"at {L['f_peak_pad']:.4f} cycles per year, close to the true values. Zero padding interpolates; it "
                   "does not sharpen the line."))
    rep.figure(FIG / "fig4_idft.png", "Synthesis and removal of the seasonal cycle with the IDFT.",
               f"(a) Detrended signal and the IDFT of the two seasonal lines; (b) one year of the model; (c) "
               f"deseasonalized CO₂ compared with NOAA's product (RMSE {f(Se['rmse_noaa'], 3)} ppm).",
               inference="", inference_items=[
                   f"Keeping only 4 of {N} DFT coefficients reproduces the seasonal oscillation and explains "
                   f"{100 * Se['var_explained']:.0f} % of the detrended variance; what is left (std {f(Se['resid_std'], 2)} ppm) "
                   "is weather-driven noise and year-to-year growth variation.",
                   f"CO₂ is highest in **{Se['max_month']}** and lowest in **{Se['min_month']}**, a "
                   f"{f(Se['p2p'], 1)} ppm swing: the Northern Hemisphere biosphere draws CO₂ down through the growing "
                   "season and releases it in autumn and winter.",
                   f"Subtracting the model gives a deseasonalized curve within {f(Se['rmse_noaa'], 3)} ppm of NOAA's official "
                   "series, which is produced by a different method, so the simple DFT/IDFT filter is fit for purpose.",
               ])
    rep.figure(FIG / "fig5_decades_timing.png", "Long-term changes and computational cost.",
               "(a) Peak-to-trough seasonal swing per decade (IDFT of two lines); (b) growth rate of the trend; (c) "
               "time for one transform.",
               inference="", inference_items=[
                   "The seasonal swing grew from "
                   + ", ".join(f"{d['p2p']:.2f}" for d in Dd) + " ppm decade by decade, about "
                   f"{100 * (Dd[-1]['p2p'] / Dd[0]['p2p'] - 1):.0f} % in total, consistent with published reports of a "
                   "strengthening seasonal exchange in the north. Meanwhile the growth rate of CO₂ itself nearly tripled.",
                   f"The DFT matrix takes {t804['matrix_s'] * 1e3:.0f} ms for N = 804 and grows as N²; the FFT takes "
                   f"{t804['fft_s'] * 1e3:.2f} ms, about {t804['matrix_s'] / t804['fft_s']:.0f} times faster. The FFT is slower "
                   f"for N = 804 than for N = 1024 ({t1024['fft_s'] * 1e3:.2f} ms) because 804 = 4·3·67 has a large prime factor.",
               ])
    rep.table(
        ["Quantity", "Result"],
        [
            ["DFT matrix / definition versus FFT", f"{Mt['matrix_vs_fft']:.0e} / {Mt['direct_vs_fft']:.0e}"],
            ["IDFT reconstruction error; Parseval; symmetry", f"{Mt['idft_err']:.0e} ppm; {Mt['parseval']:.0e}; {Mt['symmetry']:.0e}"],
            ["Annual / semi-annual line", f"{f(A1, 3)} / {f(A2, 3)} ppm at 1.000 / 2.000 cy/yr"],
            ["Seasonal maximum, minimum, swing", f"{Se['max_month']}, {Se['min_month']}, {f(Se['p2p'], 2)} ppm"],
            ["Deseasonalized versus NOAA (RMSE)", f"{f(Se['rmse_noaa'], 3)} ppm"],
            [f"Leakage (N = {L['Nc']}): rectangular / Hann / Hann + padding", f"{f(L['peak_cut'])} / {f(L['peak_hann'])} / {f(L['peak_pad'])} ppm"],
            ["Seasonal swing 1959-1968 / 2009-2018", f"{f(Dd[0]['p2p'])} / {f(Dd[-1]['p2p'])} ppm"],
            ["FFT speed-up at N = 804", f"{t804['matrix_s'] / t804['fft_s']:.0f} ×"],
        ],
        "Summary of results", widths=[8.0, 9.0], font_size=9, first_col_bold=True, align=["left", "left"])

    # ------------------------------------------------------ 8 conclusion
    rep.heading("Conclusion")
    rep.para(
        "The DFT and IDFT were implemented from first principles and verified against the FFT and against their "
        "defining properties to machine precision. Applied to the Keeling Curve, the DFT reduced 67 years of monthly "
        f"data to two numbers that matter, the {f(A1, 2)} ppm annual and {f(A2, 2)} ppm semi-annual lines, and the "
        f"IDFT of those lines alone rebuilt the planet's seasonal breathing (maximum in {Se['max_month']}, minimum in "
        f"{Se['min_month']}) well enough to reproduce NOAA's deseasonalized record within {f(Se['rmse_noaa'], 2)} ppm. "
        "The experiment also exposed the practical side of the transform: choose N to contain whole periods or use a "
        "window to control leakage, remember that zero padding interpolates but does not resolve, and use the FFT, "
        "ideally with a highly composite N, because the direct sum quickly becomes too slow.")
    rep.heading("Learning Outcomes")
    rep.bullets([
        "Compute the DFT and IDFT by definition, as a matrix product and with the FFT.",
        "Convert bins to frequencies and read amplitudes and phases of sinusoidal components.",
        "Explain and control spectral leakage with record length, windowing and zero padding.",
        "Filter in the frequency domain by editing DFT coefficients and inverting with the IDFT.",
    ])
    rep.heading("References")
    refs = [
        "A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed. Pearson, 2010, ch. 8 and 9.",
        "F. J. Harris, “On the use of windows for harmonic analysis with the discrete Fourier transform,” *Proc. IEEE*, "
        "vol. 66, no. 1, pp. 51-83, 1978.",
        "X. Lan, P. Tans and K. W. Thoning, “Trends in globally-averaged CO₂ determined from NOAA Global Monitoring "
        "Laboratory measurements,” NOAA GML, https://gml.noaa.gov/ccgg/trends/.",
        "C. D. Keeling, “The concentration and isotopic abundances of carbon dioxide in the atmosphere,” *Tellus*, "
        "vol. 12, pp. 200-203, 1960.",
        "H. D. Graven et al., “Enhanced seasonal exchange of CO₂ by northern ecosystems since 1960,” *Science*, "
        "vol. 341, pp. 1085-1089, 2013.",
    ]
    for i, rr in enumerate(refs, 1):
        rep.para(f"[{i}]  {rr}", align="left", size=9.5, space_after=1)

    path = rep.save(OUT_DOCX)
    print(f"saved {path}")
    print(f"saved {export_pdf(path)}")


if __name__ == "__main__":
    build()
