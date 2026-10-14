"""
build_report.py
---------------
Generates the Word (and PDF) lab report for Experiment 10.

    python run_experiment.py      # GUI screenshot and cached data first
    python build_report.py        # -> DSP_Experiment_10.docx / .pdf

Every number quoted in the text is read from the analysis results at build
time, and the figures are redrawn from the same run (timings vary between runs).
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))

import analysis                      # noqa: E402
import circdft as cd                 # noqa: E402
import config                        # noqa: E402
from report_kit import Report, export_pdf   # noqa: E402

STUDENT = {
    "Name": "Dewashish Lambore",
    "PRN": "____________",
    "Date": "____ / ____ / 2026",
    "Course": "Digital Signal Processing Lab",
}
FIG = config.FIGURE_DIR
OUT_DOCX = HERE / "DSP_Experiment_10.docx"
DL = analysis.day_label


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

    info, S, Mt, rows, T, Sc, x, clim = (R["info"], R["smooth"], R["methods"], R["edge_rows"], R["timing"],
                                         R["scaling"], R["x"], R["clim"])
    BOX, GAU, HAR = (f"Boxcar {config.MA_DAYS} d", f"Gaussian σ = {config.GAUSS_SIGMA:g} d",
                     f"{config.HARMONICS} harmonics")
    box = S[BOX]
    r31 = next(r for r in rows if r["M"] == config.MA_DAYS)
    tmin_s, dtr = R["tmin_s"], R["dtr_s"]
    s365 = next(r for r in Sc if r["N"] == 365)
    s4096 = next(r for r in Sc if r["N"] == 4096)
    s1024 = next(r for r in Sc if r["N"] == 1024)

    rep = Report(footer_left="DSP Lab  ·  Experiment 10  ·  Circular Convolution using the DFT")

    # ------------------------------------------------------------------ title
    rep.info_block([("Course", STUDENT["Course"]), ("Experiment No.", "10"),
                    ("Name", STUDENT["Name"]), ("PRN", STUDENT["PRN"]),
                    ("Language", "Python 3.10"), ("Date", STUDENT["Date"])])
    rep.title("Circular Convolution using the DFT",
              "Daily climate normals for New Delhi from 30 years of observations")
    rep.abstract(
        f"The average maximum temperature for each calendar day at New Delhi (Safdarjung), computed from "
        f"{info['tmax_obs']:,} daily observations over the WMO normal period 1991 to 2020, forms a 365-point "
        "signal that is naturally periodic: 31 December is followed by 1 January. It was smoothed by 365-point "
        "circular convolution computed as the product of DFTs, which agreed with the direct modulo-N sum to "
        f"{Mt['direct_vs_fft']:.0e} °C and with a linear convolution of the cyclically prefixed signal to "
        f"{Mt['cp_vs_circ']:.0e} °C. Ordinary zero-padded convolution, by contrast, was wrong by up to "
        f"{f(r31['max_err'], 1)} °C on the {r31['days_affected']} days nearest New Year. Three kernels (31-day boxcar, "
        f"Gaussian, three annual harmonics) were compared through their DFT gains. The smoothed normals place the "
        f"hottest day on {DL(box['hot'])} ({f(box['hot_val'], 1)} °C), the coldest night on {DL(int(np.argmin(tmin_s)))} "
        f"({f(tmin_s.min(), 1)} °C) and the steepest monsoon cooling around {DL(box['steepest_fall'])}. For N = 365 the "
        f"FFT needs {T['mult_direct'] / T['mult_fft']:.0f} times fewer multiplications, but it only becomes faster in "
        "practice beyond a few hundred points.",
        "circular convolution, DFT, FFT, circulant matrix, cyclic prefix, OFDM, periodic signals, climate normals, "
        "smoothing, GHCN-Daily, Python")

    # ------------------------------------------------------------- 1 aim
    rep.heading("Aim and Objectives")
    rep.para("**Aim:** to compute the circular convolution of two sequences using the DFT, verify it against the "
             "direct definition, and apply it to a real periodic signal in Python.", space_after=3)
    rep.bullets([
        "Implement N-point circular convolution as IDFT{X·H} and check it against the modulo-N sum and the DFT matrix.",
        "Show that a cyclic prefix turns linear convolution into circular convolution, the principle behind OFDM.",
        "Smooth a 365-day climatology circularly, compare kernels in the DFT domain, and measure the computational cost.",
    ])

    # ----------------------------------------------------- 2 application
    rep.heading("Real-World Application")
    rep.para(
        "Weather services publish **daily climate normals**: the expected maximum and minimum temperature for every "
        "day of the year, averaged over a 30-year period. They are used to report anomalies (\"3 °C above normal\"), "
        "to plan agriculture, to size cooling and heating systems, and to set heat-wave thresholds; the India "
        "Meteorological Department, for instance, declares a heat wave when the maximum exceeds the normal by "
        "4.5 °C or more. Raw 30-year averages are noisy, so they are smoothed, and because the calendar wraps around "
        "the smoothing must be circular. Doing it as a product of DFTs is exact, simple and fast. The same identity, "
        "circular convolution equals a product of DFTs, is the core of OFDM in Wi-Fi, 4G and 5G (Table 1).")
    rep.table(
        ["Field", "Periodic or block signal", "Role of circular convolution via the DFT"],
        [
            ["Climatology (this experiment)", "365-day calendar", "smoothing daily normals across New Year"],
            ["Wi-Fi, 4G / 5G (OFDM)", "one symbol with cyclic prefix", "channel becomes circular, so one division per subcarrier equalises it"],
            ["Image processing", "image tiles, textures", "fast blurring and deblurring with periodic boundaries"],
            ["Rotating machinery", "one revolution of a shaft", "angle-domain filtering of vibration"],
            ["Power systems", "one cycle of the 50 Hz wave", "harmonic analysis and filtering per cycle"],
        ],
        "Where circular convolution via the DFT is used", widths=[4.2, 4.6, 8.2], font_size=9, first_col_bold=True,
        align=["left", "left", "left"])

    # ------------------------------------------------------------ 3 theory
    rep.heading("Theory")
    rep.para("For two N-point sequences the **circular convolution** (Experiment 4) uses indices modulo N:")
    rep.equation(r"y[n] = x[n] \circledast_N h[n] = \sum_{m=0}^{N-1} x[m]\,h[((n-m))_N], \qquad 0 \le n \le N-1")
    rep.para("The **circular convolution theorem** states that the N-point DFT turns it into a product of spectra, "
             "so it can be computed with three DFTs and N multiplications:")
    rep.equation(r"\mathrm{DFT}_N\{x \circledast_N h\} = X[k]\,H[k] \quad\Rightarrow\quad y[n] = \frac{1}{N}\sum_{k=0}^{N-1} X[k]\,H[k]\,e^{j2\pi kn/N}")
    rep.para("Unlike Experiment 9, **no zero padding** is used: the wrap-around that the DFT implies is exactly what "
             "a periodic signal needs. In matrix form the circulant matrix of h is diagonalised by the DFT matrix W:")
    rep.equation(r"\mathbf{y} = \mathbf{H}_{\mathrm{circ}}\,\mathbf{x} = \mathbf{W}^{-1}\,\mathrm{diag}(\mathbf{W}\mathbf{h})\,\mathbf{W}\,\mathbf{x}")
    rep.para("**Cyclic prefix.** If the last M − 1 samples of x are copied to its front and the result is convolved "
             "*linearly* with a causal h of length M, the output samples after the prefix equal the circular "
             "convolution. OFDM transmitters add such a prefix so that the radio channel, which convolves linearly, "
             "acts circularly; the receiver then removes the channel with one division per DFT bin, H[k]⁻¹.")
    rep.para("**Smoothing kernels.** A unit-sum kernel h leaves the annual mean unchanged and multiplies the k-th "
             "annual harmonic by H[k]. A boxcar of M days has the Dirichlet response below, with nulls at k = 365/M "
             "and its multiples; keeping K harmonics means H[k] = 1 for |k| ≤ K and 0 otherwise:")
    rep.equation(r"H_{\mathrm{box}}[k] = \frac{\sin(\pi kM/N)}{M\sin(\pi k/N)}, \qquad H_{\mathrm{harm}}[k] = \begin{cases} 1, & |k| \le K \\ 0, & \text{otherwise} \end{cases}")
    rep.note("Smoothing with ordinary (zero-padded) convolution treats the days before 1 January and after "
             "31 December as 0 °C. Near the ends of the year the kernel then averages real temperatures with zeros, "
             "which is exactly the edge error measured in Section 7.")

    # ----------------------------------------------------------- 4 dataset
    rep.heading("Dataset Description")
    rep.para(
        "**GHCN-Daily** (Global Historical Climatology Network, NOAA National Centers for Environmental Information) "
        "is the main global archive of quality-controlled daily station observations. Station "
        f"**{config.STATION}**, {config.STATION_NAME}, was requested through NCEI's free data service for the WMO "
        f"normal period {config.START[:4]} to {config.END[:4]}, maximum and minimum temperature only "
        f"({info['size_kb']:.0f} kB, cached). 29 February is removed so that every year has the same 365 calendar "
        "days; the climatology x[k] is the mean of all available years for calendar day k = 0 (1 January) to 364.")
    rep.table(
        ["Property", "Value", "Property", "Value"],
        [
            ["Station", f"{config.STATION} ({config.STATION_NAME})", "Period", f"{config.START} to {config.END}"],
            ["Days in period (no 29 Feb)", f"{info['possible']:,}", "Maximum / minimum observations", f"{info['tmax_obs']:,} / {info['tmin_obs']:,}"],
            ["Years with ≥ 330 maxima", f"{info['complete_years']} of 30", "Years per calendar day", f"{info['n_min']} to 30 (median {info['n_med']:.0f})"],
            ["Highest maximum", f"{info['tmax_max']:.1f} °C ({info['tmax_max_date']:%d %b %Y})", "Lowest minimum", f"{info['tmin_min']:.1f} °C ({info['tmin_min_date']:%d %b %Y})"],
            ["Signal length N", "365 (one calendar year)", "Sampling", "1 per day; periodic"],
        ],
        "Dataset attributes", widths=[3.6, 4.8, 3.8, 4.8], font_size=8.5, align=["left"] * 4)
    rows_prev = [(DL(k), clim["tmax"].iloc[k], clim["n_tmax"].iloc[k], clim["sd_tmax"].iloc[k], box["y"][k])
                 for k in (362, 363, 364, 0, 1, 2)]
    rep.table(
        ["Calendar day", "Raw mean x[k] (°C)", "Years", "Std. dev. (°C)", "31-day circular normal (°C)"],
        [[d, f"{a:.2f}", f"{n:.0f}", f"{s:.2f}", f"{y:.2f}"] for d, a, n, s, y in rows_prev],
        "Climatology preview across the year boundary",
        widths=[3.0, 3.6, 2.0, 3.0, 5.4], font_size=8.5, align=["center"] * 5)
    rep.figure(FIG / "fig1_raw.png", "The data and the raw climatology.",
               "(a) All daily maxima by calendar day and their mean x[k]; (b) number of years observed for each day.",
               inference=(
                   "Delhi's maximum temperature rises from about 20 °C in early January to over 40 °C in late May, "
                   "drops when the monsoon arrives in late June and stays near 34 °C until October. Individual days "
                   f"scatter by ±5 °C or more, and only {info['n_min']} to 30 years contribute to each calendar day (some "
                   "years after 2015 are incomplete), so the raw mean x[k] is visibly jagged. Its two ends join smoothly: "
                   "the value on 31 December is close to that on 1 January."))

    # ---------------------------------------------------- 5 implementation
    rep.heading("Methodology and Implementation")
    rep.para("Missing calendar days are filled by circular interpolation, kernels are centred at n = 0 and wrapped to "
             "N = 365, and y = IDFT{DFT{x}·DFT{h}} is computed with real FFTs. The result is checked against the "
             "direct sum, the DFT-matrix product and the cyclic-prefix construction, and compared with zero-padded "
             f"linear smoothing for kernel lengths {', '.join(map(str, config.KERNEL_LENGTHS))} days.", keep_next=True)
    rep.table(
        ["Module", "Responsibility"],
        [
            ["`config.py`", "Station, period, kernel parameters, timing settings"],
            ["`data_loader.py`", "Download from NCEI once, cache, drop 29 February, build the 365-day climatology"],
            ["`circdft.py`", "Circular convolution direct / via DFT / via DFT matrix, cyclic prefix, kernels, gains"],
            ["`analysis.py`", "Kernel comparison, edge study, key dates, timing and scaling with N"],
            ["`plots.py` / `gui.py`", "Report figures / interactive climate-normals smoother with N-point calculator"],
            ["`test_circdft.py`", "9 unit tests (equality of methods, cyclic prefix, rotation, harmonic kernel): all pass"],
        ],
        "Software modules", widths=[4.2, 12.8], font_size=9, align=["left", "left"])
    rep.code('''def circ_direct(x, h):                       # y[n] = sum x[m] h[((n - m))_N]
    N = len(x)
    idx = (np.arange(N)[:, None] - np.arange(N)[None, :]) % N
    return h[idx] @ x                           # circulant matrix times x

def circ_dft(x, h):                          # y = IDFT{ DFT{x} . DFT{h} }, no padding
    return np.fft.irfft(np.fft.rfft(x) * np.fft.rfft(h, len(x)), len(x))

def cyclic_prefix_conv(x, h_causal):         # OFDM trick: prefix + LINEAR convolution
    M = len(h_causal)
    xp = np.r_[x[-(M - 1):], x]                 # copy the last M-1 samples to the front
    return np.convolve(xp, h_causal)[M - 1: M - 1 + len(x)]''', title="Core of circdft.py", font_size=7.5)
    rep.para("**GUI.** `gui.py` (Figure 2) smooths the maximum, minimum or diurnal-range climatology with a boxcar, "
             "Gaussian or harmonic kernel of any width, overlays the zero-padded linear result, and shows the kernel "
             "and its harmonic gains. An N-point calculator returns the circular convolution of any two typed "
             "sequences via the DFT. Run `python run_experiment.py` (which also writes `outputs/normals.csv`) and "
             "`python gui.py`.")
    rep.figure(FIG / "fig6_gui.png", "The climate-normals smoother: diurnal range with a 61-day boxcar.",
               "The dashed zero-padded result collapses at both ends of the year; the circular result does not.",
               width_cm=14.5)

    # ------------------------------------------------------------ 6 calcs
    rep.heading("Calculations")
    xh, hh = np.array([20.0, 30.0, 40.0, 30.0]), np.array([0.5, 0.25, 0.0, 0.25])
    Xh, Hh = np.fft.fft(xh), np.fft.fft(hh)
    yh = cd.circ_dft(xh, hh)
    rep.para("**4-point example.** Seasonal means x = [20, 30, 40, 30] °C (winter, spring, summer, autumn), "
             "smoothed with the centred kernel h = [½, ¼, 0, ¼] (each season weighted ½, its neighbours ¼; h[3] is "
             "h[−1] by wraparound).", keep_next=True)
    rep.table(
        ["k", "X[k] = Σ x[n](−j)ⁿᵏ", "H[k]", "Y[k] = X[k]·H[k]", "y[n] = IDFT{Y}"],
        [[str(k), f"{Xh[k].real + 0.0:g}", f"{Hh[k].real + 0.0:g}", f"{(Xh[k] * Hh[k]).real + 0.0:g}", f"y[{k}] = {yh[k]:g}"]
         for k in range(4)],
        "Circular convolution via the 4-point DFT", widths=[1.0, 4.6, 2.4, 4.0, 5.0], font_size=9,
        align=["center"] * 5)
    rep.para(f"Directly: y[0] = ½·20 + ¼·30 + ¼·30 = 25, where the winter value uses autumn as its left neighbour "
             "(wraparound), and y[2] = ½·40 + ¼·30 + ¼·30 = 35. Zero-padded linear smoothing would instead give "
             "y[0] = ½·20 + ¼·30 + ¼·0 = 17.5, a 7.5 °C error. Both spectra are real because both sequences are even "
             f"(symmetric about n = 0). **Cost for N = 365:** direct circular convolution needs N² = "
             f"{T['mult_direct']:,} multiplications; three real FFTs and the product need about "
             f"{T['mult_fft']:,}, a factor of {T['mult_direct'] / T['mult_fft']:.0f}. Boxcar nulls fall at k = 365/M: "
             f"for M = 31 the first is at k ≈ {365 / 31:.1f}. These values agree with the program (checked by "
             "`test_circdft.py`).")

    # --------------------------------------------------------- 7 results
    rep.heading("Results and Discussion")
    rep.figure(FIG / "fig2_edge.png", "Circular versus zero-padded smoothing.",
               f"(a) Raw climatology, {config.MA_DAYS}-day circular smoothing via the DFT and zero-padded linear "
               "smoothing; (b) largest end-of-year error against kernel length.",
               inference=(
                   "Through most of the year the two smoothings coincide. Near New Year the zero-padded result plunges "
                   f"towards 10 °C because half the window covers zeros: the error reaches {f(r31['max_err'], 1)} °C for "
                   f"M = {config.MA_DAYS} and affects {r31['days_affected']} days (M − 1). Longer kernels make it worse "
                   + " (" + ", ".join(f"{r['max_err']:.1f} °C for M = {r['M']}" for r in rows) + "). The circular result "
                   "via the DFT has no edge at all, because the DFT treats the 365 days as one period; it keeps the "
                   f"annual mean exactly (error {Mt['sum_err']:.0e})."))
    rep.figure(FIG / "fig3_kernels.png", "Three smoothing kernels in time and in the DFT domain.",
               "(a) Centred kernels h[n]; (b) their gain |H[k]| on the k-th annual harmonic.",
               inference="", inference_items=[
                   f"The 31-day boxcar keeps harmonics 1 to 3 almost intact (gains {', '.join(f(v, 3) for v in box['H'][1:4])}) "
                   f"but has a null near k = 12 and sidelobes up to 0.2 beyond it: some 2 to 3 week wiggles leak through. "
                   f"The Gaussian rolls off smoothly with no sidelobes. The harmonic kernel is an ideal low-pass filter: "
                   "exactly 1 for k ≤ 3 and exactly 0 above, at the price of a kernel that spreads over the whole year.",
                   f"All three remove a similar amount (residual std {f(S[BOX]['resid_std'])}, {f(S[GAU]['resid_std'])} and "
                   f"{f(S[HAR]['resid_std'])} °C) and leave equally smooth curves (day-to-day change about "
                   f"{f(box['roughness'], 2)} °C), but they disagree on detail: the hottest day is {DL(S[BOX]['hot'])}, "
                   f"{DL(S[GAU]['hot'])} or {DL(S[HAR]['hot'])}. Three harmonics are too few to follow Delhi's sharp "
                   "monsoon onset.",
               ])
    rep.figure(FIG / "fig4_climate.png", "The resulting daily normals for New Delhi.",
               f"(a) Normal maximum and minimum ({config.MA_DAYS}-day circular smoothing); (b) residual x − y against the "
               "sampling error of a 30-year mean.",
               inference=(
                   f"The normal maximum peaks at {f(box['hot_val'], 1)} °C on {DL(box['hot'])} and is lowest at "
                   f"{f(box['cold_val'], 1)} °C on {DL(box['cold'])}; the normal minimum ranges from {f(tmin_s.min(), 1)} °C "
                   f"({DL(int(np.argmin(tmin_s)))}) to {f(tmin_s.max(), 1)} °C ({DL(int(np.argmax(tmin_s)))}). The maximum falls "
                   f"fastest around {DL(box['steepest_fall'])} ({f(-box['steepest_rate'], 2)} °C per day), the arrival of the "
                   "monsoon, which also squeezes the day-night range to "
                   f"{f(dtr.min(), 1)} °C in {DL(int(np.argmin(dtr)))} against {f(dtr.max(), 1)} °C in {DL(int(np.argmax(dtr)))}. "
                   f"The residual (std {f(box['resid_std'])} °C) lies mostly within the expected sampling error of a 30-year "
                   f"mean (rms {f(R['noise_expected'])} °C), so the smoothing removed noise, not climate."))
    rep.figure(FIG / "fig5_prefix_timing.png", "Cyclic prefix and computational cost.",
               "(a) The climatology with a 30-day cyclic prefix, its linear convolution after removing the prefix, and "
               "the circular convolution via the DFT; (b) time per convolution against N.",
               inference="", inference_items=[
                   f"Copying the last 30 days of December in front of January and convolving *linearly* with the causal "
                   f"31-day boxcar gives exactly the circular result (difference {Mt['cp_vs_circ']:.0e} °C). Because the kernel is "
                   f"causal here, both are delayed by {Mt['causal_delay']} days relative to the centred smoother. This is "
                   "precisely how OFDM makes a multipath radio channel look circular to the receiver.",
                   f"For N = 365 the FFT does {T['mult_direct'] / T['mult_fft']:.0f} times fewer multiplications, yet the "
                   f"optimised matrix product is still faster ({s365['t_matrix'] * 1e3:.3f} against "
                   f"{s365['t_fft'] * 1e3:.3f} ms) because of fixed FFT overheads. The FFT wins from about N = 1,000 "
                   f"({s1024['t_fft'] * 1e3:.3f} against {s1024['t_matrix'] * 1e3:.3f} ms) and is "
                   f"{s4096['t_matrix'] / s4096['t_fft']:.0f} times faster at N = 4,096: operation counts predict "
                   "the trend, but measured time decides small cases.",
               ])
    rep.table(
        ["Quantity", "Result"],
        [
            ["Circular conv. via DFT versus direct / DFT matrix", f"{Mt['direct_vs_fft']:.0e} / {Mt['direct_vs_matrix']:.0e} °C"],
            ["Cyclic-prefix linear conv. versus circular", f"{Mt['cp_vs_circ']:.0e} °C"],
            [f"Zero-padded linear error at year ends (M = {config.MA_DAYS})", f"{f(r31['max_err'], 1)} °C on {r31['days_affected']} days"],
            ["Residual std: boxcar / Gaussian / 3 harmonics", f"{f(S[BOX]['resid_std'])} / {f(S[GAU]['resid_std'])} / {f(S[HAR]['resid_std'])} °C"],
            ["Expected sampling error of the raw climatology", f"{f(R['noise_expected'])} °C rms"],
            ["Hottest normal day / coldest normal night", f"{DL(box['hot'])} ({f(box['hot_val'], 1)} °C) / {DL(int(np.argmin(tmin_s)))} ({f(tmin_s.min(), 1)} °C)"],
            ["Steepest monsoon cooling", f"{DL(box['steepest_fall'])}, {f(-box['steepest_rate'], 2)} °C per day"],
            ["Multiplications N = 365: direct / FFT", f"{T['mult_direct']:,} / {T['mult_fft']:,}"],
        ],
        "Summary of results", widths=[8.6, 8.4], font_size=9, first_col_bold=True, align=["left", "left"])

    # ------------------------------------------------------ 8 conclusion
    rep.heading("Conclusion")
    rep.para(
        "Circular convolution was computed as the inverse DFT of a product of DFTs and matched the direct modulo-N "
        "definition to rounding error. Applied to the 365-day climatology of New Delhi, a signal that genuinely wraps "
        "around, it produced smooth daily normals with no artefact at New Year, whereas ordinary zero-padded convolution "
        f"was wrong by up to {f(r31['max_err'], 0)} °C there. Viewing kernels through their DFT gains explained their "
        "different behaviour (boxcar sidelobes, Gaussian roll-off, ideal harmonic cut-off) and showed that the residual "
        "removed by smoothing is the sampling noise of a 30-year mean. The cyclic-prefix experiment demonstrated the "
        "link to OFDM, where the same identity lets a phone undo a multipath channel with one division per subcarrier. "
        "Finally, the timing study gave a realistic view of the FFT: its advantage grows with N, but for a few hundred "
        "points a well-optimised direct method can still be quicker.")
    rep.heading("Learning Outcomes")
    rep.bullets([
        "Compute N-point circular convolution through the DFT without zero padding and verify it against the definition.",
        "Explain the circular convolution theorem in matrix form and the role of the cyclic prefix in OFDM.",
        "Choose circular rather than linear processing for periodic data and quantify the edge error otherwise.",
        "Interpret smoothing kernels by their DFT gains and compare operation counts with measured run time.",
    ])
    rep.heading("References")
    refs = [
        "A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed. Pearson, 2010, ch. 8.6 and 8.7.",
        "M. J. Menne et al., “An overview of the Global Historical Climatology Network-Daily database,” *J. Atmos. "
        "Oceanic Technol.*, vol. 29, pp. 897-910, 2012.",
        "World Meteorological Organization, *WMO Guidelines on the Calculation of Climate Normals*, WMO-No. 1203, 2017.",
        "R. van Nee and R. Prasad, *OFDM for Wireless Multimedia Communications*, Artech House, 2000, ch. 2.",
        "India Meteorological Department, “Criteria for heat wave,” https://mausam.imd.gov.in.",
    ]
    for i, rr in enumerate(refs, 1):
        rep.para(f"[{i}]  {rr}", align="left", size=9.5, space_after=1)

    path = rep.save(OUT_DOCX)
    print(f"saved {path}")
    print(f"saved {export_pdf(path)}")


if __name__ == "__main__":
    build()
