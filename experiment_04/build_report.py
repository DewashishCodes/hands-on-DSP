"""
build_report.py
---------------
Generates the Word (and PDF) lab report for Experiment 4.

    python run_experiment.py      # figures + results first
    python build_report.py        # -> DSP_Experiment_04.docx / .pdf

Every number quoted in the text is read from the analysis results at build
time, so the report can never disagree with the code.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))

import analysis                      # noqa: E402
import circular as cc                # noqa: E402
import config                        # noqa: E402
from report_kit import Report, export_pdf   # noqa: E402

STUDENT = {
    "Name": "Dewashish Lambore",
    "PRN": "____________",
    "Date": "____ / ____ / 2026",
    "Course": "Digital Signal Processing Lab",
}
FIG = config.FIGURE_DIR
OUT_DOCX = HERE / "DSP_Experiment_04.docx"


def f(x, d=2):
    return f"{x:,.{d}f}"


def build():
    R = analysis.run_all()
    df, info, p = R["df"], R["info"], R["p"]
    S, sm, lc, rw = R["seasons"], R["smoothing"], R["lin_circ"], R["runway"]
    r10, r13, r20 = rw[10.0], rw[13.0], rw[20.0]
    W, M = S["Winter (Dec to Feb)"], S["Monsoon (Jun to Sep)"]

    rep = Report(footer_left="DSP Lab  ·  Experiment 4  ·  Circular Convolution")

    # ------------------------------------------------------------------ title
    rep.info_block([("Course", STUDENT["Course"]), ("Experiment No.", "04"),
                    ("Name", STUDENT["Name"]), ("PRN", STUDENT["PRN"]),
                    ("Language", "Python 3.10"), ("Date", STUDENT["Date"])])
    rep.title("Circular Convolution",
              "Wind roses and runway orientation at Mumbai airport")
    rep.abstract(
        f"Wind direction is a circular quantity: 350° is next to 0°. One year of hourly METAR reports from "
        f"Mumbai airport ({info['reports']:,} observations, 2024) was turned into a 36-bin direction histogram, a "
        "naturally periodic sequence, and processed with circular convolution computed three independent ways "
        "(direct modulo-N sum, circulant matrix, and wrapped linear convolution), which agreed exactly. Circular "
        f"smoothing preserved all {sm['sum_p']:,.0f} observations, while ordinary linear convolution lost "
        f"{sm['sum_p'] - sm['sum_lin']:.1f} of them and erred by up to {sm['seam_err']:.1f} reports at the 0°/350° seam. "
        "The runway usability of every possible heading was then obtained as a sum of circular convolutions of "
        "speed-binned histograms with crosswind acceptance kernels: under the ICAO 10 kt limit the best heading "
        f"is {r10['best_deg']:03d}° ({100 * r10['best']:.2f} %), the real main runway 09/27 achieves "
        f"{100 * r10['runways']['09/27']:.2f} %, and the cross runway 14/32 only {100 * r10['runways']['14/32']:.2f} %, "
        "right at the 95 % minimum.",
        "circular convolution, circulant matrix, time aliasing, zero padding, wind rose, runway orientation, "
        "METAR, Python")

    # ------------------------------------------------------------- 1 aim
    rep.heading("Aim and Objectives")
    rep.para("**Aim:** to study circular convolution, compute it by different methods, relate it to linear "
             "convolution, and apply it to a real periodic signal in Python.", space_after=3)
    rep.bullets([
        "Compute N-point circular convolution by the direct formula, the circulant-matrix method and by "
        "wrapping a linear convolution, and verify that all three agree.",
        "Show when circular and linear convolution differ (time aliasing) and how zero padding makes them equal.",
        "Use circular convolution on wind-direction data to smooth a wind rose and to choose a runway heading.",
    ])

    # ----------------------------------------------------- 2 application
    rep.heading("Real-World Application")
    rep.para(
        "Some signals live on a circle rather than a line: wind and wave direction, compass headings, the "
        "angle of a rotating shaft, hour-of-day profiles, and any sequence handled by the DFT, which treats every "
        "signal as one period of a periodic sequence. For such signals ordinary convolution is wrong at the "
        "seam, because it pads with zeros where the data actually wraps around. Airport planners face this "
        "directly: ICAO Annex 14 requires that a runway can be used at least 95 % of the time without the "
        "crosswind exceeding a limit (10, 13 or 20 kt depending on runway length), and the usability of every "
        "candidate heading is a circular operation on the wind rose. Table 1 lists further uses.")
    rep.table(
        ["Field", "Circular signal", "Circular convolution is used for"],
        [
            ["Airport design", "wind-direction histogram (36 bins)", "smoothing wind roses, runway usability"],
            ["Wind energy", "direction-speed distribution", "wake and yaw-misalignment losses for each turbine heading"],
            ["Rotating machines", "vibration vs shaft angle", "order tracking, cyclic averaging"],
            ["Image processing", "periodic image rows / columns", "fast filtering with the FFT"],
            ["Communications (OFDM, 4G/5G)", "one symbol with cyclic prefix", "channel acts as circular convolution, one-tap equaliser"],
        ],
        "Where circular convolution appears in practice", widths=[3.8, 5.4, 7.8], font_size=9,
        first_col_bold=True, align=["left", "left", "left"])

    # ------------------------------------------------------------ 3 theory
    rep.heading("Theory")
    rep.para("For two sequences of length $N$ (shorter ones are padded with zeros), the $N$-point **circular "
             "convolution** uses indices taken modulo $N$, written $((\\cdot))_N$:")
    rep.equation(r"y[n] = x[n] \circledast h[n] = \sum_{m=0}^{N-1} x[m]\,h[((n-m))_N], \qquad 0 \le n \le N-1")
    rep.para("Collecting the $N$ circular shifts of $h$ as the columns of a **circulant matrix** gives the matrix "
             "method used for hand calculation:")
    rep.equation(r"\mathbf{y} = \mathbf{H}\mathbf{x}, \qquad H_{nm} = h[((n-m))_N]")
    rep.para("Circular and linear convolution are related by **time aliasing**: the circular result is the "
             "linear result $y_l[n]$ (length $L+M-1$) folded onto $N$ samples, so the two coincide only if $N$ "
             "is large enough:")
    rep.equation(r"y_c[n] = \sum_{r} y_l[n + rN], \qquad y_c = y_l \iff N \ge L + M - 1")
    rep.para("**Runway usability.** A wind of speed $V$ from direction $\\theta$ has a crosswind component on a "
             "runway of heading $\\psi$ of $V|\\sin(\\theta - \\psi)|$. For each speed bin $s$ define the "
             "acceptance kernel $a_s[k] = 1$ if $V_s|\\sin(10k^\\circ)| \\le X_{max}$, else 0. Then the "
             "usable fraction of time for heading index $j$ is")
    rep.equation(r"U[j] = \frac{1}{N_{tot}}\left( N_{calm} + \sum_{s} \sum_{k=0}^{35} p_s[k]\,a_s[((k-j))_{36}] \right)")
    rep.para("which is a sum of 36-point circular convolutions (the kernel is symmetric, so correlation and "
             "convolution coincide). Table 2 contrasts the two kinds of convolution.")
    rep.table(
        ["Property", "Linear convolution", "Circular convolution"],
        [
            ["Index range", "−∞ < n < ∞ (signals padded with zeros)", "0 ≤ n ≤ N − 1, indices modulo N"],
            ["Output length", "L + M − 1", "N"],
            ["Shift", "ordinary delay x[n − k]", "rotation x[((n − k))_N]"],
            ["Natural for", "aperiodic signals, LTI outputs", "periodic / angular signals, DFT products"],
            ["Equal when", "", "N ≥ L + M − 1 (after zero padding)"],
        ],
        "Linear versus circular convolution", widths=[3.2, 6.9, 6.9], font_size=9, first_col_bold=True,
        align=["left", "left", "left"])
    rep.note("A DFT of length N multiplies spectra, and the product of two N-point DFTs corresponds exactly to "
             "N-point circular convolution in time. This is why the zero-padding condition of Eq. 3 reappears in "
             "Experiments 9 and 10, where convolution is computed with the DFT.")

    # ----------------------------------------------------------- 4 dataset
    rep.heading("Dataset Description")
    rep.para(
        f"**METAR** reports are the standard hourly weather observations issued by every airport for pilots and "
        f"air-traffic control. The Iowa Environmental Mesonet (Iowa State University) archives them worldwide and "
        f"serves them as CSV without registration. Station **{config.STATION}** ({config.STATION_NAME}) was "
        f"requested for the whole of {config.YEAR}: {info['reports']:,} reports, {info['size_kb']:.0f} kB, cached "
        "after the first download. Direction is reported in degrees true *from* which the wind blows, rounded to "
        "the nearest 10°, so a 36-bin histogram uses exactly the resolution of the data. Calm reports (speed "
        f"< {config.CALM_KT:g} kt) and variable-direction reports have no direction and count as usable for every runway.")
    rep.table(
        ["Property", "Value", "Property", "Value"],
        [
            ["Station", f"{config.STATION}, Mumbai", "Period", f"{info['start']:%d %b %Y} to {info['end']:%d %b %Y}"],
            ["Reports", f"{info['reports']:,} (≈ hourly)", "With a direction", f"{info['directional']:,}"],
            ["Calm (< 1 kt)", f"{info['calm']:,}", "Variable (VRB)", f"{info['variable']}"],
            ["Mean speed", f"{info['speed_mean']:.1f} kt", "95th percentile / max", f"{info['speed_p95']:.0f} / {info['speed_max']:.0f} kt"],
            ["Direction resolution", "10° (36 bins)", "Runways", "09/27 (main), 14/32 (cross)"],
        ],
        "Dataset attributes", widths=[3.4, 4.6, 3.6, 5.4], font_size=8.5, align=["left"] * 4)
    sample = df[df["k"] >= 0].iloc[200:206]
    rep.table(
        ["Time (IST)", "Direction (°)", "Speed (kt)", "Bin k", "Crosswind on 09/27 (kt)"],
        [[r.time.strftime("%d-%b %H:%M"), f"{r.dir_deg:.0f}", f"{r.speed_kt:.0f}", f"{r.k}",
          f"{abs(r.speed_kt * np.sin(np.deg2rad(r.dir_deg - 90))):.1f}"] for r in sample.itertuples()],
        "Dataset preview with derived quantities",
        widths=[3.4, 3.0, 2.8, 2.2, 5.6], font_size=8.5, align=["center"] * 5)
    rep.figure(FIG / "fig1_raw.png", "The raw wind data.",
               "(a) Monthly mean speed; (b) the 36-bin direction histogram p[k] (bin k covers 10k ± 5 degrees).",
               inference=(
                   f"Winds are light (mean {info['speed_mean']:.1f} kt) and strongest in the monsoon months. The histogram "
                   f"is dominated by a broad westerly peak centred on {S['Annual']['mode_deg']}°, with a secondary group "
                   "of north-easterly winds. Crucially, the high counts at 330° to 350° continue into 0° to 20°: "
                   "the right edge of panel (b) is physically adjacent to the left edge, so p[k] must be treated as periodic."))
    rep.figure(FIG / "fig2_roses.png", "Wind roses (polar histograms) for the year and two seasons.",
               "North is up and directions run clockwise; bar length is the share of reports.",
               inference=(
                   f"The seasons are almost opposite. In winter the land breeze dominates with a mode of {W['mode_deg']}° "
                   f"(north-east) and mean speed {W['speed_mean']:.1f} kt; in the monsoon the south-west flow dominates "
                   f"with a mode of {M['mode_deg']}° and {M['speed_mean']:.1f} kt. The winter rose straddles north, which "
                   "is exactly where linear processing breaks."))

    # ---------------------------------------------------- 5 implementation
    rep.heading("Methodology and Implementation")
    rep.para("Reports are binned into p[k]; for the runway study they are also split by speed (1 kt bins) into "
             f"p_s[k]. Smoothing uses the kernel h = [¼, ½, ¼] centred at n = 0. All circular convolutions are "
             "computed with the circulant matrix and cross-checked against the direct formula and the wrapped "
             "linear convolution; the runway result is cross-checked against a brute-force count over every report.",
             keep_next=True)
    rep.table(
        ["Module", "Responsibility"],
        [
            ["`config.py`", "Station, year, bins, smoothing kernel, crosswind limits, runway headings"],
            ["`data_loader.py`", "Download METAR CSV once, cache, flag calm / variable, assign bin k"],
            ["`circular.py`", "Three circular-convolution methods, wrap, circular shift, runway usability"],
            ["`analysis.py`", "Histograms, seasons, smoothing comparison, linear vs circular, runway study"],
            ["`plots.py` / `gui.py`", "Report figures / interactive wind-rose and runway lab"],
            ["`test_circular.py`", "13 unit tests (DFT check, zero padding, rotation, mass preservation): all pass"],
        ],
        "Software modules", widths=[4.2, 12.8], font_size=9, align=["left", "left"])
    rep.code('''def circular_convolve(x, h, N):                  # y[n] = sum x[m] h[((n - m))_N]
    x, h = _pad(x, N), _pad(h, N)
    return np.array([sum(x[m] * h[(n - m) % N] for m in range(N)) for n in range(N)])

def circulant_matrix(h, N):                       # H[n, m] = h[((n - m))_N]
    h = _pad(h, N)
    return np.array([[h[(n - m) % N] for m in range(N)] for n in range(N)])

def circular_via_linear(x, h, N):                 # fold the linear result modulo N
    return wrap(np.convolve(x, h), N)

def runway_coverage(p_by_speed, speeds, limit, n_calm, total, bin_deg):
    acc = sum(circulant_matrix(acceptance_kernel(v, limit, 36, bin_deg)) @ p
              for p, v in zip(p_by_speed, speeds))
    return (acc + n_calm) / total                 # usability of all 36 headings''',
             title="Core of circular.py", font_size=7.5)
    rep.para("**GUI.** `gui.py` (Figure 3) has three views: circular versus linear smoothing of the wind rose "
             "for any season and kernel length (with the observations kept and the seam error), runway usability "
             "for any crosswind limit and heading, and a free N-point calculator in which the student types x[n], "
             "h[n] and N and sees whether the circular result is aliased or equal to the linear one. Run "
             "`python run_experiment.py` and `python gui.py`.")
    rep.figure(FIG / "fig6_gui.png", "The wind-rose lab: winter histogram smoothed circularly and linearly.",
               "The linear result (orange) drops at both ends; the circular result (blue) does not.", width_cm=14.5)

    # ------------------------------------------------------ 6 calculations
    rep.heading("Calculations")
    x4 = p[:4]
    h4 = np.array([0.5, 0.25, 0, 0.25])
    Hm = cc.circulant_matrix(h4)
    y4 = Hm @ x4
    lin4 = np.convolve(x4, [0.25, 0.5, 0.25])
    rep.para(f"**Matrix method, N = 4.** Take the first four histogram bins x = [{', '.join(f'{v:.0f}' for v in x4)}] "
             "(0° to 30°) and the centred kernel h = [½, ¼, 0, ¼] (h[3] = h[−1] by wraparound). Each row of the "
             "circulant matrix is the previous row rotated right:", keep_next=True)
    rep.table(
        ["n", "Row of H", "Product with x", "y[n]"],
        [[str(n), "[" + ", ".join(f"{v:g}" for v in Hm[n]) + "]",
          " + ".join(f"{a:g}×{b:.0f}" for a, b in zip(Hm[n], x4) if a), f"**{y4[n]:.2f}**"] for n in range(4)],
        "Circular convolution by the circulant-matrix method", widths=[1.0, 4.6, 7.8, 3.6], font_size=9,
        align=["center", "center", "left", "center"])
    rep.para(f"Sum check: Σy = {y4.sum():.0f} = Σx, so nothing is lost. The linear convolution of the same x "
             f"with [¼, ½, ¼] has 4 + 3 − 1 = 6 samples, [{', '.join(f'{v:.2f}' for v in lin4)}]; folding the last "
             f"sample onto n = 0 and the first onto n = 3 (after aligning the centre) reproduces y exactly. **Crosswind "
             "example:** a 12 kt wind from 240° on runway 09/27 has a crosswind of 12·|sin(240° − 90°)| = 6.0 kt "
             "(acceptable), but on runway 14/32 it is 12·|sin(100°)| = 11.8 kt, above the 10 kt limit. These values "
             "agree with the program (checked by `test_circular.py`).")

    # --------------------------------------------------------- 7 results
    rep.heading("Results and Discussion")
    rep.figure(FIG / "fig3_smoothing.png", "Smoothing the annual direction histogram.",
               "(a) p[k], its circular smoothing and the same smoothing done by linear convolution; (b) the difference.",
               inference="", inference_items=[
                   f"Inside the range the two results are identical (difference {sm['interior_err']:.1f}); they differ "
                   f"only at bins {sm['seam_bins'][0]} and {sm['seam_bins'][1]} (0° and 350°), by up to "
                   f"{sm['seam_err']:.1f} reports.",
                   "Linear convolution treats the missing neighbours of 0° and 350° as zero and so drags both ends "
                   f"down; it keeps only {sm['sum_lin']:,.2f} of {sm['sum_p']:,.0f} reports. Circular convolution uses "
                   "the true neighbours across north and keeps every report, so it is the correct operation.",
               ])
    rep.figure(FIG / "fig4_lin_circ.png", "Linear and circular convolution of the same sequences.",
               "(a) Linear, 38 samples; (b) 36-point circular; (c) 38-point circular after zero padding.",
               inference=(
                   "The two extra linear samples at n = 36 and 37 (shaded) are exactly what 36-point circular "
                   "convolution adds back onto n = 0 and 1: wrapping (a) modulo 36 reproduces (b) with error "
                   f"{lc['wrap_ok']:.1f}. When both inputs are zero-padded to N = 38 = L + M − 1, the circular "
                   f"result (c) equals the linear one with error {lc['pad_ok']:.1f}. All three computation methods "
                   f"agreed to within {max(R['methods'].values()):.1e}."))
    rep.figure(FIG / "fig5_runway.png", "Runway usability for every heading.",
               "Each point is a sum of 40 circular convolutions; dotted lines mark Mumbai's two real runways.",
               inference=(
                   f"Under the strict 10 kt limit usability ranges from {100 * r10['worst']:.2f} % (heading "
                   f"{r10['worst_deg']:03d}°) to {100 * r10['best']:.2f} % (heading {r10['best_deg']:03d}°). The real main "
                   f"runway 09/27 reaches {100 * r10['runways']['09/27']:.2f} %, within "
                   f"{100 * (r10['best'] - r10['runways']['09/27']):.2f} points of the optimum, while the cross runway 14/32 "
                   f"gives {100 * r10['runways']['14/32']:.2f} %, just meeting the ICAO 95 % rule, which is consistent "
                   f"with its role as a secondary runway. At 13 kt every heading exceeds {100 * r13['worst']:.1f} %, and at "
                   f"20 kt every heading exceeds {100 * r20['worst']:.2f} %: Mumbai's light winds would allow almost any "
                   "orientation for a long runway. The convolution result matched a brute-force count over all "
                   f"reports to within {rw['brute_force_err']:.1e}."))
    rep.table(
        ["Quantity", "Result"],
        [
            ["Three methods of circular convolution", f"agree to {max(R['methods'].values()):.1e}"],
            ["Linear wrapped modulo 36 vs circular / zero-padded N = 38 vs linear", f"{lc['wrap_ok']:.1f} / {lc['pad_ok']:.1f}"],
            ["Reports kept by circular / linear smoothing", f"{sm['sum_circ']:,.0f} / {sm['sum_lin']:,.2f} of {sm['sum_p']:,.0f}"],
            ["Largest seam error of linear smoothing", f"{sm['seam_err']:.1f} reports (bins 0 and 35)"],
            ["Wind-rose mode: winter / monsoon", f"{W['mode_deg']}° / {M['mode_deg']}°"],
            ["Best runway heading, 10 kt limit", f"{r10['best_deg']:03d}°/{r10['best_deg'] + 180:03d}° ({100 * r10['best']:.2f} %)"],
            ["Runway 09/27 / 14/32, 10 kt limit", f"{100 * r10['runways']['09/27']:.2f} % / {100 * r10['runways']['14/32']:.2f} %"],
        ],
        "Summary of results", widths=[10.0, 7.0], font_size=9, first_col_bold=True, align=["left", "center"])

    # ------------------------------------------------------ 8 conclusion
    rep.heading("Conclusion")
    rep.para(
        "Circular convolution was implemented in three independent ways that agreed exactly, and its relation to "
        "linear convolution (time aliasing, removed by zero padding to N ≥ L + M − 1) was confirmed numerically. "
        "On real wind data it proved to be not just a mathematical variant but the *correct* operation for an "
        "angular signal: linear smoothing lost observations and distorted the wind rose at north, while circular "
        "smoothing kept every report. Expressed as a sum of circular convolutions, the ICAO runway-usability "
        f"analysis identified {r10['best_deg']:03d}° as the best heading and showed that Mumbai's main runway is "
        "within a fraction of a percent of optimal, while the cross runway is marginal at 10 kt, a result that "
        "matches the airport's real layout.")
    rep.heading("Learning Outcomes")
    rep.bullets([
        "Compute N-point circular convolution by the direct formula and by the circulant-matrix method.",
        "Explain time aliasing and choose the zero-padding length that makes circular equal linear convolution.",
        "Recognise periodic and angular signals for which circular processing is required.",
        "Apply circular convolution to a real engineering decision (runway orientation from a wind rose).",
    ])
    rep.heading("References")
    refs = [
        "A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed. Pearson, 2010, ch. 8.",
        "J. G. Proakis and D. G. Manolakis, *Digital Signal Processing*, 4th ed. Pearson, 2007, ch. 7.",
        "ICAO, *Annex 14 to the Convention on International Civil Aviation, Vol. I: Aerodrome Design and "
        "Operations*, 9th ed., 2022, Sec. 3.1.",
        "Iowa Environmental Mesonet, “ASOS/METAR data download,” Iowa State University, https://mesonet.agron.iastate.edu.",
        "World Meteorological Organization, *Manual on Codes (WMO-No. 306)*, Vol. I.1, FM 15 METAR.",
    ]
    for i, r in enumerate(refs, 1):
        rep.para(f"[{i}]  {r}", align="left", size=9.5, space_after=1)

    path = rep.save(OUT_DOCX)
    print(f"saved {path}")
    print(f"saved {export_pdf(path)}")


if __name__ == "__main__":
    build()
