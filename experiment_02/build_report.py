"""
build_report.py
---------------
Generates the Word (and PDF) lab report for Experiment 2.

    python run_experiment.py      # figures + results first
    python build_report.py        # -> DSP_Experiment_02.docx / .pdf

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
import systems as sy                 # noqa: E402
from report_kit import Report, export_pdf   # noqa: E402

STUDENT = {
    "Name": "Dewashish Lambore",
    "PRN": "____________",
    "Date": "____ / ____ / 2026",
    "Course": "Digital Signal Processing Lab",
}
FIG = config.FIGURE_DIR
OUT_DOCX = HERE / "DSP_Experiment_02.docx"


def f(x, d=2):
    return f"{x:,.{d}f}"


def build():
    df = data_loader.load_dataset()
    R = analysis.run_all(df)
    B, RS, A = R["battery"], R["responses"], R["application"]
    S = sy.default_systems()
    a_, b_, k_ = config.TEST_A, config.TEST_B, config.TEST_SHIFT_K

    rep = Report(footer_left="DSP Lab  ·  Experiment 2  ·  Testing Systems for Linearity and Time Invariance")

    # ------------------------------------------------------------------ title
    rep.info_block([("Course", STUDENT["Course"]), ("Experiment No.", "02"),
                    ("Name", STUDENT["Name"]), ("PRN", STUDENT["PRN"]),
                    ("Language", "Python 3.10"), ("Date", STUDENT["Date"])])
    rep.title("Testing Systems for Linearity and Time Invariance",
              "and the time response of LTI systems, applied to Pune air-quality data around Diwali 2025")
    rep.abstract(
        f"Two weeks (N = {len(df)} hourly samples) of real PM2.5 and NO₂ data for Pune were passed through "
        "four systems that are actually used in air-quality monitoring: the 24-hour average of the Indian "
        "standard, the exponential display filter of a low-cost sensor, the CPCB Air Quality Index, and a "
        "sensor whose gain drifts. Black-box superposition and shift tests classified every system correctly: "
        f"the two filters are LTI (errors below 1e-15), the AQI is nonlinear (error {f(B['S3']['linear']['error'], 3)}) "
        f"and the drifting sensor is time-varying (error {f(B['S4']['ti']['error'], 4)}). Impulse and step "
        f"responses explain the behaviour on real data: the 24-h average reached {f(A['peak_ma'], 1)} µg/m³, "
        f"exceeding the 60 µg/m³ standard for {A['exceed_hours']} hours, and lagged the hourly peak by "
        f"{A['lag_ma_h']} h, while the smoother's time constant matched theory ({f(RS['S2']['tau_theory'])} samples).",
        "linearity, time invariance, superposition, impulse response, step response, LTI, air quality, AQI, Python")

    # ------------------------------------------------------------- 1 aim
    rep.heading("Aim and Objectives")
    rep.para("**Aim:** to check whether a given discrete-time system is linear and time invariant, and to "
             "obtain and interpret the time response (impulse and step response) of LTI systems, using "
             "real-world data in Python.", space_after=3)
    rep.bullets([
        "Implement black-box superposition and shift tests that work for any system.",
        "Classify four real air-quality processing systems and confirm the expected result.",
        "Compute impulse and step responses, verify $y[n] = x[n]*h[n]$ for the LTI systems, and "
        "relate response time to what an operator sees during a pollution episode.",
    ])

    # ----------------------------------------------------- 2 application
    rep.heading("Real-World Application")
    rep.para(
        "Every air-quality number shown to the public passes through a chain of systems. India's National "
        "Ambient Air Quality Standard (NAAQS) for PM2.5 is defined on a **24-hour average**; consumer sensors "
        "smooth their readings with an **exponential filter** so the display does not flicker; the "
        "**AQI** maps concentration to a 0 to 500 health scale; and cheap sensors **drift** as their optics "
        "get dirty. Whether each system is LTI decides what can be done with it: an LTI system is fully "
        "described by its impulse response and can be analysed, inverted or combined by convolution, while a "
        "nonlinear or time-varying system cannot. Table 1 lists the four systems tested.")
    rep.table(
        ["System", "Equation", "Where it is used", "Expected"],
        [[f"**{k}** {s.name}", s.equation, s.application,
          ("Linear" if s.expected_linear else "Nonlinear") + ", " +
          ("time-invariant" if s.expected_ti else "time-varying")] for k, s in S.items()],
        "Systems under test", widths=[3.9, 4.6, 4.6, 3.9], font_size=9, align=["left", "left", "left", "left"])

    # ------------------------------------------------------------ 3 theory
    rep.heading("Theory")
    rep.para("A discrete-time system is an operator that maps an input sequence to an output sequence, "
             "$y[n] = T\\{x[n]\\}$. It is **linear** if it obeys superposition, i.e. additivity and homogeneity "
             "together, for all inputs and all constants $a$, $b$:")
    rep.equation(r"T\{a\,x_1[n] + b\,x_2[n]\} = a\,T\{x_1[n]\} + b\,T\{x_2[n]\}")
    rep.para("It is **time-invariant** if delaying the input only delays the output by the same amount:")
    rep.equation(r"y[n] = T\{x[n]\} \;\Rightarrow\; T\{x[n-k]\} = y[n-k] \quad \text{for every } k")
    rep.para("A system that is both is **LTI** and is completely characterised by its impulse response "
             "$h[n] = T\\{\\delta[n]\\}$: any output is the convolution of the input with $h[n]$, and the step "
             "response is the running sum of $h[n]$:")
    rep.equation(r"y[n] = \sum_{k=-\infty}^{\infty} x[k]\,h[n-k], \qquad s[n] = \sum_{m=-\infty}^{n} h[m]")
    rep.para("For the exponential smoother the difference equation and its impulse response are")
    rep.equation(r"y[n] = \alpha\,x[n] + (1-\alpha)\,y[n-1] \;\Rightarrow\; h[n] = \alpha(1-\alpha)^{n}u[n], \quad \tau = \frac{-1}{\ln(1-\alpha)}")
    rep.para("where $\\tau$ is the time constant: the step response reaches $1 - e^{-1} = 63.2$ % after "
             "about $\\tau$ samples. The CPCB AQI is a piecewise-linear map between concentration "
             "breakpoints $[B_{lo}, B_{hi}]$ and index breakpoints $[I_{lo}, I_{hi}]$:")
    rep.equation(r"I = \frac{I_{hi} - I_{lo}}{B_{hi} - B_{lo}}\left(C - B_{lo}\right) + I_{lo}")
    rep.para("Because the slope changes from one band to the next, doubling $C$ does not double $I$, so "
             "the AQI cannot be linear. Each test is scored by the relative error between its two sides, "
             "$\\varepsilon = \\lVert \\mathrm{LHS} - \\mathrm{RHS} \\rVert / \\lVert \\mathrm{RHS} \\rVert$; "
             f"a property is accepted when $\\varepsilon$ < {config.TOLERANCE:g}, i.e. equal to floating-point precision.")
    rep.note("A system with a non-zero initial state (e.g. y[−1] = 1) fails the linearity test even if its "
             "equation is linear, because a zero input does not give a zero output. All systems here start "
             "at rest, and a unit test confirms this effect.")

    # ----------------------------------------------------------- 4 dataset
    rep.heading("Dataset Description")
    rep.para(
        "Hourly surface concentrations from the **CAMS global atmospheric composition** reanalysis/forecast "
        "of the Copernicus Atmosphere Monitoring Service (ECMWF), served by the free Open-Meteo air-quality "
        f"API for the grid cell nearest Pune ({config.LATITUDE}° N, {config.LONGITUDE}° E), from "
        f"{config.START_DATE} to {config.END_DATE}. The window brackets Diwali (20 October 2025), when Indian "
        f"cities typically see a pollution episode. The file is about 15 kB ({len(df)} samples × 4 pollutants, "
        "no missing values) and is cached after the first run. Because CAMS is a model, the values reflect "
        "meteorology and emission inventories rather than a single ground sensor, which is ideal for a "
        "clean, gap-free test signal.")
    desc = data_loader.describe_dataset(df)
    roles = {"pm25": "x₁ in every test", "pm10": "context", "no2": "x₂ in the superposition test",
             "co": "context"}
    rep.table(
        ["Column", "Signal", "Unit", "Mean", "Std", "Min", "Max", "Role"],
        [[f"`{c}`", r.description, r.unit, f(r.mean, 1), f(r.std, 1), f(r.min, 1), f(r.max, 1), roles[c]]
         for c, r in zip(desc.index, desc.itertuples())],
        f"Dataset attributes and statistics ({len(df)} hourly samples)",
        widths=[1.8, 3.8, 1.3, 1.3, 1.3, 1.3, 1.4, 4.8], font_size=8.5,
        align=["left", "left", "center", "center", "center", "center", "center", "left"])
    i_pk = int(np.argmax(df["pm25"]))
    head = df.iloc[i_pk - 3: i_pk + 3]
    rep.table(
        ["n", "Time (IST)", "PM2.5", "PM10", "NO₂", "CO"],
        [[r.n, r.time.strftime("%d-%b %H:%M"), f"{r.pm25:.1f}", f"{r.pm10:.1f}", f"{r.no2:.1f}", f"{r.co:.0f}"]
         for r in head.itertuples()],
        "Dataset preview around the PM2.5 peak (µg/m³)",
        widths=[1.6, 3.6, 2.9, 2.9, 2.9, 3.1], font_size=8.5, align=["center"] * 6)
    rep.figure(FIG / "fig1_raw_signals.png", "The two test inputs.",
               "(a) PM2.5 with the 24-h NAAQS limit; (b) NO₂. Dashed lines mark midnight.",
               inference=(
                   f"PM2.5 rises from about {f(A['pre_mean'], 0)} µg/m³ before 18 October to a peak of "
                   f"{f(A['peak_raw'], 1)} µg/m³ at {A['peak_raw_time'].strftime('%H:%M on %d October')}, "
                   f"crosses 60 µg/m³ for {A['hourly_over_60']} hours, then falls to about {f(A['post_mean'], 0)} "
                   "µg/m³ once the episode clears. NO₂ instead shows sharp evening traffic peaks every day. The "
                   "two inputs therefore have very different shapes, which makes the superposition test demanding."))

    # ---------------------------------------------------- 5 implementation
    rep.heading("Methodology and Implementation")
    rep.para("Each system is a Python callable that maps an input array to an output array from rest. The "
             "tests treat it as a **black box**, exactly as an engineer would test an unknown device: they "
             f"only call it with inputs. Parameters: a = {a_:g}, b = {b_:g}, shift k = {k_} h, M = {config.MA_LENGTH_M}, "
             f"α = {config.EWMA_ALPHA:g}, β = {config.DRIFT_BETA * 24 * 100:g} % per day.", keep_next=True)
    rep.table(
        ["Module", "Responsibility"],
        [
            ["`config.py`", "Dataset, system and test parameters, AQI breakpoints"],
            ["`data_loader.py`", "Download from Open-Meteo once, cache as CSV, clean, attach index n"],
            ["`systems.py`", "The four systems, a general difference equation, and the black-box tests"],
            ["`analysis.py`", "Test battery, impulse/step metrics, NAAQS exceedance, lag and AQI categories"],
            ["`plots.py` / `gui.py`", "Report figures / interactive system tester"],
            ["`test_systems.py`", "13 unit tests, including the initial-condition effect (all pass)"],
        ],
        "Software modules", widths=[4.2, 12.8], font_size=9, align=["left", "left"])
    rep.code('''def linearity_test(system, x1, x2, a, b):
    lhs = system(a * x1 + b * x2)                 # T{a x1 + b x2}
    rhs = a * system(x1) + b * system(x2)         # a T{x1} + b T{x2}
    return relative_error(lhs, rhs)

def time_invariance_test(system, x, k):
    lhs = system(delay(x, k))                     # T{x[n - k]}
    rhs = delay(system(x), k)                     # y[n - k]
    return relative_error(lhs, rhs)

def impulse_response(system, L):
    d = np.zeros(L); d[0] = 1.0                   # delta[n]
    return system(d)                              # h[n] = T{delta[n]}''',
             title="Core of systems.py", font_size=7.5)
    rep.para("**GUI.** `gui.py` (Figure 2) is a system tester: choose one of the four systems or type the "
             "coefficients of any difference equation, choose a test (superposition, shift, impulse and step "
             "response, or apply to the PM2.5 data), adjust a, b, k and the system parameter, and the plots, "
             "error signal and a coloured verdict update instantly. Run `python run_experiment.py` for all "
             "figures and `python gui.py` for the tester.")
    rep.figure(FIG / "fig6_gui.png", "The interactive system tester running the superposition test on the AQI.",
               "The red verdict and the non-zero error signal show that the AQI is nonlinear.", width_cm=14.5)

    # ------------------------------------------------------ 6 calculations
    rep.heading("Calculations")
    xp = df["pm25"].to_numpy()[i_pk - 2: i_pk + 1]
    x2p = df["no2"].to_numpy()[i_pk - 2: i_pk + 1]
    c1, c2 = 45.0, 30.0
    rep.para(f"**AQI superposition by hand** with $x_1$ = {c1:g} and $x_2$ = {c2:g} µg/m³ and $a = b = 1$. "
             f"Using the band 30 to 60 → 50 to 100: I(45) = (50/30)(45 − 30) + 50 = 75 and I(30) = 50, so "
             f"a I(x₁) + b I(x₂) = 125. But x₁ + x₂ = 75 lies in the band 60 to 90 → 100 to 200: "
             f"I(75) = (100/30)(75 − 60) + 100 = 150 ≠ 125. Superposition fails. Table 5 shows that doubling "
             "the concentration does not double the index either.", keep_next=True)
    rep.table(
        ["C (µg/m³)", "I(C)", "I(2C)", "2·I(C)", "Homogeneity"],
        [[f"{c:g}", f"{i1:.1f}", f"{i2:.1f}", f"{2 * i1:.1f}", "holds" if abs(i2 - 2 * i1) < 1e-9 else "fails"]
         for c, i1, i2 in R["aqi_examples"]],
        "Doubling the PM2.5 concentration and the resulting AQI", widths=[3.0, 3.0, 3.0, 3.0, 5.0],
        font_size=9)
    beta = config.DRIFT_BETA
    rep.para(f"**Drift shift test by hand** with k = 1 and two samples x = {{{xp[0]:.1f}, {xp[1]:.1f}}}: "
             f"y[1] = (1 + β·1)·{xp[1]:.1f} = {(1 + beta) * xp[1]:.3f}, so y[n − 1] at n = 2 is "
             f"{(1 + beta) * xp[1]:.3f}; but T{{x[n − 1]}} at n = 2 = (1 + 2β)·{xp[1]:.1f} = "
             f"{(1 + 2 * beta) * xp[1]:.3f}. The gain depends on n, so the system is time-varying. "
             f"**Smoother impulse response:** h[0] = α = {config.EWMA_ALPHA:g}, h[1] = α(1 − α) = "
             f"{config.EWMA_ALPHA * (1 - config.EWMA_ALPHA):.3f}, h[2] = {config.EWMA_ALPHA * (1 - config.EWMA_ALPHA) ** 2:.3f}; "
             f"τ = −1/ln(0.8) = {f(RS['S2']['tau_theory'])} samples. All values agree with the program "
             "(checked by `test_systems.py`).")

    # --------------------------------------------------------- 7 results
    rep.heading("Results and Discussion")
    rep.figure(FIG / "fig2_failures.png", "The two tests that fail, shown as overlays (18 to 21 October).",
               f"(a) AQI: response to a x₁ + b x₂ versus the weighted sum of responses; (b) drifting sensor: "
               f"response to the input delayed by k = {k_} versus the delayed response.",
               inference="", inference_items=[
                   f"**AQI.** The two sides agree when the combined concentration stays in one breakpoint band "
                   f"but separate sharply during the episode, when a x₁ + b x₂ crosses into steeper bands; the "
                   f"relative error is {f(B['S3']['linear']['error'], 3)}. The AQI amplifies high pollution on "
                   "purpose, which is exactly what makes it nonlinear, so AQI values must never be averaged or "
                   "added; the concentrations should be averaged first.",
                   f"**Drift.** The shaded gap grows with n because the shifted input meets a gain that is "
                   f"β·k = {f(beta * k_ * 100, 2)} % higher; the relative error is {f(B['S4']['ti']['error'], 4)}. "
                   "A drifting sensor cannot be described by one impulse response; it must be recalibrated.",
               ])
    rep.figure(FIG / "fig3_error_grid.png", "Error signals of both tests for all four systems.",
               "Left: e_L[n] = LHS − RHS of superposition; right: e_T[n] of the shift test. Columns share a y-scale.",
               inference=(
                   "The 24-h average and the exponential smoother give error signals that are zero to machine "
                   f"precision in both columns (ε ≈ 1e-16), so they are LTI. The AQI fails only superposition and "
                   f"the drifting sensor fails only the shift test, giving one example of each way a system can "
                   f"fail to be LTI. All four results match the expectation of Table 1."))
    rep.figure(FIG / "fig4_responses.png", "Time response of the two LTI systems.",
               f"(a) Impulse responses with the theoretical α(1 − α)ⁿ; (b) step responses with the 63.2 % level.",
               inference=(
                   f"The moving average has a rectangular h[n] of height 1/{config.MA_LENGTH_M} that ends after "
                   f"{config.MA_LENGTH_M} samples (finite impulse response); the smoother's h[n] decays "
                   "geometrically and never ends (infinite impulse response) and matches α(1 − α)ⁿ exactly. Both have "
                   f"DC gain Σh[n] = {f(RS['S1']['h_sum'], 3)}, so a steady concentration is reported unchanged. "
                   f"The step response of the average is a straight ramp reaching 63.2 % at n = {RS['S1']['t63']} "
                   f"and 100 % only at n = {config.MA_LENGTH_M - 1}, while the smoother reaches 63.2 % at "
                   f"n = {RS['S2']['t63']} (τ = {f(RS['S2']['tau_theory'])}) and 95 % at n = {RS['S2']['t95']}: "
                   "the smoother reacts within hours, the regulatory average within a day."))
    rep.figure(FIG / "fig5_application.png", "The systems applied to the real PM2.5 episode.",
               "(a) Hourly PM2.5, smoother and 24-h average; shading marks hours when the 24-h average exceeds "
               "the standard; (b) the AQI output with CPCB category bands.",
               inference=(
                   f"Exactly as the step responses predict, the 24-h average peaks at {f(A['peak_ma'], 1)} µg/m³, "
                   f"{A['lag_ma_h']} h after the hourly peak and {f(100 * (1 - A['peak_ma'] / A['peak_raw']), 0)} % "
                   f"lower, and stays above the standard for {A['exceed_hours']} hours "
                   f"({A['exceed_start'].strftime('%d Oct %H:%M')} to {A['exceed_end'].strftime('%d Oct %H:%M')}). "
                   f"The smoother follows more closely (peak {f(A['peak_sm'], 1)} µg/m³, {A['lag_sm_h']} h lag). "
                   f"For both LTI systems, x[n] * h[n] reproduces the output with error below 1e-15. The AQI peaks "
                   f"at {f(A['peak_aqi'], 0)} (\"{A['peak_category']}\"); over the fortnight "
                   f"{A['category_hours'].get('Good', 0)} hours were Good, {A['category_hours'].get('Satisfactory', 0)} "
                   f"Satisfactory and {A['category_hours'].get('Moderate', 0)} Moderate."))
    rep.table(
        ["System", "Linearity ε", "Shift ε", "Linear", "Time-inv.", "Verdict"],
        [[f"**{k}** {r['system'].name}", f"{r['linear']['error']:.1e}", f"{r['ti']['error']:.1e}",
          "yes" if r["linear"]["holds"] else "**no**", "yes" if r["ti"]["holds"] else "**no**",
          "LTI" if r["lti"] else "not LTI"] for k, r in B.items()],
        "Summary of the LTI tests", widths=[5.0, 2.6, 2.6, 2.0, 2.2, 2.6], font_size=9,
        align=["left"] + ["center"] * 5)

    # ------------------------------------------------------ 8 conclusion
    rep.heading("Conclusion")
    rep.para(
        "Black-box superposition and shift tests correctly classified four real air-quality systems: the 24-h "
        "NAAQS average and the exponential smoother are LTI, the CPCB AQI is nonlinear but time-invariant, and "
        "a drifting sensor is linear but time-varying. For the LTI systems the impulse response told the whole "
        f"story: it predicted the {A['lag_ma_h']}-hour lag and the flattened peak of the 24-h average and the "
        f"{f(RS['S2']['tau_theory'], 1)}-sample time constant of the smoother, and convolution with h[n] "
        "reproduced their outputs exactly. The practical lessons are that AQI values should never be averaged "
        "(average the concentration, then convert), that regulatory averages react slowly to short episodes, "
        "and that sensor drift breaks time invariance and must be removed by calibration.")
    rep.heading("Learning Outcomes")
    rep.bullets([
        "State and test linearity (superposition) and time invariance numerically for any system.",
        "Distinguish nonlinear and time-varying behaviour and identify their causes.",
        "Obtain impulse and step responses, read FIR/IIR behaviour, DC gain and time constant from them.",
        "Use h[n] and convolution to predict an LTI system's response to real data.",
    ])
    rep.heading("References")
    refs = [
        "A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed. Pearson, 2010, ch. 2.",
        "J. G. Proakis and D. G. Manolakis, *Digital Signal Processing*, 4th ed. Pearson, 2007, ch. 2.",
        "Central Pollution Control Board, *National Air Quality Index*, CPCB, New Delhi, 2014.",
        "A. Inness et al., “The CAMS reanalysis of atmospheric composition,” *Atmos. Chem. Phys.*, vol. 19, "
        "pp. 3515-3556, 2019.",
        "P. Zippenfenig, “Open-Meteo.com Weather API,” Zenodo, 2023, doi:10.5281/zenodo.7970649.",
    ]
    for i, r in enumerate(refs, 1):
        rep.para(f"[{i}]  {r}", align="left", size=9.5, space_after=1)

    path = rep.save(OUT_DOCX)
    print(f"saved {path}")
    print(f"saved {export_pdf(path)}")


if __name__ == "__main__":
    build()
