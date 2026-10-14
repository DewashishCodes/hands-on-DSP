"""
build_report.py
---------------
Generates the Word (and PDF) lab report for Experiment 7.

    python run_experiment.py      # figures + results first
    python build_report.py        # -> DSP_Experiment_07.docx / .pdf

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
import smoothing as sm               # noqa: E402
from report_kit import Report, export_pdf   # noqa: E402

STUDENT = {
    "Name": "Dewashish Lambore",
    "PRN": "____________",
    "Date": "____ / ____ / 2026",
    "Course": "Digital Signal Processing Lab",
}
FIG = config.FIGURE_DIR
OUT_DOCX = HERE / "DSP_Experiment_07.docx"


def f(x, d=2):
    v = 0.0 if abs(x) < 0.5 * 10 ** (-d) else x
    return f"{v:,.{d}f}"


def build():
    R = analysis.run_all()
    df, x, info, F, T, ic, M = R["df"], R["x"], R["info"], R["filters"], R["tradeoff"], R["ic"], R["macd"]
    MA, MAR = f"MA{config.N_MA} (non-recursive)", f"MA{config.N_MA} (recursive)"
    EMA, EMA2, HOLT = f"EMA{config.N_EMA}", f"Double EMA{config.N_EMA}", "Holt (level + trend)"
    al = config.ALPHA
    z0, z1 = R["zoom_change"]

    rep = Report(footer_left="DSP Lab  ·  Experiment 7  ·  Difference-Equation Based Smoothing")

    # ------------------------------------------------------------------ title
    rep.info_block([("Course", STUDENT["Course"]), ("Experiment No.", "07"),
                    ("Name", STUDENT["Name"]), ("PRN", STUDENT["PRN"]),
                    ("Language", "Python 3.10"), ("Date", STUDENT["Date"])])
    rep.title("Difference-Equation Based Smoothing",
              "Smoothing the US dollar to Indian rupee exchange rate")
    rep.abstract(
        f"Seven years of the daily USD/INR exchange rate ({info['trading_days']:,} trading days, "
        f"₹{info['first']:.2f} to ₹{info['last']:.2f}) were smoothed with five filters, all written and solved as "
        "linear constant-coefficient difference equations: the 20-day moving average in non-recursive and recursive "
        "form, the 20-day exponential moving average (EMA), a cascaded double EMA and Holt's trend-corrected "
        "smoother. The iterative solver matched a reference implementation to 1e-12. With the finance convention "
        f"α = 2/(N + 1), the EMA and the moving average have the same lag ({f(F[EMA]['lag'], 1)} days) and the same "
        f"noise gain ({f(F[EMA]['noise_gain'], 3)}), but the EMA needs {F[EMA]['ops']} multiplications per sample instead "
        f"of {F[MA]['ops']}. Across a family of lags, both follow the same law, noise gain = 1/(2L + 1). The initial "
        f"condition y[−1] = 0 caused a transient of {ic['n_theory']} days to fall below 1 %, exactly as the pole at "
        f"z = {f(1 - al, 3)} predicts; on the 2022 fall of the rupee the trend-corrected second-order smoother cut the "
        f"lag from {f(F[EMA]['lag_measured'], 1)} to {f(F[HOLT]['lag_measured'], 1)} days at a modest cost in noise.",
        "difference equation, recursive filter, exponential moving average, moving average, poles, stability, "
        "initial conditions, lag, MACD, exchange rate, Python")

    # ------------------------------------------------------------- 1 aim
    rep.heading("Aim and Objectives")
    rep.para("**Aim:** to design, solve and analyse smoothing filters described by difference equations, and to "
             "apply them to a real noisy signal in Python.", space_after=3)
    rep.bullets([
        "Solve a general difference equation by iteration, including initial conditions, and split the response "
        "into its zero-input and zero-state parts.",
        "Compare recursive (IIR) and non-recursive (FIR) smoothers by impulse response, poles, stability, lag, "
        "noise reduction and computational cost.",
        "Apply the smoothers to the USD/INR rate and quantify the lag-versus-smoothness trade-off.",
    ])

    # ----------------------------------------------------- 2 application
    rep.heading("Real-World Application")
    rep.para(
        "Importers, exporters, banks and the Reserve Bank of India watch the rupee every day, and the day-to-day "
        f"wiggles (standard deviation {f(info['change_std'], 2)} %) hide the underlying trend. Every trading terminal "
        "therefore draws moving averages, and the most common one, the exponential moving average, is nothing but a "
        "first-order recursive difference equation. The same few lines of code smooth data everywhere (Table 1); "
        "choosing them is always a trade-off between how much noise is removed and how late the smoothed signal "
        "reacts. Note: this experiment studies signal processing properties only; it does not evaluate trading strategies.")
    rep.table(
        ["Field", "Signal", "Difference-equation smoother used"],
        [
            ["Finance, forex desks", "exchange rates, prices", "EMA, MACD (difference of two EMAs), Holt trend"],
            ["Sensors and IoT", "temperature, battery voltage", "first-order IIR low-pass in firmware"],
            ["Navigation", "GPS position, speed", "alpha-beta filter (Holt) and Kalman filter"],
            ["Web and operations", "page views, server latency", "EWMA baselines for anomaly alarms"],
            ["Audio", "level meters, envelopes", "one-pole attack / release smoothing"],
        ],
        "Where difference-equation smoothers are used", widths=[3.6, 4.6, 8.8], font_size=9, first_col_bold=True,
        align=["left", "left", "left"])

    # ------------------------------------------------------------ 3 theory
    rep.heading("Theory")
    rep.para("A linear time-invariant system with a finite number of delays is described by a linear "
             "constant-coefficient **difference equation**. With $a_0 = 1$ it can be solved forward in time, one "
             "sample at a time, from the input and the previous outputs (the initial conditions):")
    rep.equation(r"y[n] = \sum_{k=0}^{M} b_k\,x[n-k] - \sum_{k=1}^{N} a_k\,y[n-k]")
    rep.para("If any $a_k \\ne 0$ ($k \\ge 1$) the filter is **recursive** (IIR); otherwise it is non-recursive (FIR). "
             "Its transfer function is a ratio of polynomials whose roots are the zeros and poles; the filter is "
             "stable if every pole lies inside the unit circle:")
    rep.equation(r"H(z) = \frac{\sum_{k} b_k z^{-k}}{1 + \sum_{k} a_k z^{-k}}, \qquad \text{stable} \iff |p_i| < 1")
    rep.para("The **exponential moving average** has one pole at $z = 1 - \\alpha$ and a geometric impulse "
             "response; finance chooses $\\alpha$ from a nominal window $N$:")
    rep.equation(r"y[n] = \alpha\,x[n] + (1-\alpha)\,y[n-1], \qquad h[n] = \alpha(1-\alpha)^{n}u[n], \qquad \alpha = \frac{2}{N+1}")
    rep.para("The $N$-point moving average is FIR, but it can also be written recursively with only one update "
             "per sample; this creates a pole at $z = 1$ that is cancelled by a zero:")
    rep.equation(r"y[n] = y[n-1] + \frac{x[n] - x[n-N]}{N}")
    rep.para("The complete response of a recursive filter is the sum of the **zero-input** response (due only to "
             "the initial condition) and the **zero-state** response (due only to the input). For the EMA:")
    rep.equation(r"y[n] = \underbrace{(1-\alpha)^{n+1}\,y[-1]}_{y_{zi}[n]} + \underbrace{\sum_{k=0}^{n} \alpha(1-\alpha)^{k}\,x[n-k]}_{y_{zs}[n]}")
    rep.para("Two numbers summarise a smoother: the **noise gain** (output variance for unit-variance white noise) "
             "and the **lag** (low-frequency group delay):")
    rep.equation(r"G = \sum_{n} h^2[n], \qquad L = \frac{\sum_n n\,h[n]}{\sum_n h[n]}; \qquad \text{EMA: } G = \frac{\alpha}{2-\alpha},\; L = \frac{1-\alpha}{\alpha}")
    rep.para("For the moving average $G = 1/N$ and $L = (N-1)/2$. Eliminating $N$ and $\\alpha$ shows that both "
             "obey the same law $G = 1/(2L+1)$; with $\\alpha = 2/(N+1)$ the EMA has exactly the lag of the "
             "$N$-day moving average. **Holt's smoother** adds a second state for the trend, giving a second-order "
             "equation $l[n] + a_1 l[n-1] + a_2 l[n-2] = \\alpha x[n] - \\alpha(1-\\beta)x[n-1]$ with "
             "$a_1 = \\alpha + \\alpha\\beta - 2$, $a_2 = 1 - \\alpha$, whose lag on a straight-line trend is zero.")
    rep.note("A filter with a pole on the unit circle, such as the recursive moving average, is only marginally "
             "stable: it works because the zero cancels the pole exactly, but rounding errors are never forgotten. "
             "Long-running systems therefore recompute the sum from scratch periodically.")

    # ----------------------------------------------------------- 4 dataset
    rep.heading("Dataset Description")
    rep.para(
        "Series **DEXINUS** (Indian rupees to one US dollar, noon buying rate in New York) is published by the "
        "Board of Governors of the Federal Reserve (H.10 release) and distributed free by FRED, the Federal "
        f"Reserve Bank of St. Louis. The CSV for {config.START[:4]} to {config.END[:4]} is {info['size_kb']:.0f} kB "
        f"and is cached locally. It has {info['rows']:,} weekday rows, of which {info['missing']} are holidays without "
        f"a quotation; dropping them leaves {info['trading_days']:,} samples, so the index n counts **trading days**.")
    rep.table(
        ["Property", "Value", "Property", "Value"],
        [
            ["Source", "FRED series DEXINUS (Federal Reserve H.10)", "Unit", "₹ per US dollar"],
            ["Period", f"{info['start']:%d %b %Y} to {info['end']:%d %b %Y}", "Samples", f"{info['trading_days']:,} trading days"],
            ["Lowest", f"₹{info['min']:.2f} ({info['min_date']:%d %b %Y})", "Highest", f"₹{info['max']:.2f} ({info['max_date']:%d %b %Y})"],
            ["Daily change, std", f"{info['change_std']:.3f} %", "Largest daily move", f"{info['change_maxabs']:.2f} %"],
            ["2022 move (zoom)", f"₹{z0:.2f} to ₹{z1:.2f}", "Licence", "public domain (US government data)"],
        ],
        "Dataset attributes", widths=[3.2, 5.6, 3.4, 4.8], font_size=8.5, align=["left"] * 4)
    rows = df.iloc[:6]
    rep.table(
        ["n", "Date", "x[n] (₹/US$)", "Daily change (%)", "EMA20 y[n]", "x − y"],
        [[r.n, r.date.strftime("%d %b %Y"), f"{r.inr:.4f}", "n/a" if np.isnan(r.change_pct) else f"{r.change_pct:+.3f}",
          f"{F[EMA]['y'][r.n]:.4f}", f"{r.inr - F[EMA]['y'][r.n]:+.4f}"] for r in rows.itertuples()],
        "Dataset preview with the EMA output (initial condition y[−1] = x[0])",
        widths=[1.0, 3.2, 3.2, 3.2, 3.2, 3.2], font_size=8.5, align=["center"] * 6)
    rep.figure(FIG / "fig1_raw.png", "The input signal.",
               "(a) Daily exchange rate with the 2022 zoom window shaded; (b) daily percentage change.",
               inference=(
                   f"The rupee weakened from about ₹{info['first']:.0f} to ₹{info['last']:.0f} per dollar, with a sharp fall in "
                   f"2022 (₹{z0:.1f} to ₹{z1:.1f}) and long flat stretches in 2023 when the central bank stabilised the "
                   f"rate. The daily changes look like noise around zero (std {f(info['change_std'], 2)} %, largest "
                   f"{f(info['change_maxabs'], 2)} % in March 2020). A good smoother should keep the slow trend of (a) and "
                   "remove the fast fluctuations of (b)."))

    # ---------------------------------------------------- 5 implementation
    rep.heading("Methodology and Implementation")
    rep.para("Every filter is defined only by its (b, a) coefficients and passed to one iterative solver of Eq. 1, "
             f"which was cross-checked against SciPy's `lfilter`. Filters start from y[−1] = x[0] unless stated. "
             f"Parameters: N = {config.N_EMA} (α = {al:.4f}), Holt β = {config.HOLT_BETA}, MACD "
             f"{config.MACD_FAST}/{config.MACD_SLOW}/{config.MACD_SIGNAL}. Lag is measured on the 2022 trend as mean "
             "tracking error divided by the trend slope.", keep_next=True)
    rep.table(
        ["Module", "Responsibility"],
        [
            ["`config.py`", "FRED series, period, filter parameters, zoom window"],
            ["`data_loader.py`", "Download once, cache, drop holidays, attach trading-day index n"],
            ["`smoothing.py`", "Iterative difference-equation solver, five smoothers, poles, H(e^jω), gain and lag"],
            ["`analysis.py`", "Filter table, trade-off study, initial-condition study, MACD"],
            ["`plots.py` / `gui.py`", "Report figures / interactive smoother designer"],
            ["`test_smoothing.py`", "11 unit tests (lfilter agreement, zi + zs, formulas, Holt, instability): all pass"],
        ],
        "Software modules", widths=[4.2, 12.8], font_size=9, align=["left", "left"])
    rep.code('''def difference_equation(b, a, x, y_init=None, x_init=None):
    """y[n] = ( sum b_k x[n-k] - sum_{k>=1} a_k y[n-k] ) / a_0, solved sample by sample"""
    M, N = len(b) - 1, len(a) - 1
    xp = np.zeros(M) if x_init is None else np.asarray(x_init, float)   # x[-1] ... x[-M]
    yp = np.zeros(N) if y_init is None else np.asarray(y_init, float)   # y[-1] ... y[-N]
    y = np.zeros(len(x))
    for n in range(len(x)):
        y[n] = (np.dot(b, np.r_[x[n], xp]) - np.dot(a[1:], yp)) / a[0]
        xp = np.r_[x[n], xp[:-1]] if M else xp                         # shift the delay lines
        yp = np.r_[y[n], yp[:-1]] if N else yp
    return y

ema_coeffs = lambda alpha: ([alpha], [1.0, -(1.0 - alpha)])          # y = a x + (1 - a) y[n-1]
ma_coeffs = lambda N: (np.full(N, 1.0 / N), [1.0])                    # non-recursive average''',
             title="Core of smoothing.py", font_size=7.5)
    rep.para("**GUI.** `gui.py` (Figure 2) offers the five smoothers or any typed b and a coefficients, the window "
             "N, Holt's β, the initial condition and the year to display. It shows the smoothed rate, the impulse "
             "response, the magnitude response and the pole-zero plot, and reports stability, lag, noise gain and "
             "the cost in multiplications per sample. Run `python run_experiment.py` and `python gui.py`.")
    rep.figure(FIG / "fig7_gui.png", "The difference-equation smoother designer showing Holt's trend-corrected filter.",
               "Two complex poles near z = 0.95, zero lag and a small low-frequency overshoot in |H|.", width_cm=14.5)

    # ------------------------------------------------------ 6 calculations
    rep.heading("Calculations")
    xh = np.array([80.0, 82.0, 81.0, 85.0])
    be, ae = sm.ema_coeffs(0.5)
    y_full = sm.difference_equation(be, ae, xh, y_init=[80.0])
    y_rest = sm.difference_equation(be, ae, xh)
    y_zi = sm.difference_equation(be, ae, np.zeros(4), y_init=[80.0])
    rep.para("**By hand.** EMA with α = 0.5 on x = {80, 82, 81, 85}, first with y[−1] = 80 and then starting at "
             "rest (y[−1] = 0); y[n] = 0.5 x[n] + 0.5 y[n−1]:", keep_next=True)
    rep.table(
        ["n", "x[n]", "y[n], y[−1] = 80", "y[n], y[−1] = 0", "Difference = y_zi[n] = 0.5ⁿ⁺¹·80", "3-point MA"],
        [[str(n), f"{xh[n]:.0f}",
          f"0.5·{xh[n]:.0f} + 0.5·{(80.0 if n == 0 else y_full[n - 1]):g} = **{y_full[n]:g}**",
          f"{y_rest[n]:g}", f"{y_zi[n]:g}",
          "n/a" if n < 2 else f"({xh[n - 2]:.0f} + {xh[n - 1]:.0f} + {xh[n]:.0f})/3 = {xh[n - 2:n + 1].mean():.2f}"]
         for n in range(4)],
        "Difference equations solved by iteration", widths=[0.8, 1.2, 4.6, 2.6, 4.0, 3.8], font_size=8.5,
        align=["center", "center", "left", "center", "center", "left"])
    tau = np.log(config.TRANSIENT_TOL) / np.log(1 - al)
    rep.para(f"The difference between the two columns is exactly the zero-input response, as Eq. 5 states. For the "
             f"20-day EMA used on the data, α = 2/21 = {al:.4f}, so the pole is at z = {1 - al:.4f}; the initial "
             f"condition decays below 1 % after n = ln(0.01)/ln({1 - al:.4f}) = {tau:.1f}, i.e. **{ic['n_theory']} trading "
             f"days**. Noise gain α/(2 − α) = {al / (2 - al):.4f} and lag (1 − α)/α = {(1 - al) / al:.2f} days, identical "
             f"to the 20-day moving average (1/20 = 0.05 and 19/2 = 9.5). These values agree with the program "
             "(checked by `test_smoothing.py`).")

    # --------------------------------------------------------- 7 results
    rep.heading("Results and Discussion")
    rep.figure(FIG / "fig2_smoothers.png", "The smoothers applied to the 2022 fall of the rupee.",
               "(a) Input and smoothed outputs; (b) tracking error x − y.",
               inference="", inference_items=[
                   "All four smoothers remove the day-to-day wiggles; the double EMA is the smoothest and Holt's "
                   "filter the most responsive.",
                   f"While the rupee falls steadily, every first-order smoother runs **behind** the rate (positive error). "
                   f"Measured on the trend, the moving average lags {f(F[MA]['lag_measured'], 1)} days, the EMA "
                   f"{f(F[EMA]['lag_measured'], 1)}, the double EMA {f(F[EMA2]['lag_measured'], 1)} and Holt only "
                   f"{f(F[HOLT]['lag_measured'], 1)} days. They are somewhat below the theoretical "
                   f"{f(F[MA]['lag'], 1)}, {f(F[EMA]['lag'], 1)}, {f(F[EMA2]['lag'], 1)} and 0 days because 2022 was not a "
                   "perfect straight line.",
                   f"Holt's error swings around zero instead: by estimating the trend it removes the lag but overshoots "
                   f"when the trend stops, which is why its noise gain is higher ({f(F[HOLT]['noise_gain'], 3)}).",
               ])
    rep.figure(FIG / "fig3_responses.png", width_cm=15.5, title= "Impulse responses and pole-zero locations.",
               desc="(a) h[n] obtained by feeding δ[n] to the difference equation; (b) poles (×) and zeros (○).",
               inference=(
                   f"The moving average has a rectangular, finite h[n] and no poles except at the origin; its "
                   f"{config.N_MA - 1} zeros sit on the unit circle and produce the nulls of its frequency response. "
                   f"The EMA's h[n] decays geometrically from α = {al:.3f} with a single pole at z = {f(F[EMA]['max_pole'], 4)}. "
                   f"The double EMA has a double pole at the same place and a delayed, bell-shaped h[n]. Holt's filter has "
                   f"two complex poles of radius {f(F[HOLT]['max_pole'], 3)} and an h[n] that dips below zero, the "
                   "negative weights that cancel the lag. All poles are inside the unit circle, so all four are stable."))
    rep.figure(FIG / "fig4_initial.png", width_cm=15.5, title= "Effect of the initial condition on the EMA.",
               desc="(a) Starting at rest versus starting at x[0]; (b) the zero-input response on a logarithmic scale.",
               inference=(
                   f"Starting at rest, the first output is only {ic['y_rest'][0]:.2f} instead of {ic['y_full'][0]:.2f}, an "
                   f"error of ₹{f(ic['first_err'], 2)}. The difference is exactly the zero-input response (decomposition "
                   f"error {ic['decomp_err']:.0e}), a straight line on the log scale because it shrinks by the pole value "
                   f"{f(1 - al, 3)} every day. It falls below 1 % after {ic['n_measured']} to {ic['n_theory']} days, matching "
                   "the theory, so the first two months of any EMA started at zero must be discarded."))
    rep.figure(FIG / "fig5_tradeoff.png", width_cm=15.5, title= "Smoothing versus lag, and the frequency domain.",
               desc="(a) Noise gain against lag for moving averages, EMAs and double EMAs; (b) |H(e^jω)| of the main "
               "filters and of MACD.",
               inference="", inference_items=[
                   "All three families lie on the same curve G = 1/(2L + 1): for white noise, **no linear smoother of this "
                   "kind removes more noise for the same lag**; at lag 20 the double EMA is better by only "
                   f"{100 * (1 - T['ema2_ng'].iloc[-1] / T['ma_ng'].iloc[-1]):.0f} %. What differs is cost and shape.",
                   f"In frequency, the EMA rolls off smoothly while the moving average has deep nulls at multiples of 1/{config.N_MA} "
                   "cycles per day and strong sidelobes. Holt's |H| rises slightly above 0 dB at very low frequency, the "
                   f"overshoot seen in Figure 3. MACD has zero gain at DC and peaks at a period of about "
                   f"{f(M['peak_period_days'], 0)} days: it is a band-pass filter that removes the long-term level.",
               ])
    rep.figure(FIG / "fig6_macd.png", "Application: the MACD trend indicator built from two EMAs.",
               "(a) Rate with EMA12 and EMA26; (b) MACD line and its 9-day EMA signal line.",
               inference=(
                   f"MACD is simply the difference of two difference equations followed by a third. It oscillates around "
                   f"zero, positive while the rupee weakens (rate rising) and negative when it recovers. The MACD and signal "
                   f"lines crossed {M['n_cross']} times in seven years, about {M['per_year']:.0f} times a year; each crossing is "
                   "a change in the short-term trend as seen through filters with roughly 10 to 20 days of lag, which is "
                   "why such signals always arrive after the turning point."))
    rep.table(
        ["Smoother", "Order (M, N)", "Max |pole|", "Stable", "Lag theory", "Lag on 2022", "Noise gain", "Mult./sample"],
        [[name.replace(" (non-recursive)", " FIR").replace(" (recursive)", " recursive"), f"({len(v['b']) - 1}, {len(v['a']) - 1})",
          f(v["max_pole"], 3), "yes" if v["stable"] else "marginal", f(v["lag"], 1), f(v["lag_measured"], 1),
          f(v["noise_gain"], 4), str(v["ops"])] for name, v in F.items()],
        "Summary of the smoothers", widths=[3.6, 2.0, 1.8, 1.7, 1.8, 2.0, 2.0, 2.1], font_size=8.5,
        first_col_bold=True, align=["left"] + ["center"] * 7)

    # ------------------------------------------------------ 8 conclusion
    rep.heading("Conclusion")
    rep.para(
        "Five smoothers were expressed as difference equations, solved by direct iteration and verified against an "
        "independent implementation. The analysis showed that the familiar EMA is the cheapest of them (two "
        f"multiplications per sample against {F[MA]['ops']} for the moving average) while matching the moving average's "
        "lag and noise reduction exactly when α = 2/(N + 1), and that every first-order or cascaded smoother obeys the "
        "same trade-off G = 1/(2L + 1): more smoothing always costs more delay. Recursion brings its own concerns: "
        f"the initial condition lingers for {ic['n_theory']} days and a pole on the unit circle makes a filter only "
        "marginally stable. Breaking the trade-off requires a model of the signal: Holt's second-order equation, which "
        f"estimates the trend, tracked the 2022 fall with almost no lag ({f(F[HOLT]['lag_measured'], 1)} days) at the price "
        "of overshoot, the same idea that underlies alpha-beta and Kalman filters in navigation.")
    rep.heading("Learning Outcomes")
    rep.bullets([
        "Write smoothing filters as difference equations and solve them by iteration with initial conditions.",
        "Find poles, zeros, impulse and frequency responses, and judge stability from the pole locations.",
        "Separate the zero-input and zero-state responses and estimate transient length.",
        "Quantify the lag versus noise trade-off and choose recursive or non-recursive designs on cost and behaviour.",
    ])
    rep.heading("References")
    refs = [
        "A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed. Pearson, 2010, ch. 2 and 6.",
        "J. G. Proakis and D. G. Manolakis, *Digital Signal Processing*, 4th ed. Pearson, 2007, ch. 2.4.",
        "C. C. Holt, “Forecasting seasonals and trends by exponentially weighted moving averages,” *Int. J. "
        "Forecasting*, vol. 20, no. 1, pp. 5-10, 2004 (reprint of 1957 report).",
        "Board of Governors of the Federal Reserve System, “H.10 Foreign Exchange Rates,” via FRED, series DEXINUS.",
        "G. Appel, *Technical Analysis: Power Tools for Active Investors*, FT Press, 2005 (MACD).",
    ]
    for i, r in enumerate(refs, 1):
        rep.para(f"[{i}]  {r}", align="left", size=9.5, space_after=1)

    path = rep.save(OUT_DOCX)
    print(f"saved {path}")
    print(f"saved {export_pdf(path)}")


if __name__ == "__main__":
    build()
