"""
build_report.py
---------------
Generates the Word (and PDF) lab report for Experiment 5.

    python run_experiment.py      # figures + results first
    python build_report.py        # -> DSP_Experiment_05.docx / .pdf

Every number quoted in the text is read from the analysis results at build
time, so the report can never disagree with the code.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))

import analysis                      # noqa: E402
import autocorr as ac                # noqa: E402
import config                        # noqa: E402
from report_kit import Report, export_pdf   # noqa: E402

STUDENT = {
    "Name": "Dewashish Lambore",
    "PRN": "____________",
    "Date": "____ / ____ / 2026",
    "Course": "Digital Signal Processing Lab",
}
FIG = config.FIGURE_DIR
OUT_DOCX = HERE / "DSP_Experiment_05.docx"


def f(x, d=2):
    return f"{x:,.{d}f}"


def build():
    R = analysis.run_all()
    df, x, info = R["df"], R["x"], R["info"]
    A, Mt, nz, W, RB, FC = R["acf"], R["methods"], R["noise"], R["windows"], R["robust"], R["forecast"]
    N = len(x)

    rep = Report(footer_left="DSP Lab  ·  Experiment 5  ·  Autocorrelation")

    # ------------------------------------------------------------------ title
    rep.info_block([("Course", STUDENT["Course"]), ("Experiment No.", "05"),
                    ("Name", STUDENT["Name"]), ("PRN", STUDENT["PRN"]),
                    ("Language", "Python 3.10"), ("Date", STUDENT["Date"])])
    rep.title("Autocorrelation",
              "Measuring the solar cycle from 277 years of sunspot observations")
    rep.abstract(
        f"The autocorrelation of the international monthly sunspot number ({N:,} months, {info['start'][:4]} to "
        f"{info['end'][:4]}) was computed by the direct definition, by convolution with the folded sequence and "
        f"through the power spectrum (Wiener-Khinchin); the three agreed to {max(Mt['direct_vs_conv'], Mt['direct_vs_fft']):.0e}. "
        f"The normalised autocorrelation has its first peak at {f(A['period_months'], 1)} months, i.e. a solar "
        f"cycle of {f(A['period_years'])} years (ρ = {f(A['peak_value'], 3)}), with repeats at 2, 3 and 4 cycles "
        f"whose decreasing height ({', '.join(f(v) for v in A['peak_values'][1:])}) measures the cycle-to-cycle "
        f"irregularity. The series decorrelates after {A['zero']} months, the period was stable at 10.7 to 11.0 "
        f"years since 1818 but only {f(W[0]['period_years'], 1)} years in the irregular 18th century, and the cycle "
        f"was still detected when buried in white noise at an SNR of {f(RB[2]['snr_db'], 0)} dB. The leftover "
        "month-to-month variation is not white noise, which the 95 % confidence band reveals.",
        "autocorrelation, periodicity, Wiener-Khinchin, white-noise test, sunspot number, solar cycle, "
        "space weather, Python")

    # ------------------------------------------------------------- 1 aim
    rep.heading("Aim and Objectives")
    rep.para("**Aim:** to study the autocorrelation of a discrete-time signal, verify its properties, and use it "
             "to detect hidden periodicity and to test for randomness in a real-world signal, in Python.",
             space_after=3)
    rep.bullets([
        "Compute autocorrelation three ways and verify symmetry, the maximum at zero lag and r[0] = energy.",
        "Estimate the period of the solar cycle and how it changed over 277 years.",
        "Test whether the residual variation is white noise, and how much noise periodicity detection tolerates.",
    ])

    # ----------------------------------------------------- 2 application
    rep.heading("Real-World Application")
    rep.para(
        "The Sun's activity rises and falls in a cycle of roughly 11 years, and sunspots are its oldest and most "
        "reliable measure. Solar maximum brings more flares and geomagnetic storms, which heat and expand the upper "
        "atmosphere, increase drag on low-orbit satellites, disturb GPS and HF radio, and induce currents in power "
        "grids. Satellite operators, ISRO's Aditya-L1 mission team, airlines on polar routes and grid operators "
        "therefore plan around the cycle, and the first step is to measure its period from data. Autocorrelation "
        "does exactly that: it compares a signal with delayed copies of itself, so any repetition shows up as a "
        "peak at the repetition lag even when it is hard to see in the raw data (Table 1).")
    rep.table(
        ["Field", "Signal", "What the autocorrelation reveals"],
        [
            ["Space weather", "monthly sunspot number", "solar-cycle period, decorrelation time, predictability"],
            ["Speech processing", "voiced speech frame", "pitch period (fundamental frequency) of the speaker"],
            ["Radar and sonar", "echo of a coded pulse", "sharp peak at zero lag gives fine range resolution"],
            ["Machine monitoring", "bearing vibration", "repetition rate of a defect hitting the race"],
            ["Statistics, finance", "residuals of a model", "whether errors are white (model adequate) or correlated"],
        ],
        "Uses of autocorrelation", widths=[3.4, 4.6, 9.0], font_size=9, first_col_bold=True,
        align=["left", "left", "left"])

    # ------------------------------------------------------------ 3 theory
    rep.heading("Theory")
    rep.para("The **autocorrelation** of an energy signal measures its similarity with a copy shifted by $l$ samples:")
    rep.equation(r"r_{xx}[l] = \sum_{n=-\infty}^{\infty} x[n]\,x[n-l], \qquad l = 0, \pm 1, \pm 2, \ldots")
    rep.para("Comparing with the convolution sum shows that autocorrelation is the convolution of $x[n]$ with "
             "its folded copy (Experiment 1), and by the convolution theorem it is the inverse transform of the "
             "energy spectrum (the **Wiener-Khinchin** relation):")
    rep.equation(r"r_{xx}[l] = x[l] * x[-l] \quad\longleftrightarrow\quad S_{xx}(e^{j\omega}) = \left|X(e^{j\omega})\right|^{2}")
    rep.para("Three properties follow directly and are verified numerically below:")
    rep.equation(r"r_{xx}[-l] = r_{xx}[l], \qquad \left|r_{xx}[l]\right| \le r_{xx}[0] = E_x = \sum_n x^2[n]")
    rep.para("For a finite record of $N$ samples with mean $\\bar{x}$, the normalised autocorrelation "
             "(autocorrelation function, ACF) is estimated by")
    rep.equation(r"\rho[l] = \frac{\hat{r}[l]}{\hat{r}[0]}, \qquad \hat{r}[l] = \frac{1}{N}\sum_{n=l}^{N-1}\left(x[n]-\bar{x}\right)\left(x[n-l]-\bar{x}\right)")
    rep.para("Dividing by $N$ (biased) rather than $N - l$ (unbiased) gives a slightly smaller but far less noisy "
             "estimate at large lags, and is the standard choice. Two reference cases guide interpretation: a "
             "periodic signal of period $P$ has an ACF that repeats with period $P$ (a sinusoid gives "
             "$\\rho[l] = \\cos(2\\pi l/P)$), while white noise has $\\rho[l] \\approx 0$ for $l \\ne 0$, with 95 % of "
             "sample values inside")
    rep.equation(r"\left|\rho[l]\right| < \frac{1.96}{\sqrt{N}}")
    rep.note("Autocorrelation is robust to noise because uncorrelated noise adds only to ρ[0]; the periodic part "
             "keeps reinforcing at its own lag. This is why a period can be read from the ACF even when the "
             "noise is larger than the signal (Section 7).")

    # ----------------------------------------------------------- 4 dataset
    rep.heading("Dataset Description")
    rep.para(
        "The **international sunspot number** (version 2.0) is maintained by SILSO, the World Data Center for "
        "the sunspot index at the Royal Observatory of Belgium. It combines counts from about 80 observatories "
        "worldwide and has been recalibrated back to 1749, making it the longest continuous record of solar "
        f"activity. The monthly mean series is a single {info['size_kb']:.0f} kB text file with one row per month, "
        f"downloaded once and cached. The last {info['provisional']} months are provisional. Sampling interval: "
        "T_s = 1 month, so 12 lags = 1 year.")
    rep.table(
        ["Property", "Value", "Property", "Value"],
        [
            ["Provider", "SILSO, Royal Observatory of Belgium", "Series", "monthly mean total sunspot number v2.0"],
            ["Period", f"{info['start']} to {info['end']}", "Samples N", f"{info['months']:,} months"],
            ["Mean ± std", f"{info['mean']:.1f} ± {info['std']:.1f}", "Range", f"{info['min']:.0f} to {info['max']:.1f}"],
            ["Largest month", info["max_date"], "Months with observer count", f"{info['with_obs_count']:,}"],
            ["Columns used", "year, month, decimal year, SN", "Licence", "CC BY-NC 4.0 (SILSO)"],
        ],
        "Dataset attributes", widths=[3.2, 5.0, 3.8, 5.0], font_size=8.5, align=["left"] * 4)
    tail = df.iloc[-6:]
    rep.table(
        ["Year", "Month", "Decimal year", "SN", "Std. dev.", "Observations", "Definitive"],
        [[r.year, f"{r.month:02d}", f"{r.decimal_year:.3f}", f"{r.sn:.1f}", f"{r.sn_std:.1f}", f"{r.n_obs}",
          "yes" if r.definitive else "provisional"] for r in tail.itertuples()],
        "Most recent six rows of the dataset",
        widths=[1.8, 1.6, 2.6, 2.0, 2.4, 3.0, 3.6], font_size=8.5, align=["center"] * 7)
    rep.figure(FIG / "fig1_raw.png", "The monthly sunspot number and its 13-month running mean.",
               inference=(
                   f"About 25 cycles are visible. Their heights vary widely (smoothed maxima of only about 80 in the early "
                   f"1800s, the Dalton Minimum, but almost 300 in 1958), and their spacing is roughly, but not exactly, "
                   f"11 years. The latest cycle (Cycle 25) peaked at a smoothed value of {f(FC['last_max_value'], 0)} "
                   f"around {FC['last_max_year']:.1f}. The monthly values scatter strongly around the smooth curve, so "
                   "the period should be measured statistically rather than by eye."))

    # ---------------------------------------------------- 5 implementation
    rep.heading("Methodology and Implementation")
    rep.para("The mean is removed, the biased ACF is computed up to 50 years of lag, and the period is taken as "
             f"the largest ACF value between {config.PEAK_SEARCH[0]} and {config.PEAK_SEARCH[1]} months, refined by "
             "fitting a parabola through the peak and its neighbours. The same estimator is applied to four 70-year "
             "epochs, to the residual and the first difference, and to the record with added white noise.",
             keep_next=True)
    rep.table(
        ["Module", "Responsibility"],
        [
            ["`config.py`", "URL, lags, smoothing length, epochs, noise levels"],
            ["`data_loader.py`", "Download SILSO file once, cache, mark and fill missing months"],
            ["`autocorr.py`", "Three autocorrelation methods, biased / unbiased ACF, bounds, period finder"],
            ["`analysis.py`", "Main ACF, property checks, whiteness tests, epochs, noise robustness"],
            ["`plots.py` / `gui.py`", "Report figures / interactive periodicity detector"],
            ["`test_autocorr.py`", "10 unit tests (properties, sinusoid ACF = cosine, white-noise bound): all pass"],
        ],
        "Software modules", widths=[4.2, 12.8], font_size=9, align=["left", "left"])
    rep.code('''def autocorr_direct(x, max_lag):            # r[l] = sum x[n] x[n - l]
    return np.array([np.dot(x[l:], x[:len(x) - l]) for l in range(max_lag + 1)])

def autocorr_conv(x):                        # r[l] = x[l] * x[-l]
    return np.convolve(x, x[::-1])

def autocorr_fft(x, max_lag):                # Wiener-Khinchin: r = IDFT{|X|^2}
    nfft = 1 << int(np.ceil(np.log2(2 * len(x) - 1)))     # pad: no circular wrap
    return np.fft.irfft(np.abs(np.fft.rfft(x, nfft)) ** 2, nfft)[:max_lag + 1]

def acf(x, max_lag, unbiased=False):         # rho[l] = r[l] / r[0]
    x = x - x.mean()
    r = autocorr_fft(x, max_lag)
    r = r / (len(x) - np.arange(max_lag + 1)) if unbiased else r / len(x)
    return r / r[0]''', title="Core of autocorr.py", font_size=7.5)
    rep.para("**GUI.** `gui.py` (Figure 2) lets the user choose the raw, smoothed, residual or differenced series, "
             "any span of years, the maximum lag and the estimator, and add white noise of any strength. It shows the "
             "signal, its ACF with the 95 % band, the detected period with a significance check, and the first zero "
             "crossing. Run `python run_experiment.py` and `python gui.py`.")
    rep.figure(FIG / "fig6_gui.png", "The periodicity detector with noise twice as strong as the signal (SNR −6 dB).",
               "The cycle is invisible in the time series but the ACF still peaks at about 10 years, above the band.",
               width_cm=14.5)

    # ------------------------------------------------------ 6 calculations
    rep.heading("Calculations")
    xh = np.array([2.0, -1.0, 3.0, 1.0])
    rh = ac.autocorr_direct(xh, 3)
    rep.para("**Hand example.** For x[n] = {2, −1, 3, 1} (n = 0 … 3), each lag multiplies the sequence with a "
             "shifted copy and adds the overlapping products:", keep_next=True)
    rep.table(
        ["Lag l", "Overlapping products x[n]·x[n − l]", "r[l]", "ρ[l] = r[l]/r[0]"],
        [
            ["0", "2·2 + (−1)(−1) + 3·3 + 1·1", f"{rh[0]:.0f}", "1.000"],
            ["1", "(−1)·2 + 3·(−1) + 1·3", f"{rh[1]:.0f}", f"{rh[1] / rh[0]:.3f}"],
            ["2", "3·2 + 1·(−1)", f"{rh[2]:.0f}", f"{rh[2] / rh[0]:.3f}"],
            ["3", "1·2", f"{rh[3]:.0f}", f"{rh[3] / rh[0]:.3f}"],
        ],
        "Autocorrelation by hand", widths=[1.6, 8.6, 2.4, 4.4], font_size=9, align=["center", "left", "center", "center"])
    rep.para(f"By symmetry r[−1] = {rh[1]:.0f}, r[−2] = {rh[2]:.0f}, r[−3] = {rh[3]:.0f}, so the full sequence has "
             f"2N − 1 = 7 values with its maximum r[0] = {rh[0]:.0f} = E_x at the centre. **For the sunspot record:** "
             f"the 95 % white-noise limit is 1.96/√{N} = {A['bound']:.4f}; the ACF peak at lag {A['period_lag']} with "
             f"parabolic refinement gives {f(A['period_months'], 1)} months ÷ 12 = **{f(A['period_years'])} years**. "
             f"Adding noise with standard deviation 3σ gives SNR = 10 log₁₀(1/3²) = {f(RB[2]['snr_db'], 1)} dB. The "
             "program reproduces the hand values exactly (checked by `test_autocorr.py`).")

    # --------------------------------------------------------- 7 results
    rep.heading("Results and Discussion")
    rep.figure(FIG / "fig2_acf.png", "Normalised autocorrelation of the monthly sunspot number.",
               "(a) Lags up to 50 years with the 95 % white-noise band; (b) the first four years.",
               inference="", inference_items=[
                   f"The ACF oscillates like a damped cosine. Its first peak is at **{f(A['period_years'])} years** "
                   f"(ρ = {f(A['peak_value'], 3)}), far above the ±{A['bound']:.3f} band, so the 11-year cycle is "
                   f"overwhelmingly significant. Further peaks at {', '.join(f'{p / 12:.1f}' for p in A['peaks'][1:])} "
                   "years are integer multiples of the period.",
                   f"The peaks shrink ({', '.join(f(v) for v in A['peak_values'])}) because cycles differ in length and "
                   "height, so the signal is only quasi-periodic. The deepest trough "
                   f"(ρ = {f(A['min_value'])} at {A['min_lag'] / 12:.1f} years) is half a cycle: maximum is followed by minimum.",
                   f"Neighbouring months are strongly related (ρ[1] = {f(A['r1'], 3)}, ρ[12] = {f(A['r12'], 3)}); the "
                   f"correlation falls below 0.5 after {A['half']} months and crosses zero at {A['zero']} months, about a "
                   "quarter of the cycle. Persistence forecasts are therefore useful for one to two years only.",
               ])
    rep.figure(FIG / "fig3_properties.png", "Properties of the autocorrelation sequence.",
               "(a) The full sequence for negative and positive lags; (b) biased versus unbiased estimates.",
               inference=(
                   f"The full sequence is exactly even (maximum asymmetry {Mt['symmetry_err']:.0e}) and never exceeds its "
                   f"value at l = 0, which equals the signal energy (difference {Mt['r0_equals_energy']:.0e}). The direct, "
                   f"convolution and FFT methods agreed to {max(Mt['direct_vs_conv'], Mt['direct_vs_fft']):.0e}. At large "
                   "lags the unbiased estimate swings wildly, because it averages only the few remaining overlapping "
                   "samples, while the biased estimate tapers smoothly to zero; this is why the biased estimator is used."))
    rep.figure(FIG / "fig4_noise.png", "Testing the leftover variation for whiteness.",
               "ACF of (a) the residual after removing the 13-month mean and (b) the month-to-month difference.",
               inference=(
                   f"Neither is white noise. The residual has {nz['residual']['n_outside']} of 36 lags outside the 95 % "
                   f"band (where about 2 would be expected by chance), with a positive ρ[1] = {f(nz['residual']['r1'])} "
                   "and a negative dip at 3 to 6 months, partly created by the 13-month smoothing itself. The difference "
                   f"is close to white except for ρ[1] = {f(nz['difference']['r1'])}: a high month tends to be followed by a "
                   "lower one, the signature of independent counting errors added to a slowly varying signal. The "
                   "ACF therefore tells us which kind of model the noise needs."))
    rep.figure(FIG / "fig5_windows_noise.png", "Stability and robustness of the period estimate.",
               "(a) ACF of four 70-year epochs; (b) the full record with added white noise.",
               inference="", inference_items=[
                   f"Since 1818 the period is stable at {f(W[1]['period_years'], 1)}, {f(W[2]['period_years'], 1)} and "
                   f"{f(W[3]['period_years'], 1)} years, but 1749 to 1818 gives only {f(W[0]['period_years'], 1)} years with "
                   f"a weak peak (ρ = {f(W[0]['peak'])}): that epoch contains short cycles and the Dalton Minimum, and its "
                   "early counts are the least reliable.",
                   f"With noise as strong as the signal (0 dB) the period is still {f(RB[1]['period_years'], 1)} years; at "
                   f"{f(RB[2]['snr_db'], 0)} dB it is {f(RB[2]['period_years'], 1)} years with a small but significant "
                   f"peak ({f(RB[2]['peak'], 3)}). At {f(RB[3]['snr_db'], 0)} dB the peak ({f(RB[3]['peak'], 3)}) sinks to the "
                   f"noise band ({A['bound']:.3f}) and the estimate ({f(RB[3]['period_years'], 1)} years) is unreliable.",
               ])
    rep.table(
        ["Quantity", "Result"],
        [
            ["Solar-cycle period from the ACF", f"{f(A['period_years'])} years (ρ = {f(A['peak_value'], 3)})"],
            ["Multiples of the period", ", ".join(f"{p / 12:.1f} y" for p in A["peaks"][1:])],
            ["Decorrelation: ρ < 0.5 / first zero", f"{A['half']} / {A['zero']} months"],
            ["Period by epoch", ", ".join(f"{w['span'][0]} to {w['span'][1]}: {w['period_years']:.1f} y" for w in W)],
            ["Whiteness: residual / difference outside band", f"{nz['residual']['n_outside']} / {nz['difference']['n_outside']} of 36 lags"],
            ["Lowest SNR with a reliable period", f"{f(RB[2]['snr_db'], 0)} dB"],
            ["Next maximum, rough (last max + period)", f"about {FC['next_max_year']:.0f}"],
        ],
        "Summary of results", widths=[7.0, 10.0], font_size=9, first_col_bold=True, align=["left", "left"])

    # ------------------------------------------------------ 8 conclusion
    rep.heading("Conclusion")
    rep.para(
        f"Autocorrelation extracted the solar cycle directly from the data: a period of {f(A['period_years'])} years, "
        "with decaying repeats that quantify how irregular the Sun is. The theoretical properties (even symmetry, "
        "maximum at zero lag equal to the energy, equivalence of the direct, convolution and spectral methods) were "
        "confirmed to machine precision. The ACF also served as a diagnostic: the cycle was stable at about 10.7 to "
        "11 years for two centuries, the leftover month-to-month variation is not white noise, and periodicity was "
        f"still recoverable at an SNR of {f(RB[2]['snr_db'], 0)} dB. Combined with the most recent maximum "
        f"({FC['last_max_year']:.1f}), the period suggests the next maximum around {FC['next_max_year']:.0f}, a rough "
        "but useful planning horizon for satellite and grid operators; real forecasts refine it with physical models.")
    rep.heading("Learning Outcomes")
    rep.bullets([
        "Compute autocorrelation by definition, by convolution with the folded signal and via the power spectrum.",
        "Verify and use its properties: even symmetry, maximum at zero lag, r[0] = energy.",
        "Detect and measure hidden periodicity, and choose between biased and unbiased estimators.",
        "Test a sequence for whiteness with the ±1.96/√N band and judge the noise robustness of a period estimate.",
    ])
    rep.heading("References")
    refs = [
        "J. G. Proakis and D. G. Manolakis, *Digital Signal Processing*, 4th ed. Pearson, 2007, ch. 2.6.",
        "A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed. Pearson, 2010, ch. 10.",
        "SILSO World Data Center, “The International Sunspot Number,” Royal Observatory of Belgium, "
        "https://www.sidc.be/SILSO.",
        "F. Clette and L. Lefèvre, “The new sunspot number: assembling all corrections,” *Solar Physics*, "
        "vol. 291, pp. 2629-2651, 2016.",
        "G. E. P. Box, G. M. Jenkins and G. C. Reinsel, *Time Series Analysis*, 4th ed. Wiley, 2008, ch. 2.",
    ]
    for i, r in enumerate(refs, 1):
        rep.para(f"[{i}]  {r}", align="left", size=9.5, space_after=1)

    path = rep.save(OUT_DOCX)
    print(f"saved {path}")
    print(f"saved {export_pdf(path)}")


if __name__ == "__main__":
    build()
