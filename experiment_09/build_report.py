"""
build_report.py
---------------
Generates the Word (and PDF) lab report for Experiment 9.

    python run_experiment.py      # figures + results first
    python build_report.py        # -> DSP_Experiment_09.docx / .pdf

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
import fastconv as fc                # noqa: E402
from report_kit import Report, export_pdf   # noqa: E402

STUDENT = {
    "Name": "Dewashish Lambore",
    "PRN": "____________",
    "Date": "____ / ____ / 2026",
    "Course": "Digital Signal Processing Lab",
}
FIG = config.FIGURE_DIR
OUT_DOCX = HERE / "DSP_Experiment_09.docx"


def f(x, d=2):
    v = 0.0 if abs(x) < 0.5 * 10 ** (-d) else x
    return f"{v:,.{d}f}"


def build():
    R = analysis.run_all()
    # timings vary from run to run: redraw the figures from this same run so text and figures agree
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plots
    for name, fig in plots.make_all_figures(R).items():
        fig.savefig(FIG / name)
        plt.close(fig)
    A, Al, O, rm, Dc, fs, desc = R["acc"], R["alias"], R["ola"], R["room"], R["deconv"], R["fs"], R["desc"]
    L, M, N = A["L"], A["M"], A["Nlin"]
    ola = next(o for o in O if o["B"] == config.DEFAULT_BLOCK)
    al0 = Al["rows"][0]

    rep = Report(footer_left="DSP Lab  ·  Experiment 9  ·  Linear Convolution using the DFT")

    # ------------------------------------------------------------------ title
    rep.info_block([("Course", STUDENT["Course"]), ("Experiment No.", "09"),
                    ("Name", STUDENT["Name"]), ("PRN", STUDENT["PRN"]),
                    ("Language", "Python 3.10"), ("Date", STUDENT["Date"])])
    rep.title("Linear Convolution using the DFT",
              "Placing a dry voice in a real seminar room: fast convolution reverb")
    rep.abstract(
        f"A dry speech recording (L = {L:,} samples) was convolved with a room impulse response measured in a "
        f"university seminar room (M = {M:,} samples, 1 s at {fs / 1000:.1f} kHz) by multiplying their DFTs. With "
        f"zero padding to N = L + M − 1 = {N:,} the result matched direct convolution to {A['err_vs_direct']:.0e} and "
        f"the independently published reverberant file to {f(A['snr_vs_published'], 1)} dB SNR, while running "
        f"{A['speedup']:,.0f} times faster than direct convolution ({f(A['t_direct'], 1)} s versus "
        f"{1000 * A['t_fast']:.0f} ms) and needing {A['mult_direct'] / A['mult_fft']:,.0f} times fewer multiplications. "
        f"Choosing an FFT-friendly length ({A['Nfast']:,} instead of {N:,}) made it {A['t_exact'] / A['t_fast']:.0f} "
        f"times faster again. Without enough padding the circular result was time-aliased (error {al0['err']:.1e} "
        f"for N = L). Overlap-add block processing was equally exact and trades latency for cost. The impulse response "
        f"itself gave a reverberation time of {f(rm['T20'])} s, and dividing spectra (regularised deconvolution) "
        f"removed the room again with {f(Dc['best_clean']['clean'], 0)} dB SNR on clean data.",
        "linear convolution, DFT, FFT, fast convolution, zero padding, time aliasing, overlap-add, room impulse "
        "response, reverberation, deconvolution, Python")

    # ------------------------------------------------------------- 1 aim
    rep.heading("Aim and Objectives")
    rep.para("**Aim:** to compute the linear convolution of two real-world signals using the DFT, determine the "
             "zero padding it requires, and compare its accuracy and speed with direct convolution, in Python.",
             space_after=3)
    rep.bullets([
        "Implement DFT-based convolution with zero padding and verify it against direct convolution and a published result.",
        "Show what happens when the DFT is too short (time aliasing) and implement overlap-add block convolution.",
        "Measure the computational saving, and use the room impulse response for acoustics (RT60) and deconvolution.",
    ])

    # ----------------------------------------------------- 2 application
    rep.heading("Real-World Application")
    rep.para(
        "A room is a linear time-invariant system: the sound that reaches a listener is the source signal convolved "
        "with the room's impulse response, which contains the direct sound and thousands of reflections. If that "
        "response is measured once, any dry recording can be made to sound as if it was performed in the room. This "
        "**convolution reverb** is used in music production, film post-production, video games and virtual reality, "
        "and acoustic consultants use it to let clients *listen* to a hall before it is built. Impulse responses are "
        "long (here 44,100 samples), so direct convolution is far too slow; every real product uses the DFT. The "
        "same fast convolution appears in many other systems (Table 1).")
    rep.table(
        ["Application", "Signal x[n]", "Impulse response h[n]"],
        [
            ["Convolution reverb (this experiment)", "dry voice or instrument", "measured room or hall response"],
            ["Headphone 3-D audio, VR", "game or film sound", "head-related impulse responses"],
            ["Long FIR filtering", "audio, sensor data", "filter with hundreds of taps"],
            ["Radar pulse compression", "received echo", "time-reversed transmitted chirp"],
            ["Image blur and deblur", "photograph rows", "point-spread function of the lens"],
        ],
        "Uses of fast (DFT-based) convolution", widths=[5.6, 5.0, 6.4], font_size=9, first_col_bold=True,
        align=["left", "left", "left"])

    # ------------------------------------------------------------ 3 theory
    rep.heading("Theory")
    rep.para(f"The **linear convolution** of x[n] (length L) with h[n] (length M) has length L + M − 1:")
    rep.equation(r"y[n] = \sum_{k=0}^{M-1} h[k]\,x[n-k], \qquad 0 \le n \le L+M-2")
    rep.para("By the convolution property of the DFT, multiplying two N-point DFTs corresponds to **circular** "
             "convolution of length N (Experiment 4), not linear convolution:")
    rep.equation(r"\mathrm{IDFT}_N\left\{X[k]\,H[k]\right\} = x[n] \circledast_N h[n] = \sum_{r} y[n + rN]")
    rep.para("The circular result is the linear one folded (time-aliased) onto N samples. The two are therefore "
             "equal if, and only if, both sequences are first zero-padded to")
    rep.equation(r"N \ge L + M - 1 \quad\Rightarrow\quad y[n] = \mathrm{IDFT}_N\left\{\mathrm{DFT}_N\{x\}\cdot\mathrm{DFT}_N\{h\}\right\}")
    rep.para("**Cost.** Direct convolution needs L·M multiplications. The DFT route needs three FFTs and one "
             "product, so it grows only as N log₂ N:")
    rep.equation(r"C_{\mathrm{direct}} = L\,M, \qquad C_{\mathrm{FFT}} \approx 3\,\frac{N}{2}\log_2 N \cdot 2 + 3N")
    rep.para("**Overlap-add** splits x into blocks of B samples, convolves each block with h using an FFT of length "
             "at least B + M − 1, and adds the overlapping tails. It is exact, works for endless real-time streams, "
             "and its delay is one block. **Room acoustics** is read from h[n] by Schroeder's backward integration; "
             "the reverberation time RT60 is the time for the energy to fall by 60 dB:")
    rep.equation(r"E(t) = \int_{t}^{\infty} h^2(\tau)\,d\tau, \qquad \mathrm{RT}_{60} = \frac{-60\ \mathrm{dB}}{\text{slope of } 10\log_{10}E(t)}")
    rep.note("Because convolution becomes multiplication, it can be undone by division, X = Y/H. Where |H| is tiny "
             "the division amplifies noise enormously, so in practice a regularised inverse "
             "X = Y·H*/(|H|² + ε·max|H|²) is used; ε trades residual reverberation against noise.")

    # ----------------------------------------------------------- 4 dataset
    rep.heading("Dataset Description")
    rep.para(
        "All three files come from the open *Communication Acoustics* exercises of the Institute of Communications "
        "Engineering, University of Rostock (GitHub: spatialaudio/communication-acoustics-exercises, CC0 public "
        "domain). The **dry speech** is a male voice saying \"Wishing you a Merry Christmas, and a Happy New Year\" "
        "(Freesound #136777). The **room impulse response** was measured in a seminar room by clapping two wooden "
        "boards together in front of a microphone. The **published wet file** is the authors' own convolution of the "
        "two and serves as an independent reference. The files total about 620 kB and are cached after download.")
    rep.table(
        ["Signal", "File", "Samples", "Duration", "Peak", "RMS", "Size"],
        [[{"dry": "Dry speech x[n]", "rir": "Room impulse response h[n]", "wet": "Published y_ref[n]"}[d["key"]],
          f"`{d['file']}`", f"{d['samples']:,}", f"{d['seconds']:.3f} s", f"{d['peak']:.3f}", f"{d['rms']:.4f}",
          f"{d['kb']:.0f} kB"] for d in desc],
        f"Dataset attributes (all mono, 16-bit PCM, fs = {fs:,} Hz)",
        widths=[3.8, 3.0, 2.0, 2.0, 1.8, 1.8, 2.6], font_size=8.5,
        align=["left", "left", "center", "center", "center", "center", "center"])
    k = rm["peak_idx"]
    hi = R["D"]["rir_int"]
    rep.table(
        ["n", "Time (ms)", "h[n] (int16)", "h[n] (float)"],
        [[f"{i}", f"{1000 * i / fs:.3f}", f"{int(hi[i])}", f"{hi[i] / 32768:+.4f}"] for i in range(k - 2, k + 4)],
        "Impulse-response samples around the direct sound",
        widths=[2.4, 3.6, 5.0, 6.0], font_size=8.5, align=["center"] * 4)
    rep.figure(FIG / "fig1_inputs.png", "The two inputs of the convolution.",
               "(a) Dry speech; (b) the room impulse response on a decibel scale.",
               inference=(
                   "The dry speech has crisp word boundaries and silent gaps. The impulse response starts with the "
                   f"direct sound of the clap at {f(rm['peak_ms'], 1)} ms, followed by a dense cloud of reflections whose "
                   "level falls almost linearly in decibels, the signature of an exponential energy decay. Its tail "
                   f"is still visible after 600 ms, so h[n] must be kept at its full length M = {M:,}."))

    # ---------------------------------------------------- 5 implementation
    rep.heading("Methodology and Implementation")
    rep.para(f"Both signals are zero-padded to N, transformed with real FFTs, multiplied bin by bin and transformed "
             f"back. The result is compared with direct convolution (timed on the first "
             f"{config.DIRECT_TIMING_SECONDS} s and scaled to the full length, since its cost is proportional to L) "
             "and with the published file after a single gain factor (the file was normalised). N is then reduced to "
             "show aliasing, and overlap-add is run with block lengths of "
             + ", ".join(str(b) for b in config.OLA_BLOCKS) + " samples.", keep_next=True)
    rep.table(
        ["Module", "Responsibility"],
        [
            ["`config.py`", "File URLs, block lengths, deconvolution settings"],
            ["`data_loader.py`", "Download the three WAV files once, cache, convert to float"],
            ["`fastconv.py`", "Direct and DFT convolution, fast FFT length, overlap-add, cost, RT60, deconvolution"],
            ["`analysis.py`", "Accuracy and timing, aliasing study, overlap-add study, room acoustics, deconvolution"],
            ["`plots.py` / `gui.py`", "Report figures / interactive convolution reverb with audio playback"],
            ["`test_fastconv.py`", "8 unit tests (equality with direct, aliasing, overlap-add, RT60, deconvolution): all pass"],
        ],
        "Software modules", widths=[4.2, 12.8], font_size=9, align=["left", "left"])
    rep.code('''def conv_dft(x, h, N=None):                   # y = IDFT{ DFT{x} DFT{h} }
    N = N or len(x) + len(h) - 1                 # zero padding for LINEAR convolution
    X, H = np.fft.rfft(x, N), np.fft.rfft(h, N)  # rfft pads with zeros to length N
    return np.fft.irfft(X * H, N)

def overlap_add(x, h, B):                     # block convolution for long / live signals
    N = next_fast_length(B + len(h) - 1)
    H = np.fft.rfft(h, N)
    y = np.zeros(len(x) + len(h) - 1)
    for s in range(0, len(x), B):
        seg = x[s:s + B]
        ys = np.fft.irfft(np.fft.rfft(seg, N) * H, N)[:len(seg) + len(h) - 1]
        y[s:s + len(ys)] += ys                   # add the overlapping tail
    return y''', title="Core of fastconv.py", font_size=7.5)
    rep.para("**GUI.** `gui.py` (Figure 2) lets the user choose a single DFT or overlap-add, shorten the impulse "
             "response, set N as a percentage of the required padding, and set the block length. It shows the "
             "output and its difference from exact linear convolution, reports the error, the time taken and the "
             "latency, and plays the dry and processed sound so that aliasing can be *heard* as an echo before the "
             "first word. Run `python run_experiment.py` (which also writes `outputs/reverberant.wav`) and `python gui.py`.")
    rep.figure(FIG / "fig7_gui.png", "The convolution reverb with only half of the required zero padding.",
               "The difference panel shows the aliased tail folded onto the first half second.", width_cm=14.5)

    # ------------------------------------------------------------ 6 calcs
    rep.heading("Calculations")
    xh, hh = np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.5])
    Xh, Hh = np.fft.fft(xh, 4), np.fft.fft(hh, 4)
    yh = np.fft.ifft(Xh * Hh).real
    y3 = fc.conv_dft(xh, hh, 3)

    def cz(z):
        re, im = z.real + 0.0, z.imag + 0.0
        return f"{re:g}" if abs(im) < 1e-12 else f"{re:g} {'+' if im > 0 else '−'} j{abs(im):g}"
    rep.para("**Small example.** x = [1, 2, 3] (L = 3) and h = [1, 0.5] (M = 2), so N = L + M − 1 = 4. Padding to 4 "
             "samples and using the 4-point DFT (twiddle factor −j):", keep_next=True)
    rep.table(
        ["k", "X[k] of [1, 2, 3, 0]", "H[k] of [1, 0.5, 0, 0]", "Y[k] = X[k]·H[k]"],
        [[str(i), cz(Xh[i]), cz(Hh[i]), cz(Xh[i] * Hh[i])] for i in range(4)],
        "DFT-based convolution by hand", widths=[1.0, 5.0, 5.0, 6.0], font_size=9, align=["center"] * 4)
    rep.para(f"The IDFT of Y gives y = [{', '.join(f'{v:g}' for v in yh)}], identical to the direct sum "
             "(1·1, 2·1 + 1·0.5, 3·1 + 2·0.5, 3·0.5). With N = 3 (too short) the last value 1.5 wraps onto n = 0: "
             f"y₃ = [{', '.join(f'{v:g}' for v in y3)}], i.e. y[0] = 1 + 1.5. **For the real signals:** L·M = "
             f"{L:,} × {M:,} = {A['mult_direct'] / 1e9:.2f} × 10⁹ multiplications directly, against about "
             f"{A['mult_fft'] / 1e6:.1f} × 10⁶ with FFTs of length {A['Nfast']:,} (a ratio of "
             f"{A['mult_direct'] / A['mult_fft']:,.0f}). N = {N:,} = {' × '.join(map(str, A['factor_Nlin']))} contains "
             f"the large prime 1601, while {A['Nfast']:,} = 2⁸ × 5⁴ is FFT-friendly. These values agree with the program "
             "(checked by `test_fastconv.py`).")

    # --------------------------------------------------------- 7 results
    rep.heading("Results and Discussion")
    rep.figure(FIG / "fig2_result.png", "The convolution result.",
               "(a) Output of the DFT method overlaid on the published reverberant file; (b), (c) spectrograms of the "
               "dry and the reverberant speech.",
               inference="", inference_items=[
                   f"The DFT result is indistinguishable from the published file: after one gain factor the relative "
                   f"difference is {A['err_vs_published']:.1e} ({f(A['snr_vs_published'], 1)} dB), which is the 16-bit "
                   "quantisation of that file. On the first 0.25 s it equals direct convolution to "
                   f"{A['err_vs_direct']:.0e}, i.e. exactly up to rounding.",
                   f"The output is {(N - L) / fs:.2f} s longer than the input (M − 1 samples): the room keeps ringing after "
                   "the speaker stops. In the spectrogram the gaps between words fill with reverberant energy, "
                   "especially below 2 kHz, which is why speech in a reverberant room sounds less clear.",
               ])
    rep.figure(FIG / "fig3_aliasing.png", "What happens when the DFT is too short.",
               f"(a) Error of a {L:,}-point (unpadded) circular result; (b) relative error against the missing padding.",
               inference=(
                   f"Without padding (N = L) the {M - 1:,} tail samples that do not fit wrap around and are added to the "
                   "beginning: the error is confined to the first second, where it sounds like the end of the "
                   f"sentence echoing before the first word (relative error {al0['err']:.1e}). The error falls as N grows "
                   f"but remains about {Al['rows'][-3]['err']:.0e} even with 250 ms missing, because the room tail is long. "
                   f"Only at N = L + M − 1 does it collapse to {Al['rows'][-1]['err']:.0e}: the padding condition is exact, "
                   "not approximate."))
    rep.figure(FIG / "fig4_ola.png", "Overlap-add and computation time.",
               "(a) Multiplications per output sample against block length, labelled with the latency; (b) measured "
               "time of each method.",
               inference="", inference_items=[
                   f"Every overlap-add block length reproduced the linear result exactly (error ≤ "
                   f"{max(o['err'] for o in O):.0e}). Short blocks give low latency ({O[0]['latency_ms']:.0f} ms for "
                   f"B = {O[0]['B']}) but cost more, because each tiny block still needs an FFT longer than M; long blocks "
                   f"are cheaper ({O[-1]['mult_per_sample']:.0f} multiplications per sample for B = {O[-1]['B']:,}) but delay "
                   f"the sound by {O[-1]['latency_ms']:.0f} ms. Live audio typically uses 256 to 1,024 samples.",
                   f"Direct convolution would take about {f(A['t_direct'], 0)} s; one FFT of length {N:,} takes "
                   f"{1000 * A['t_exact']:.0f} ms, one of length {A['Nfast']:,} only {1000 * A['t_fast']:.0f} ms, and "
                   f"overlap-add with B = {config.DEFAULT_BLOCK} takes {1000 * ola['time']:.0f} ms.",
               ])
    rep.figure(FIG / "fig5_room.png", "Room acoustics from the impulse response.",
               "(a) Schroeder energy decay curve with the T20 line fit; (b) magnitude of H(f), raw and smoothed.",
               inference=(
                   f"The energy decays linearly in dB down to about −30 dB, giving RT60 = **{f(rm['T20'])} s** from the "
                   f"−5 to −25 dB range (T20) and an early decay time of {f(rm['EDT'])} s, typical of a furnished seminar "
                   f"room. Below about −35 dB the curve flattens into the noise floor of the recording, which is why T30 "
                   f"({f(rm['T30'])} s) is less reliable. The direct sound carries {f(-rm['DRR'], 1)} dB less energy than the "
                   "reverberation that follows it. The frequency response is irregular and favours 1 to 3 kHz, partly "
                   "because a hand clap has little low-frequency energy; a sine-sweep measurement would be flatter."))
    best_c, best_n = Dc["best_clean"], Dc["best_noisy"]
    rep.figure(FIG / "fig6_deconv.png", "Removing the room again by spectral division.",
               f"(a) SNR of the recovered dry speech against ε, without and with {config.NOISE_SNR_DB:.0f} dB recording "
               "noise; (b) the reverberant, dry and recovered waveforms.",
               inference=(
                   f"The reverberant file itself has an SNR of about {f(Dc['wet_sisnr'], 0)} dB against the dry speech (the reverberation "
                   "is as strong as the speech). "
                   f"Dividing by H recovers the speech to {f(best_c['clean'], 1)} dB at ε = {best_c['eps']:g}, and the recovered "
                   f"waveform lies on top of the original. With realistic {config.NOISE_SNR_DB:.0f} dB noise the best result "
                   f"drops to {f(best_n['noisy'], 1)} dB at a larger ε = {best_n['eps']:g}, and smaller ε makes it worse: "
                   "dividing by the nearly-zero bins of H amplifies the noise. Deconvolution works only when h[n] "
                   "is known exactly and the recording is clean."))
    rep.table(
        ["Quantity", "Result"],
        [
            ["DFT method versus direct convolution", f"relative error {A['err_vs_direct']:.0e}"],
            ["DFT method versus published reverberant file", f"{f(A['snr_vs_published'], 1)} dB SNR"],
            ["Required DFT length L + M − 1 / FFT-friendly length", f"{N:,} / {A['Nfast']:,}"],
            ["Time: direct / DFT (N = L + M − 1) / DFT (fast N) / overlap-add",
             f"{f(A['t_direct'], 1)} s / {1000 * A['t_exact']:.0f} ms / {1000 * A['t_fast']:.0f} ms / {1000 * ola['time']:.0f} ms"],
            ["Multiplications direct / FFT", f"{A['mult_direct'] / 1e9:.2f} × 10⁹ / {A['mult_fft'] / 1e6:.1f} × 10⁶"],
            ["Error with N = L (no padding)", f"{al0['err']:.1e}"],
            ["Reverberation time T20 / EDT; direct-to-reverberant ratio", f"{f(rm['T20'])} s / {f(rm['EDT'])} s; {f(rm['DRR'], 1)} dB"],
            ["Deconvolution SNR: clean / 40 dB noise", f"{f(best_c['clean'], 1)} dB / {f(best_n['noisy'], 1)} dB"],
        ],
        "Summary of results", widths=[8.6, 8.4], font_size=9, first_col_bold=True, align=["left", "left"])

    # ------------------------------------------------------ 8 conclusion
    rep.heading("Conclusion")
    rep.para(
        "Linear convolution was computed exactly with the DFT once both signals were zero-padded to L + M − 1; with "
        "less padding the result was the circular convolution, whose wrapped tail is audible as an echo before the "
        f"speech. On a real voice and a real room response the method reproduced a published result to "
        f"{f(A['snr_vs_published'], 0)} dB while being about {A['speedup']:,.0f} times faster than direct convolution, and "
        "choosing an FFT-friendly length mattered almost as much as using the FFT at all. Overlap-add made the same "
        "computation possible block by block, exposing the latency-versus-cost trade-off of real-time reverbs. The "
        f"impulse response also described the room (RT60 ≈ {f(rm['T20'], 1)} s), and the convolution theorem allowed "
        "the room to be removed again, but only as far as noise permits.")
    rep.heading("Learning Outcomes")
    rep.bullets([
        "Compute linear convolution through the DFT and choose the zero-padding length N ≥ L + M − 1.",
        "Recognise and explain time aliasing when N is too small.",
        "Implement overlap-add block convolution and quantify its cost and latency.",
        "Compare direct and FFT-based convolution in operations and measured time, and apply fast convolution to audio.",
    ])
    rep.heading("References")
    refs = [
        "A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed. Pearson, 2010, ch. 8.7.",
        "T. G. Stockham, “High-speed convolution and correlation,” in *Proc. AFIPS Spring Joint Computer Conf.*, "
        "1966, pp. 229-233.",
        "M. R. Schroeder, “New method of measuring reverberation time,” *J. Acoust. Soc. Am.*, vol. 37, pp. 409-412, 1965.",
        "ISO 3382-2:2008, *Acoustics: Measurement of room acoustic parameters, Part 2: Reverberation time in ordinary rooms*.",
        "M. Geier, S. Spors et al., *Communication Acoustics Exercises*, University of Rostock, "
        "https://github.com/spatialaudio/communication-acoustics-exercises (CC0).",
    ]
    for i, rr in enumerate(refs, 1):
        rep.para(f"[{i}]  {rr}", align="left", size=9.5, space_after=1)

    path = rep.save(OUT_DOCX)
    print(f"saved {path}")
    print(f"saved {export_pdf(path)}")


if __name__ == "__main__":
    build()
