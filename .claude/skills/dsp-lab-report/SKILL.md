---
name: dsp-lab-report
description: Generate a DSP lab practical (code + GUI + 8 to 12 page Symbiosis-header Word/PDF report) for a given experiment number, in the locked design of Experiment 1. Use when the user asks to make/generate/do "experiment N", "exp N", or the next DSP practical.
---

# DSP Lab Report: one experiment end-to-end

Produce `experiment_NN/` with modular Python code, a Tkinter GUI, figures, and
`DSP_Experiment_NN.docx` + `.pdf` in the **exact design locked in Experiment 1**.

**Reference implementation:** `experiment_01/`: read its files before starting and
mirror their structure, style and idioms. **Shared formatter:** `tools/report_kit.py`
(header, equations, captions, callouts, tables, code, PDF export): reuse it, don't fork it.
If you improve the kit, rebuild Experiment 1 afterwards to confirm nothing broke.

## Hard rules (from the user / professor)

- **No em dashes (—) or en dashes (–) anywhere:** report text, tables, captions, figure
  titles/labels, GUI text, code comments, and chat replies. Use commas, colons, parentheses,
  "to" for ranges ("1 to 7 Oct"), or a plain hyphen. The math minus sign (−) is fine.
  `report_kit.Report.save()` refuses to save a document containing either character; also
  grep the experiment's `.py` files for them before building.
- **Python only.** Never MATLAB; don't mention MATLAB anywhere.
- **Real-world dataset, small and quick:** < 1 MB, no login/API key, downloads in seconds,
  cached locally after the first run. Explain its source and real-life relevance.
- **Never use the classmate's datasets** (folder `Downloads/24070123035`): NASA Mill
  (`mill.csv`), MIT-BIH ECG (`mit_bih.csv`), his `sensor_data*.csv`, or "user-defined" data.
  Prefer a different dataset from those already used by earlier experiments in `dsp/`.
- **Length 8 to 12 pages.** 43 pages was rejected as far too long. Check the page count every build.
- Symbiosis header on every page (handled by `report_kit`).
- Every graph: title + caption description + **Inference** box. Good theory, dataset
  description + snippet, calculations, detailed results, conclusion, learning outcomes,
  real-life application.
- Student: Name `Dewashish Lambore`; PRN and Date are `____` placeholders unless the user
  has given them (check memory / ask once at the end, don't block on it).
- No co-author/Claude signatures if committing (user preference).

## Experiment list (from the course)

| No. | Topic |
|---|---|
| 1 | Basic operations on DT signals (**done**: Pune ERA5 weather/solar via Open-Meteo) |
| 2 | Check whether a system is LTI; time response of LTI systems (**done**: Pune CAMS PM2.5/NO₂ around Diwali 2025 via Open-Meteo air-quality API) |
| 3 | Moving-average filter; sampling and reconstruction (**done**: ESC-50 bird-song clip 1-34495-A-14 from GitHub raw) |
| 4 | Circular convolution (**done**: Mumbai airport VABB METAR winds 2024 from Iowa Environmental Mesonet; wind rose + ICAO runway usability) |
| 5 | Autocorrelation (**done**: SILSO monthly sunspot number 1749 to 2026; solar-cycle period 10.65 y) |
| 6 | Cross-correlation (**done**: NOAA CO-OPS hourly tide gauges, 5 Chesapeake Bay stations, Jan 2024; tidal delay and wave speed) |
| 7 | Difference-equation-based smoothing (**done**: FRED DEXINUS USD/INR daily 2019 to 2025; MA, EMA, double EMA, Holt, MACD) |
| 8 | DFT and IDFT (**done**: NOAA GML Mauna Loa monthly CO₂, 67 whole years; seasonal lines, leakage, IDFT deseasonalization vs NOAA) |
| 9 | Linear convolution using the DFT (**done**: Rostock communication-acoustics CC0 files: dry speech xmas.wav * seminar-room RIR rir_clap.wav vs published xmas_clap.wav; reverb, OLA, RT60, deconvolution) |
| 10 | Circular convolution using the DFT (**done**: NOAA GHCN-Daily New Delhi Safdarjung TMAX/TMIN 1991 to 2020 via NCEI data service; circular smoothing of daily normals, cyclic prefix / OFDM) |
| 11 | FIR filter design by windowing + difference equation, frequency response (**done**: PhysioNet BIDMC record bidmc01 PPG, WFDB format 16 read directly; HR and RR vs monitor / annotations) |
| 12 | IIR filter design + difference equation, frequency response (**done**: PhysioNet eegmmidb S001 R01/R02 EEG channel O1 via AWS mirror physionet-open.s3.amazonaws.com; notch, bilinear Butterworth/Chebyshev/elliptic, Berger effect) |

Dataset ideas (verify size/availability before committing to one): Open-Meteo air-quality
API (PM2.5/PM10, Indian cities); SILSO monthly sunspot number CSV (≈11-year cycle → autocorr);
NOAA Mauna Loa monthly CO₂ CSV (seasonal line in the DFT); NOAA CO-OPS tide predictions at two
stations (cross-correlation lag); short WAV clips from ESC-50 on GitHub raw (audio for
sampling, FIR/IIR, reverb = linear convolution with a room impulse response); USGS earthquake
feed or an IRIS waveform snippet (IIR band-pass). Pick what makes the operation's
real-world meaning obvious, like "direct + diffuse = GHI" did for addition.

## Workflow

1. **Plan (briefly):** topic → dataset → 4 to 6 real-world analyses where each DSP operation
   yields a physical, quantitative result (error, lag, frequency, kWh, …). Fetch a sample
   of the data first and confirm it shows the effect clearly.
2. **Scaffold `experiment_NN/`** mirroring Exp 1:
   - `config.py`: every constant/parameter/path; dataset URL + period.
   - `data_loader.py`: download once → cache CSV/WAV in `data/`; clean; attach `n`.
   - `signal_ops.py` (or a topic-named core module): pure functions, no I/O, reuse the
     `DTSignal(n, x)` idea when index support matters; registry for the GUI.
   - `analysis.py`: physical interpretation; returns signals + scalar results.
   - `plots.py`: copy Exp 1's style block and helpers (`C` palette, `apply_style`, `stem`,
     `day_guides`, `top_legend`); `make_all_figures(df, results)` returns {name: fig}.
   - `gui.py`: Tkinter + embedded Matplotlib, same layout as Exp 1 (controls left, plot +
     stats table right, sliders per parameter, save/export buttons, `--screenshot` flag).
   - `run_experiment.py` (UTF-8 stdout, saves figures + `results.json`), `test_*.py`
     (unittest, verifies the theory's identities), `build_report.py`.
3. **Run:** tests → `python run_experiment.py` → GUI screenshot via `python gui.py ... --screenshot outputs/figures/figN_gui.png` → `python build_report.py`.
4. **Verify visually:** render the PDF with PyMuPDF (`fitz`) into the scratchpad, contact-sheet
   the pages, and *look* at them: legends vs titles, clipped data, tables split by one row,
   orphaned inference boxes, page count 8 to 12. Fix and rebuild until clean.
5. **Report back** to the user: dataset + why, key numbers, page count, file paths, and what
   to review. Update the project memory (`project_dsp_lab.md`) with which experiment is done
   and which dataset it used.

## Report structure and budget (≈10 pages total)

| § | Section | Budget |
|---|---|---|
|: | `info_block` (Course, Exp No., Name, PRN, Language, Date), `title` + subtitle, `abstract` with real numbers | ½ p |
| 1 | Aim and Objectives (aim paragraph + 3 bullets) | ¼ p |
| 2 | Real-World Application (paragraph + Table: operation / definition / use here / industry) | ¾ p |
| 3 | Theory: definitions, **≤ 6 numbered equations**, one summary table, one `note` | 1½ p |
| 4 | Dataset Description: source paragraph, attributes+statistics table, 6-row preview table, raw-signal figure + inference | 1½ p |
| 5 | Methodology and Implementation: module table, **one code excerpt ≤ 25 lines**, GUI paragraph + 1 screenshot (width ≈ 14.5 cm) | 1½ p |
| 6 | Calculations: one hand-calculation table on 3 to 5 real samples + one worked paragraph; state agreement with program | ¾ p |
| 7 | Results and Discussion: **3 to 5 figures** (multi-panel, full width), each with caption + inference (bullets per panel when multi-panel), then one summary-of-results table | 3 p |
| 8 to 10 | Conclusion (1 paragraph), Learning Outcomes (4 bullets), References (≤ 6, IEEE style) | ¾ p |

Totals: **≤ 6 to 7 figures, ≤ 8 tables**. No full source appendix (code lives in the folder).

## report_kit usage (see `experiment_01/build_report.py`)

- `Report(footer_left=...)`; `heading(text, level)` auto-numbers; `para` supports
  `**bold**`, `*italic*`, `` `code` ``, `$inline LaTeX$`; `equation(latex)` → native Word math,
  numbered; `table(header, rows, title, widths=[cm…] summing to 17, font_size, align, first_col_bold)`;
  `figure(path, title, desc, width_cm, inference=..., inference_items=[...])`; `note(...)`;
  `code(src, title, font_size=7.5)`; `bullets([...])`; `save()` then `export_pdf()` (uses Word).
- **Pull every number from the analysis results at build time**: never hard-code values in prose.
- Keep cross-references (Table N / Figure N / Eq. N) consistent with actual order; recount
  after restructuring.
- Avoid literal `$` in prose (it starts inline math) and a lone `*` such as "X*[k]" or "x * h" (it starts italics); write "complex conjugate of X[k]" or put the expression in `$...$`.

## Plot rules (dataviz-validated palette, already in Exp 1 `plots.py`)

- Width `FULL_W = 6.7` in, DPI 220, Segoe UI 7.5 pt; titles left-aligned "(a) …";
  legend on the title row via `top_legend`: shorten titles so they don't collide; in
  half-width panels put the legend inside the axes with headroom (raise `ylim`).
- Fixed colour order blue, orange, aqua, violet; text in ink colours; one y-axis per panel;
  stem plots for short sequences (≤ 30 samples), lines for long ones; offset overlapping stems.
- Figure heights ≈ 2.2 to 5 in so 1 to 2 figures + inferences fit a page. The usable page height is ~23 cm: a figure plus caption plus inference must be ≤ ~11.5 cm for two to share a page. Check `17 * h / w` of each PNG; place wide two-panel figures with `width_cm=15.5` when a pair just misses.

## Gotchas already hit (don't repeat)

- GUI screenshots: use the app's `capture()` (Win32 `PrintWindow`): **never** `ImageGrab`
  of a screen region (it captured the user's browser once). Keep the window ≤ 1280×700.
- Windows console is cp1252: `sys.stdout.reconfigure(encoding="utf-8")` in entry points.
- Don't bulk-edit Python with multi-line regexes: they swallowed code across lines; use
  exact-string edits and syntax-check (`ast.parse`) afterwards.
- `figure(..., inference="", inference_items=[...])` is supported (bullets-only inference).
- Short tables and code (≤ 15 rows / ≤ 40 lines) are kept on one page by the kit; long
  ones split with a repeated header row.
- Word drops MathML `<mspace>`; the kit converts `\quad`/`\,` to real spaces already.
- latex2mathml does not know `rgmax` / `rgmin`: write `\mathrm{argmax}`.
- Every physical explanation in an inference must be backed by a number from the data; if it is not, compute it or drop it.
- Anything non-deterministic (timings, random noise) must be drawn and quoted from the SAME run: have `build_report.py` regenerate the figures from its own results before building.
- Work out hand-calculation values by actually computing them (or from the code) before writing them into tests or text; never guess them.
- IRIS irisws/timeseries is retired; FDSN dataselect returns miniSEED (needs ObsPy, not installed). PhysioNet is slow but its open-data mirror https://physionet-open.s3.amazonaws.com/<db>/<ver>/... is fast; WFDB format-16 .dat files are plain interleaved int16.
- FRED (fred.stlouisfed.org) drops Python urllib requests unless the User-Agent is `curl/8.4.0`.
- Never put `\n` inside strings in a Bash heredoc that writes Python: it arrives as a real newline. Use the Edit tool for such lines.
- Temp-file cleanup in Bash: use `rm -f "${S:?}"/*.png`, never `rm -f "$S"/*` (blocked by the safety check).
