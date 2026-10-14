"""
gui.py
------
Interactive "FIR filter designer" for Experiment 11.

    python gui.py

Choose low-pass or band-pass, the cut-off frequencies, the number of taps and
the window (or a Kaiser design from attenuation and transition width).  The
GUI shows the impulse response, the magnitude response, the filtered PPG and
the heart or breathing rate estimated from it, with the error against the
hospital reference.  All processing is delegated to ``fir`` and ``analysis``.
"""

import argparse
import tkinter as tk
from tkinter import filedialog, ttk

import matplotlib

matplotlib.use("TkAgg")
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy.signal import find_peaks

import analysis
import config
import data_loader
import fir
import plots

FS = config.FS
WINS = config.WINDOWS + ["kaiser"]


class FIRDesignerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DSP Lab · Experiment 11 · FIR Filter Design by Windowing")
        self.geometry("1280x700+10+10")
        self.minsize(1100, 640)
        self.configure(bg="#f6f6f3")
        self.D = data_loader.load()
        self.x = self.D["ppg"] - self.D["ppg"].mean()
        self._build_style()
        self._build_controls()
        self._build_plot_area()
        self.update_plot()

    def _build_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background="#f6f6f3", font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI Semibold", 13))
        style.configure("Section.TLabel", font=("Segoe UI Semibold", 10), foreground="#52514e")
        style.configure("Big.TLabel", font=("Segoe UI Semibold", 10), foreground="#2a78d6",
                        background="#ffffff", padding=(8, 6))

    def _scale(self, parent, label, var, lo, hi, res):
        ttk.Label(parent, text=label).pack(anchor="w", pady=(2, 0))
        s = tk.Scale(parent, variable=var, orient=tk.HORIZONTAL, length=270, from_=lo, to=hi, resolution=res,
                     bg="#f6f6f3", highlightthickness=0, troughcolor="#e1e0d9", font=("Segoe UI", 8))
        s.bind("<ButtonRelease-1>", lambda e: self.update_plot())
        s.pack(anchor="w")

    def _build_controls(self):
        panel = ttk.Frame(self, padding=(16, 10))
        panel.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(panel, text="FIR Filter Designer", style="Header.TLabel").pack(anchor="w")
        ttk.Label(panel, text="BIDMC PPG, 8 min at 125 Hz", foreground="#898781").pack(anchor="w", pady=(0, 6))
        ttk.Label(panel, text="TASK", style="Section.TLabel").pack(anchor="w")
        self.task = tk.StringVar(value="pulse")
        for v, t in (("pulse", "Heart rate: band-pass"), ("resp", "Breathing rate: low-pass")):
            ttk.Radiobutton(panel, text=t, value=v, variable=self.task, command=self._task_defaults).pack(anchor="w")
        self.f1, self.f2 = tk.DoubleVar(value=config.PULSE_BAND[0]), tk.DoubleVar(value=config.PULSE_BAND[1])
        self._scale(panel, "Lower cut-off (Hz, band-pass only)", self.f1, 0.1, 2.0, 0.05)
        self._scale(panel, "Upper cut-off (Hz)", self.f2, 0.2, 10.0, 0.05)
        self.N = tk.IntVar(value=config.N_PULSE)
        self._scale(panel, "Taps N (odd)", self.N, 11, 2001, 10)
        ttk.Label(panel, text="Window").pack(anchor="w", pady=(4, 0))
        self.win = tk.StringVar(value=config.DEFAULT_WINDOW)
        box = ttk.Combobox(panel, textvariable=self.win, values=WINS, state="readonly", width=16)
        box.pack(anchor="w")
        box.bind("<<ComboboxSelected>>", lambda e: self.update_plot())
        self.beta = tk.DoubleVar(value=5.65)
        self._scale(panel, "Kaiser beta", self.beta, 0.0, 12.0, 0.05)
        self.result = ttk.Label(panel, text="", style="Big.TLabel", width=34, anchor="w", wraplength=290)
        self.result.pack(anchor="w", fill=tk.X, pady=(8, 4))
        ttk.Button(panel, text="Save coefficients…", command=self.save_coeffs).pack(anchor="w")

    def _build_plot_area(self):
        right = ttk.Frame(self, padding=(6, 10, 14, 10))
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(8.5, 5.6), dpi=100)
        gs = self.fig.add_gridspec(2, 2, hspace=0.42, wspace=0.25)
        self.ax_h = self.fig.add_subplot(gs[0, 0])
        self.ax_f = self.fig.add_subplot(gs[0, 1])
        self.ax_y = self.fig.add_subplot(gs[1, :])
        self.fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.08)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        toolbar = NavigationToolbar2Tk(self.canvas, right, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _task_defaults(self):
        if self.task.get() == "pulse":
            self.f1.set(config.PULSE_BAND[0]); self.f2.set(config.PULSE_BAND[1]); self.N.set(config.N_PULSE)
        else:
            self.f2.set(config.RESP_CUTOFF); self.N.set(config.N_RESP)
        self.update_plot()

    def _design(self):
        N = int(self.N.get()) | 1
        w, beta = self.win.get(), float(self.beta.get())
        if self.task.get() == "pulse":
            lo, hi = sorted((float(self.f1.get()), float(self.f2.get())))
            return fir.design_bandpass(lo, hi, FS, N, w, beta)
        return fir.design_lowpass(float(self.f2.get()), FS, N, w, beta)

    def update_plot(self):
        b = self._design()
        self.b = b
        N = len(b)
        C = plots.C
        for ax in (self.ax_h, self.ax_f, self.ax_y):
            ax.clear()
        self.ax_h.plot((np.arange(N) - (N - 1) / 2) / FS, b, color=C["blue"], lw=1.0)
        self.ax_h.set_title(f"h[n], N = {N} (delay {(N - 1) / 2 / FS:.2f} s)")
        f, H = fir.freq_response(b, FS)
        self.ax_f.plot(f, plots.db(H), color=C["orange"], lw=1.0)
        self.ax_f.set_xscale("log")
        self.ax_f.set_xlim(0.05, 60)
        self.ax_f.set_ylim(-100, 5)
        self.ax_f.set_title("|H(f)| (dB)")
        y = fir.compensate_delay(fir.fir_filter(b, self.x), N)
        g = max(N, 250)
        t = np.arange(len(y)) / FS
        m = (t >= 100) & (t < 130)
        self.ax_y.plot(t[m], self.x[m], color=C["axis"], lw=0.9, label="raw")
        self.ax_y.plot(t[m], y[m], color=C["blue"], lw=1.2, label="filtered")
        self.ax_y.set_xlabel("Time (s)")
        self.ax_y.set_title("PPG 100 to 130 s")
        plots.top_legend(self.ax_y, ncol=2)
        A = self.D
        if self.task.get() == "pulse":
            pk, _ = find_peaks(y, distance=int(config.MIN_BEAT_S * FS))
            pk = pk[(pk > g) & (pk < len(y) - g)]
            tc, hr = analysis.rate_series(pk, FS, len(y) / FS, config.HR_WINDOW_S, 60)
            num = A["numerics"].dropna(subset=["PULSE"])
            ref = np.interp(tc, num["Time [s]"], num["PULSE"])
            ok = ~np.isnan(hr)
            msg = f"Heart rate {np.nanmean(hr):.1f} bpm, monitor {ref[ok].mean():.1f}\nMAE {np.mean(np.abs(hr[ok] - ref[ok])):.2f} bpm"
        else:
            core = y[g:-g]
            pk, _ = find_peaks(y, distance=int(config.MIN_BREATH_S * FS), prominence=0.3 * np.std(core))
            pk = pk[(pk > g) & (pk < len(y) - g)]
            tc, rr = analysis.rate_series(pk, FS, len(y) / FS, config.RR_WINDOW_S, 60)
            _, rref = analysis.rate_series(A["breaths"], FS, len(y) / FS, config.RR_WINDOW_S, 60)
            ok = ~np.isnan(rr) & ~np.isnan(rref)
            msg = f"Breathing {np.nanmean(rr):.1f} per min, annotated {rref[ok].mean():.1f}\nMAE {np.mean(np.abs(rr[ok] - rref[ok])):.2f} per min"
        stop = 20 * np.log10(np.abs(H[f > float(self.f2.get()) + 2 * FS / N]).max() + 1e-15)
        self.result.configure(text=msg + f"\nStopband peak {stop:.0f} dB; linear phase: {fir.is_linear_phase(b)}")
        self.canvas.draw_idle()

    def save_coeffs(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            np.savetxt(path, self.b, delimiter=",")

    # ------------------------------------------------------- report screenshots
    def capture(self, path, delay_ms=2000):
        """Render this window's own pixels into a PNG via Win32 PrintWindow (never the screen)."""
        def grab():
            import ctypes
            from ctypes import wintypes
            from PIL import Image

            self.update()
            user32, gdi32 = ctypes.windll.user32, ctypes.windll.gdi32
            hwnd = user32.GetParent(self.winfo_id()) or self.winfo_id()
            rect = wintypes.RECT()
            user32.GetClientRect(hwnd, ctypes.byref(rect))
            w, h = rect.right, rect.bottom
            hdc = user32.GetDC(hwnd)
            mem = gdi32.CreateCompatibleDC(hdc)
            bmp = gdi32.CreateCompatibleBitmap(hdc, w, h)
            gdi32.SelectObject(mem, bmp)
            user32.PrintWindow(hwnd, mem, 1 | 2)     # PW_CLIENTONLY | PW_RENDERFULLCONTENT

            class BITMAPINFOHEADER(ctypes.Structure):
                _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
                            ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
                            ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                            ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                            ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
                            ("biClrImportant", wintypes.DWORD)]

            header = BITMAPINFOHEADER(ctypes.sizeof(BITMAPINFOHEADER), w, -h, 1, 32, 0, 0, 0, 0, 0, 0)
            buf = ctypes.create_string_buffer(w * h * 4)
            gdi32.GetDIBits(mem, bmp, 0, h, buf, ctypes.byref(header), 0)
            Image.frombuffer("RGB", (w, h), buf, "raw", "BGRX", 0, 1).save(path)
            gdi32.DeleteObject(bmp)
            gdi32.DeleteDC(mem)
            user32.ReleaseDC(hwnd, hdc)
            self.destroy()
        self.after(delay_ms, grab)


def main():
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--taps", type=int, default=0)
    parser.add_argument("--screenshot", help="save a screenshot to this path and exit")
    args = parser.parse_args()
    app = FIRDesignerApp()
    if args.taps:
        app.N.set(args.taps)
    app.update_plot()
    if args.screenshot:
        app.capture(args.screenshot)
    app.mainloop()


if __name__ == "__main__":
    main()
