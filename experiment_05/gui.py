"""
gui.py
------
Interactive "periodicity detector" for Experiment 5.

    python gui.py

Choose a version of the sunspot series (raw, smoothed, residual, difference),
a span of years, the estimator and the maximum lag; optionally bury the
signal in white noise.  The signal, its autocorrelation with the 95 % white-
noise band, the detected period and the decorrelation lag update instantly.
All processing is delegated to ``autocorr``.
"""

import argparse
import tkinter as tk
from tkinter import filedialog, ttk

import matplotlib

matplotlib.use("TkAgg")
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import autocorr as ac
import config
import data_loader
import plots

SERIES = ["Monthly sunspot number", "13-month running mean", "Residual (x − running mean)",
          "First difference"]


class PeriodicityApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DSP Lab · Experiment 5 · Autocorrelation")
        self.geometry("1280x700+10+10")
        self.minsize(1100, 640)
        self.configure(bg="#f6f6f3")
        self.df = data_loader.load_dataset()
        self.x = self.df["sn"].to_numpy()
        self.sm = ac.running_mean(self.x, config.SMOOTH_MONTHS)
        self.noise = np.random.default_rng(config.SEED).normal(0, 1, len(self.x))
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
        style.configure("Big.TLabel", font=("Segoe UI Semibold", 11), foreground="#2a78d6",
                        background="#ffffff", padding=(8, 6))

    def _scale(self, parent, label, var, lo, hi, res):
        ttk.Label(parent, text=label).pack(anchor="w")
        tk.Scale(parent, variable=var, orient=tk.HORIZONTAL, length=270, from_=lo, to=hi, resolution=res,
                 command=lambda v: self.update_plot(), bg="#f6f6f3", highlightthickness=0,
                 troughcolor="#e1e0d9", font=("Segoe UI", 8)).pack(anchor="w")

    def _build_controls(self):
        panel = ttk.Frame(self, padding=(16, 12))
        panel.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(panel, text="Periodicity Detector", style="Header.TLabel").pack(anchor="w")
        ttk.Label(panel, text=f"SILSO sunspot number · {len(self.x):,} months",
                  foreground="#898781").pack(anchor="w", pady=(0, 8))

        ttk.Label(panel, text="SIGNAL", style="Section.TLabel").pack(anchor="w")
        self.series_var = tk.StringVar(value=SERIES[0])
        box = ttk.Combobox(panel, textvariable=self.series_var, values=SERIES, state="readonly", width=30)
        box.pack(anchor="w", pady=(2, 6))
        box.bind("<<ComboboxSelected>>", lambda e: self.update_plot())

        y0, y1 = int(self.df["year"].min()), int(self.df["year"].max())
        self.start_var, self.end_var = tk.IntVar(value=y0), tk.IntVar(value=y1)
        self._scale(panel, "Start year", self.start_var, y0, y1 - 25, 1)
        self._scale(panel, "End year", self.end_var, y0 + 25, y1, 1)
        self.lag_var = tk.IntVar(value=40)
        self._scale(panel, "Maximum lag (years)", self.lag_var, 2, 100, 1)
        self.noise_var = tk.DoubleVar(value=0.0)
        self._scale(panel, "Added white noise (× signal std)", self.noise_var, 0, 8, 0.25)

        self.unbiased_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(panel, text="Unbiased estimator  r[l] / (N − l)", variable=self.unbiased_var,
                        command=self.update_plot).pack(anchor="w", pady=(6, 0))

        self.result = ttk.Label(panel, text="", style="Big.TLabel", width=32, anchor="w", wraplength=280)
        self.result.pack(anchor="w", fill=tk.X, pady=(10, 4))
        ttk.Button(panel, text="Save figure…", command=self.save_figure).pack(anchor="w", pady=(4, 0))

    def _build_plot_area(self):
        right = ttk.Frame(self, padding=(6, 10, 14, 10))
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(8.5, 5.6), dpi=100)
        self.ax1 = self.fig.add_subplot(2, 1, 1)
        self.ax2 = self.fig.add_subplot(2, 1, 2)
        self.fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.09, hspace=0.42)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        toolbar = NavigationToolbar2Tk(self.canvas, right, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _signal(self):
        s = self.series_var.get()
        if s.startswith("13"):
            y, t = self.sm, self.df["decimal_year"].to_numpy()
        elif s.startswith("Residual"):
            y, t = self.x - self.sm, self.df["decimal_year"].to_numpy()
        elif s.startswith("First"):
            y, t = np.diff(self.x), self.df["decimal_year"].to_numpy()[1:]
        else:
            y, t = self.x, self.df["decimal_year"].to_numpy()
        lvl = float(self.noise_var.get())
        if lvl:
            y = y + lvl * np.std(y) * self.noise[: len(y)]
        a, b = sorted((self.start_var.get(), self.end_var.get()))
        keep = (t >= a) & (t < b + 1)
        return t[keep], y[keep], lvl

    def update_plot(self):
        t, y, lvl = self._signal()
        N = len(y)
        L = min(int(self.lag_var.get()) * 12, N - 2)
        rho = ac.acf(y, L, unbiased=self.unbiased_var.get())
        b = ac.confidence_bound(N)
        C = plots.C
        a1, a2 = self.ax1, self.ax2
        a1.clear()
        a2.clear()
        a1.plot(t, y, color=C["blue"], lw=0.6)
        a1.set_title(f"{self.series_var.get()}  ({N:,} samples)")
        a1.set_xlabel("Year")
        lags = np.arange(L + 1) / 12
        a2.fill_between([0, lags[-1]], -b, b, color=C["blue"], alpha=0.12, lw=0, label="95 % white-noise band")
        a2.plot(lags, rho, color=C["blue"], lw=1.2, label="ρ[l]")
        a2.axhline(0, color=C["axis"], lw=0.7)
        msg = []
        if L > config.PEAK_SEARCH[0]:
            k, kr, v = ac.dominant_period(rho, (config.PEAK_SEARCH[0], min(config.PEAK_SEARCH[1], L)))
            a2.plot([k / 12], [v], "o", color=C["orange"], ms=6, mec="white")
            sig = "significant" if v > b else "not significant"
            msg.append(f"Period ≈ {kr / 12:.2f} years (ρ = {v:.2f}, {sig})")
        z = ac.first_zero_crossing(rho)
        if z:
            msg.append(f"First zero crossing: {z} lags ({z / 12:.1f} years)")
        if lvl:
            msg.append(f"Added noise SNR = {10 * np.log10(1 / lvl ** 2):.1f} dB")
        a2.set_xlim(0, lags[-1])
        a2.set_xlabel("Lag (years)")
        a2.set_title("Normalised autocorrelation ρ[l]")
        plots.top_legend(a2, ncol=2)
        self.result.configure(text="\n".join(msg))
        self.canvas.draw_idle()

    def save_figure(self):
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path:
            self.fig.savefig(path, dpi=200)

    # ------------------------------------------------------- report screenshots
    def capture(self, path, delay_ms=1500):
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
    parser.add_argument("--series", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--screenshot", help="save a screenshot to this path and exit")
    args = parser.parse_args()
    app = PeriodicityApp()
    app.series_var.set(SERIES[args.series])
    app.noise_var.set(args.noise)
    app.update_plot()
    if args.screenshot:
        app.capture(args.screenshot)
    app.mainloop()


if __name__ == "__main__":
    main()
