"""
gui.py
------
Interactive "time-delay estimator" for Experiment 6.

    python gui.py

Pick any two tide gauges, the band (raw, tidal or subtidal) and the lag range;
the two signals, their cross-correlation, the estimated delay and the implied
wave speed update instantly.  A "manual shift" slider lets the student slide
one signal over the other and watch the correlation coefficient change, which
is exactly what cross-correlation computes for every lag.
"""

import argparse
import tkinter as tk
from tkinter import filedialog, ttk

import matplotlib

matplotlib.use("TkAgg")
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import analysis
import config
import data_loader
import plots
import xcorr as xc

BANDS = ["raw", "tidal", "subtidal"]
NAMES = [f"{sid}  {v[0]}" for sid, v in config.STATIONS.items()]


class DelayApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DSP Lab · Experiment 6 · Cross-Correlation")
        self.geometry("1280x700+10+10")
        self.minsize(1100, 640)
        self.configure(bg="#f6f6f3")
        self.df = data_loader.load_dataset()
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

    def _combo(self, parent, label, var, values):
        ttk.Label(parent, text=label).pack(anchor="w", pady=(4, 0))
        box = ttk.Combobox(parent, textvariable=var, values=values, state="readonly", width=30)
        box.pack(anchor="w")
        box.bind("<<ComboboxSelected>>", lambda e: self.update_plot())

    def _scale(self, parent, label, var, lo, hi, res):
        ttk.Label(parent, text=label).pack(anchor="w", pady=(4, 0))
        tk.Scale(parent, variable=var, orient=tk.HORIZONTAL, length=270, from_=lo, to=hi, resolution=res,
                 command=lambda v: self.update_plot(), bg="#f6f6f3", highlightthickness=0,
                 troughcolor="#e1e0d9", font=("Segoe UI", 8)).pack(anchor="w")

    def _build_controls(self):
        panel = ttk.Frame(self, padding=(16, 12))
        panel.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(panel, text="Time-Delay Estimator", style="Header.TLabel").pack(anchor="w")
        ttk.Label(panel, text=f"NOAA tide gauges · {len(self.df)} hourly samples",
                  foreground="#898781").pack(anchor="w", pady=(0, 6))
        ttk.Label(panel, text="SIGNALS", style="Section.TLabel").pack(anchor="w")
        self.x_var, self.y_var = tk.StringVar(value=NAMES[0]), tk.StringVar(value=NAMES[-1])
        self._combo(panel, "Reference x[n]", self.x_var, NAMES)
        self._combo(panel, "Delayed signal y[n]", self.y_var, NAMES)
        self.band_var = tk.StringVar(value="tidal")
        self._combo(panel, "Band", self.band_var, BANDS)
        ttk.Label(panel, text="ANALYSIS", style="Section.TLabel").pack(anchor="w", pady=(8, 0))
        self.lag_var = tk.IntVar(value=36)
        self._scale(panel, "Maximum lag (hours)", self.lag_var, 6, 72, 1)
        self.shift_var = tk.DoubleVar(value=0.0)
        self._scale(panel, "Manual shift of y (hours)", self.shift_var, -24, 24, 0.5)
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

    def update_plot(self):
        sx, sy = self.x_var.get().split()[0], self.y_var.get().split()[0]
        which = self.band_var.get()
        x, y = analysis.band(self.df, sx, which), analysis.band(self.df, sy, which)
        L = int(self.lag_var.get())
        lags, rho = xc.ccf(y, x, L)
        k, kr, v = xc.peak_lag(lags, rho, (-L, L))
        C = plots.C
        a1, a2 = self.ax1, self.ax2
        a1.clear()
        a2.clear()
        t = np.arange(len(x))
        sh = float(self.shift_var.get())
        a1.plot(t, x, color=C["ink2"], lw=1.0, label=f"x: {config.STATIONS[sx][0]}")
        a1.plot(t - sh, y, color=C["violet"], lw=1.0, label=f"y shifted by {sh:+g} h: {config.STATIONS[sy][0]}")
        a1.set_xlim(200, 320)
        a1.set_xlabel("Hour of January 2024")
        a1.set_title(f"Signals ({which} band)")
        plots.top_legend(a1, ncol=2)
        a2.plot(lags, rho, color=C["blue"], lw=1.4, label="ρ_yx[l]")
        a2.plot([kr], [v], "o", color=C["orange"], ms=6, mec="white")
        a2.axvline(sh, color=C["violet"], lw=0.9, ls="--", label="manual shift")
        a2.axhline(0, color=C["axis"], lw=0.7)
        a2.set_xlim(-L, L)
        a2.set_ylim(-1.05, 1.05)
        a2.set_xlabel("Lag l (hours)")
        a2.set_title("Normalised cross-correlation")
        plots.top_legend(a2, ncol=2)
        cur = float(np.interp(sh, lags, rho))
        d = analysis.distances()
        dist = abs(d[sy] - d[sx])
        msg = [f"Estimated delay: {kr:+.2f} h (ρ = {v:.3f})", f"ρ at manual shift: {cur:.3f}"]
        if abs(kr) > 0.3 and dist > 1:
            msg.append(f"Distance {dist:.0f} km → speed {dist / abs(kr) / 3.6:.2f} m/s")
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
    parser.add_argument("--x", type=int, default=0)
    parser.add_argument("--y", type=int, default=4)
    parser.add_argument("--shift", type=float, default=0.0)
    parser.add_argument("--screenshot", help="save a screenshot to this path and exit")
    args = parser.parse_args()
    app = DelayApp()
    app.x_var.set(NAMES[args.x])
    app.y_var.set(NAMES[args.y])
    app.shift_var.set(args.shift)
    app.update_plot()
    if args.screenshot:
        app.capture(args.screenshot)
    app.mainloop()


if __name__ == "__main__":
    main()
