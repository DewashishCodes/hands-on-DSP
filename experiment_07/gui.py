"""
gui.py
------
Interactive "difference-equation smoother" for Experiment 7.

    python gui.py

Choose a preset (moving average, EMA, double EMA, Holt) or type any b and a
coefficients, set the initial condition and the time window, and the GUI
shows the smoothed exchange rate, the impulse response, the frequency
response and the pole-zero plot, together with stability, lag and noise gain.
All processing is delegated to ``smoothing``.
"""

import argparse
import tkinter as tk
from tkinter import filedialog, ttk

import matplotlib

matplotlib.use("TkAgg")
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import config
import data_loader
import plots
import smoothing as sm

PRESETS = ["Moving average (N)", "EMA (N, alpha = 2/(N+1))", "Double EMA (N)", "Holt (alpha, beta)",
           "Custom b, a"]


class SmootherApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DSP Lab · Experiment 7 · Difference-Equation Smoothing")
        self.geometry("1280x700+10+10")
        self.minsize(1100, 640)
        self.configure(bg="#f6f6f3")
        self.df = data_loader.load_dataset()
        self.x = self.df["inr"].to_numpy()
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
        tk.Scale(parent, variable=var, orient=tk.HORIZONTAL, length=270, from_=lo, to=hi, resolution=res,
                 command=lambda v: self.update_plot(), bg="#f6f6f3", highlightthickness=0,
                 troughcolor="#e1e0d9", font=("Segoe UI", 8)).pack(anchor="w")

    def _build_controls(self):
        panel = ttk.Frame(self, padding=(16, 10))
        panel.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(panel, text="Difference-Equation Smoother", style="Header.TLabel").pack(anchor="w")
        ttk.Label(panel, text=f"USD/INR, {len(self.x):,} trading days", foreground="#898781").pack(anchor="w", pady=(0, 6))
        ttk.Label(panel, text="FILTER", style="Section.TLabel").pack(anchor="w")
        self.preset_var = tk.StringVar(value=PRESETS[1])
        box = ttk.Combobox(panel, textvariable=self.preset_var, values=PRESETS, state="readonly", width=30)
        box.pack(anchor="w", pady=(2, 2))
        box.bind("<<ComboboxSelected>>", lambda e: self.update_plot())
        self.n_var = tk.IntVar(value=config.N_EMA)
        self._scale(panel, "N (days)", self.n_var, 1, 100, 1)
        self.beta_var = tk.DoubleVar(value=config.HOLT_BETA)
        self._scale(panel, "Holt trend factor beta", self.beta_var, 0.0, 0.5, 0.01)
        self.b_var, self.a_var = tk.StringVar(value="0.1"), tk.StringVar(value="1, -0.9")
        for txt, var in (("b (x coefficients)", self.b_var), ("a (y coefficients)", self.a_var)):
            row = ttk.Frame(panel)
            row.pack(anchor="w", pady=(2, 0))
            ttk.Label(row, text=txt, width=16).pack(side=tk.LEFT)
            e = ttk.Entry(row, textvariable=var, width=16)
            e.pack(side=tk.LEFT)
            e.bind("<Return>", lambda ev: self.update_plot())
        ttk.Label(panel, text="INITIAL CONDITION AND VIEW", style="Section.TLabel").pack(anchor="w", pady=(8, 0))
        self.ic_var = tk.StringVar(value="y[-1] = x[0]")
        for v in ("y[-1] = x[0]", "y[-1] = 0 (at rest)"):
            ttk.Radiobutton(panel, text=v, value=v, variable=self.ic_var, command=self.update_plot).pack(anchor="w")
        self.year_var = tk.IntVar(value=2022)
        self._scale(panel, "Year shown", self.year_var, 2019, 2025, 1)
        self.result = ttk.Label(panel, text="", style="Big.TLabel", width=34, anchor="w", wraplength=290)
        self.result.pack(anchor="w", fill=tk.X, pady=(8, 4))
        ttk.Button(panel, text="Save figure…", command=self.save_figure).pack(anchor="w")

    def _build_plot_area(self):
        right = ttk.Frame(self, padding=(6, 10, 14, 10))
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(8.5, 5.6), dpi=100)
        gs = self.fig.add_gridspec(2, 3, height_ratios=[1.3, 1], hspace=0.45, wspace=0.32)
        self.ax_t = self.fig.add_subplot(gs[0, :])
        self.ax_h = self.fig.add_subplot(gs[1, 0])
        self.ax_f = self.fig.add_subplot(gs[1, 1])
        self.ax_z = self.fig.add_subplot(gs[1, 2])
        self.fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.08)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        toolbar = NavigationToolbar2Tk(self.canvas, right, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _coeffs(self):
        p, N = self.preset_var.get(), max(1, int(self.n_var.get()))
        alpha = 2.0 / (N + 1)
        if p.startswith("Moving"):
            return sm.ma_coeffs(N)
        if p.startswith("EMA"):
            return sm.ema_coeffs(alpha)
        if p.startswith("Double"):
            e = sm.ema_coeffs(alpha)
            return sm.cascade_coeffs(*e, *e)
        if p.startswith("Holt"):
            return sm.holt_coeffs(alpha, float(self.beta_var.get()))
        b = [float(v) for v in self.b_var.get().split(",") if v.strip()]
        a = [float(v) for v in self.a_var.get().split(",") if v.strip()]
        return np.array(b), np.array(a)

    def update_plot(self):
        try:
            b, a = self._coeffs()
        except ValueError:
            self.result.configure(text="Invalid coefficients")
            return
        x = self.x
        if self.ic_var.get().startswith("y[-1] = x"):
            y = sm.difference_equation(b, a, x, y_init=[x[0]] * (len(a) - 1) or None,
                                       x_init=[x[0]] * (len(b) - 1) or None)
        else:
            y = sm.difference_equation(b, a, x)
        C = plots.C
        for ax in (self.ax_t, self.ax_h, self.ax_f, self.ax_z):
            ax.clear()
        yr = int(self.year_var.get())
        m = (self.df["date"].dt.year == yr).to_numpy()
        if yr == 2019:
            m[:150] = True
        d = self.df["date"][m]
        self.ax_t.plot(d, x[m], color=C["axis"], lw=1.5, label="x[n]")
        self.ax_t.plot(d, y[m], color=C["blue"], lw=1.3, label="y[n]")
        self.ax_t.set_title(f"USD/INR {yr} and the smoothed output")
        plots.top_legend(self.ax_t, ncol=2)
        h = sm.impulse_response(b, a, 60)
        self.ax_h.plot(np.arange(60), h, "o-", color=C["violet"], ms=2, lw=0.9)
        self.ax_h.axhline(0, color=C["axis"], lw=0.7)
        self.ax_h.set_title("h[n]")
        w = np.linspace(1e-3, np.pi, 800)
        H = sm.frequency_response(b, a, w)
        self.ax_f.plot(w / (2 * np.pi), 20 * np.log10(np.abs(H) + 1e-9), color=C["orange"], lw=1.1)
        self.ax_f.set_ylim(-50, 6)
        self.ax_f.set_title("|H| (dB) vs cycles/day")
        p, z = sm.poles_zeros(b, a)
        t = np.linspace(0, 2 * np.pi, 300)
        self.ax_z.plot(np.cos(t), np.sin(t), color=C["axis"], lw=0.8)
        self.ax_z.plot(z.real, z.imag, "o", mfc="none", mec=C["blue"], ms=4)
        self.ax_z.plot(p.real, p.imag, "x", color=C["red"], ms=6, mew=1.3)
        self.ax_z.set_aspect("equal")
        self.ax_z.set_xlim(-1.3, 1.3)
        self.ax_z.set_ylim(-1.3, 1.3)
        self.ax_z.set_title("poles ×, zeros ○")
        stable = sm.is_stable(b, a)
        if stable:
            msg = (f"Stable (max |pole| = {np.max(np.abs(p)) if len(p) else 0:.3f})\n"
                   f"Lag = {sm.dc_lag(b, a) + 0.0 if abs(sm.dc_lag(b, a)) >= 5e-3 else 0.0:.2f} days, noise gain Σh² = {sm.noise_gain(b, a):.4f}\n"
                   f"{sm.ops_per_sample(b, a)} multiplications per sample")
        else:
            msg = f"NOT stable: a pole lies on or outside the unit circle (max |pole| = {np.max(np.abs(p)):.3f})"
        self.result.configure(text=msg)
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
    parser.add_argument("--preset", type=int, default=1)
    parser.add_argument("--screenshot", help="save a screenshot to this path and exit")
    args = parser.parse_args()
    app = SmootherApp()
    app.preset_var.set(PRESETS[args.preset])
    app.update_plot()
    if args.screenshot:
        app.capture(args.screenshot)
    app.mainloop()


if __name__ == "__main__":
    main()
