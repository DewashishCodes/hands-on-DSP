"""
gui.py
------
Interactive "wind rose and runway planner" for Experiment 4.

    python gui.py

Choose a season and a smoothing kernel to compare circular and linear
convolution on the wind-direction histogram, or switch to the runway view to
see the usability of every heading for any crosswind limit.  A small
"N-point" panel shows circular convolution of two short sequences typed by
the user.  All processing is delegated to ``circular`` and ``analysis``.
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
import circular as cc
import config
import data_loader
import plots

VIEWS = ["Smoothing: circular vs linear", "Runway usability", "Custom N-point sequences"]


class WindLabApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DSP Lab · Experiment 4 · Circular Convolution")
        self.geometry("1280x700+10+10")
        self.minsize(1100, 640)
        self.configure(bg="#f6f6f3")
        self.df = data_loader.load_dataset()
        self.H, self.speeds = analysis.histogram_by_speed(self.df)
        self.always = int((self.df["calm"] | self.df["variable"]).sum())
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
                        background="#ffffff", padding=(8, 4))

    def _scale(self, parent, label, var, lo, hi, res):
        ttk.Label(parent, text=label).pack(anchor="w")
        tk.Scale(parent, variable=var, orient=tk.HORIZONTAL, length=270, from_=lo, to=hi, resolution=res,
                 command=lambda v: self.update_plot(), bg="#f6f6f3", highlightthickness=0,
                 troughcolor="#e1e0d9", font=("Segoe UI", 8)).pack(anchor="w")

    def _build_controls(self):
        panel = ttk.Frame(self, padding=(16, 12))
        panel.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(panel, text="Wind Rose Lab", style="Header.TLabel").pack(anchor="w")
        ttk.Label(panel, text=f"{config.STATION} METAR {config.YEAR} · {len(self.df):,} reports",
                  foreground="#898781").pack(anchor="w", pady=(0, 8))

        ttk.Label(panel, text="VIEW", style="Section.TLabel").pack(anchor="w")
        self.view_var = tk.StringVar(value=VIEWS[0])
        for v in VIEWS:
            ttk.Radiobutton(panel, text=v, value=v, variable=self.view_var,
                            command=self.update_plot).pack(anchor="w")

        ttk.Label(panel, text="SEASON", style="Section.TLabel").pack(anchor="w", pady=(8, 2))
        self.season_var = tk.StringVar(value="Annual")
        box = ttk.Combobox(panel, textvariable=self.season_var, values=list(config.SEASONS),
                           state="readonly", width=24)
        box.pack(anchor="w")
        box.bind("<<ComboboxSelected>>", lambda e: self.update_plot())

        self.width_var = tk.IntVar(value=3)
        self._scale(panel, "Smoothing kernel length (triangular, bins)", self.width_var, 1, 15, 2)
        self.limit_var = tk.DoubleVar(value=config.DEFAULT_LIMIT_KT)
        self._scale(panel, "Crosswind limit (kt)", self.limit_var, 3, 25, 0.5)
        self.head_var = tk.IntVar(value=90)
        self._scale(panel, "Runway heading (degrees)", self.head_var, 0, 170, 10)

        ttk.Label(panel, text="CUSTOM SEQUENCES (comma separated)", style="Section.TLabel").pack(anchor="w", pady=(8, 2))
        self.x_var = tk.StringVar(value="1, 2, 3, 4")
        self.h_var = tk.StringVar(value="1, 1, 0, 0")
        self.n_var = tk.IntVar(value=4)
        for txt, var in (("x[n]", self.x_var), ("h[n]", self.h_var)):
            row = ttk.Frame(panel)
            row.pack(anchor="w")
            ttk.Label(row, text=txt, width=5).pack(side=tk.LEFT)
            e = ttk.Entry(row, textvariable=var, width=24)
            e.pack(side=tk.LEFT)
            e.bind("<Return>", lambda ev: self.update_plot())
        row = ttk.Frame(panel)
        row.pack(anchor="w", pady=(2, 0))
        ttk.Label(row, text="N", width=5).pack(side=tk.LEFT)
        ttk.Spinbox(row, from_=1, to=32, textvariable=self.n_var, width=5,
                    command=self.update_plot).pack(side=tk.LEFT)

        self.result = ttk.Label(panel, text="", style="Big.TLabel", width=34, anchor="w", wraplength=290)
        self.result.pack(anchor="w", fill=tk.X, pady=(10, 4))
        ttk.Button(panel, text="Save figure…", command=self.save_figure).pack(anchor="w", pady=(4, 0))

    def _build_plot_area(self):
        right = ttk.Frame(self, padding=(6, 10, 14, 10))
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(8.5, 5.6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        toolbar = NavigationToolbar2Tk(self.canvas, right, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # --------------------------------------------------------------- behaviour
    def _kernel(self):
        L = int(self.width_var.get()) | 1
        w = np.r_[np.arange(1, L // 2 + 2), np.arange(L // 2, 0, -1)].astype(float)
        return w / w.sum()

    def update_plot(self):
        self.fig.clear()
        view = self.view_var.get()
        C = plots.C
        if view.startswith("Smoothing"):
            p = analysis.direction_histogram(self.df, config.SEASONS[self.season_var.get()])
            w = self._kernel()
            yc, yl = cc.smooth_circular(p, w), cc.smooth_linear(p, w)
            ax = self.fig.add_subplot(1, 2, 1)
            ax.bar(plots.DEG, p, width=8, color=C["axis"], label="p[k]")
            ax.plot(plots.DEG, yc, "o-", color=C["blue"], ms=3, label="circular")
            ax.plot(plots.DEG, yl, "s--", color=C["orange"], ms=2.5, label="linear")
            ax.set_xticks(range(0, 360, 60))
            ax.set_xlabel("Direction (degrees)")
            ax.set_title(f"{self.season_var.get()}, kernel length {len(w)}")
            plots.top_legend(ax, ncol=3)
            pax = self.fig.add_subplot(1, 2, 2, projection="polar")
            plots.rose(pax, yc, C["blue"], "Smoothed wind rose (circular)")
            self.result.configure(text=f"Total kept: circular {yc.sum():.0f}, linear {yl.sum():.1f} of {p.sum():.0f}. "
                                       f"Max seam error {np.max(np.abs(yc - yl)):.1f} reports.")
        elif view.startswith("Runway"):
            lim = float(self.limit_var.get())
            cov = cc.runway_coverage(self.H, self.speeds, lim, self.always, len(self.df), config.BIN_DEG)
            ax = self.fig.add_subplot(1, 1, 1)
            ax.plot(plots.DEG[:18], 100 * cov[:18], "o-", color=C["blue"])
            ax.axhline(95, color=C["red"], ls="--", lw=0.9)
            h = int(self.head_var.get())
            ax.axvline(h, color=C["orange"], lw=1.2)
            ax.set_xticks(range(0, 180, 10))
            ax.set_xlabel("Runway heading (degrees)")
            ax.set_ylabel("Usability (%)")
            ax.set_title(f"Usability of every heading, crosswind limit {lim:g} kt")
            best = int(np.argmax(cov[:18])) * config.BIN_DEG
            self.result.configure(text=f"Heading {h:03d}/{h + 180:03d}: {100 * cov[h // 10]:.2f} % usable. "
                                       f"Best heading {best:03d}° ({100 * cov[:18].max():.2f} %).")
        else:
            try:
                x = [float(v) for v in self.x_var.get().split(",") if v.strip()]
                h = [float(v) for v in self.h_var.get().split(",") if v.strip()]
                N = max(int(self.n_var.get()), len(x), len(h))
            except ValueError:
                self.result.configure(text="Invalid sequence")
                self.canvas.draw_idle()
                return
            yc = cc.circular_convolve(x, h, N)
            yl = np.convolve(x, h)
            a1 = self.fig.add_subplot(2, 1, 1)
            plots.stem(a1, np.arange(len(yl)), yl, C["ink2"], "linear x * h")
            a1.set_title(f"Linear convolution, length {len(yl)}")
            a1.legend(loc="upper right")
            a2 = self.fig.add_subplot(2, 1, 2, sharex=a1)
            plots.stem(a2, np.arange(N), yc, C["blue"], f"{N}-point circular")
            a2.set_title(f"{N}-point circular convolution" + ("  (equals linear)" if N >= len(yl) else "  (aliased)"))
            a2.legend(loc="upper right")
            self.result.configure(text="y = [" + ", ".join(f"{v:g}" for v in yc) + "]")
        self.fig.tight_layout()
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
    parser.add_argument("--view", type=int, default=0)
    parser.add_argument("--season", type=int, default=0)
    parser.add_argument("--screenshot", help="save a screenshot to this path and exit")
    args = parser.parse_args()
    app = WindLabApp()
    app.view_var.set(VIEWS[args.view])
    app.season_var.set(list(config.SEASONS)[args.season])
    app.update_plot()
    if args.screenshot:
        app.capture(args.screenshot)
    app.mainloop()


if __name__ == "__main__":
    main()
