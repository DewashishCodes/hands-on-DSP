"""
gui.py
------
Interactive "climate-normals smoother" for Experiment 10.

    python gui.py

Choose the variable (maximum, minimum, diurnal range), the kernel (boxcar,
Gaussian, K harmonics) and its width, and compare circular convolution via
the DFT with ordinary zero-padded convolution; the climatology, the kernel's
harmonic gains and the key dates update instantly.  A small calculator does
N-point circular convolution of two typed sequences via the DFT.
All processing is delegated to ``circdft``.
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
import circdft as cd
import config
import data_loader
import plots

KERNELS = ["Boxcar (M days)", "Gaussian (sigma = M/4 days)", "Harmonics (K = M/10)"]
VARS = ["Maximum temperature", "Minimum temperature", "Diurnal range (max − min)"]
N = config.N_DAYS


class NormalsApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DSP Lab · Experiment 10 · Circular Convolution using the DFT")
        self.geometry("1280x700+10+10")
        self.minsize(1100, 640)
        self.configure(bg="#f6f6f3")
        df = data_loader.load_daily()
        clim = data_loader.climatology(df)
        self.tmax = cd.fill_circular(clim["tmax"].to_numpy())
        self.tmin = cd.fill_circular(clim["tmin"].to_numpy())
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

    def _build_controls(self):
        panel = ttk.Frame(self, padding=(16, 10))
        panel.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(panel, text="Climate-Normals Smoother", style="Header.TLabel").pack(anchor="w")
        ttk.Label(panel, text=f"{config.STATION_NAME}, 1991 to 2020", foreground="#898781").pack(anchor="w", pady=(0, 6))
        ttk.Label(panel, text="SIGNAL", style="Section.TLabel").pack(anchor="w")
        self.var_var = tk.StringVar(value=VARS[0])
        box = ttk.Combobox(panel, textvariable=self.var_var, values=VARS, state="readonly", width=30)
        box.pack(anchor="w", pady=(2, 6))
        box.bind("<<ComboboxSelected>>", lambda e: self.update_plot())
        ttk.Label(panel, text="KERNEL", style="Section.TLabel").pack(anchor="w")
        self.k_var = tk.StringVar(value=KERNELS[0])
        for k in KERNELS:
            ttk.Radiobutton(panel, text=k, value=k, variable=self.k_var, command=self.update_plot).pack(anchor="w")
        ttk.Label(panel, text="Width M (days)").pack(anchor="w", pady=(4, 0))
        self.m_var = tk.IntVar(value=config.MA_DAYS)
        tk.Scale(panel, variable=self.m_var, orient=tk.HORIZONTAL, length=270, from_=1, to=121, resolution=2,
                 command=lambda v: self.update_plot(), bg="#f6f6f3", highlightthickness=0,
                 troughcolor="#e1e0d9", font=("Segoe UI", 8)).pack(anchor="w")
        self.lin_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(panel, text="Also show zero-padded linear result", variable=self.lin_var,
                        command=self.update_plot).pack(anchor="w", pady=(4, 0))
        ttk.Label(panel, text="N-POINT CALCULATOR (via DFT)", style="Section.TLabel").pack(anchor="w", pady=(8, 0))
        self.x_s, self.h_s = tk.StringVar(value="20, 30, 40, 30"), tk.StringVar(value="0.5, 0.25, 0, 0.25")
        for txt, v in (("x", self.x_s), ("h", self.h_s)):
            row = ttk.Frame(panel)
            row.pack(anchor="w")
            ttk.Label(row, text=txt, width=3).pack(side=tk.LEFT)
            e = ttk.Entry(row, textvariable=v, width=28)
            e.pack(side=tk.LEFT)
            e.bind("<Return>", lambda ev: self.update_plot())
        self.calc = ttk.Label(panel, text="", foreground="#52514e", wraplength=290)
        self.calc.pack(anchor="w")
        self.result = ttk.Label(panel, text="", style="Big.TLabel", width=34, anchor="w", wraplength=290)
        self.result.pack(anchor="w", fill=tk.X, pady=(6, 4))
        ttk.Button(panel, text="Save figure…", command=self.save_figure).pack(anchor="w")

    def _build_plot_area(self):
        right = ttk.Frame(self, padding=(6, 10, 14, 10))
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(8.5, 5.6), dpi=100)
        gs = self.fig.add_gridspec(2, 2, height_ratios=[1.5, 1], hspace=0.42, wspace=0.25)
        self.ax1 = self.fig.add_subplot(gs[0, :])
        self.ax2 = self.fig.add_subplot(gs[1, 0])
        self.ax3 = self.fig.add_subplot(gs[1, 1])
        self.fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.08)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        toolbar = NavigationToolbar2Tk(self.canvas, right, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _kernel(self):
        M = max(1, int(self.m_var.get()))
        k = self.k_var.get()
        if k.startswith("Boxcar"):
            return cd.boxcar(M, N), M // 2
        if k.startswith("Gauss"):
            return cd.gaussian(max(M / 4, 0.5), N), int(np.ceil(M))
        return cd.harmonic_kernel(max(1, M // 10), N), None

    def update_plot(self):
        v = self.var_var.get()
        x = self.tmax if v.startswith("Max") else self.tmin if v.startswith("Min") else self.tmax - self.tmin
        h, half = self._kernel()
        y = cd.circ_dft(x, h)
        C = plots.C
        for ax in (self.ax1, self.ax2, self.ax3):
            ax.clear()
        k = np.arange(N)
        self.ax1.plot(k, x, color=C["axis"], lw=1.1, label="raw x[k]")
        if self.lin_var.get() and half is not None and half > 0:
            self.ax1.plot(k, cd.linear_same(x, h, half), color=C["red"], lw=1.0, ls="--", label="linear (zero-padded)")
        self.ax1.plot(k, y, color=C["blue"], lw=1.5, label="circular via DFT")
        plots.month_axis(self.ax1)
        self.ax1.set_title(v)
        plots.top_legend(self.ax1, ncol=3)
        n = np.arange(-60, 61)
        self.ax2.plot(n, h[n % N], color=C["violet"], lw=1.2)
        self.ax2.set_title("kernel h[n]")
        H = cd.frequency_response(h)
        self.ax3.plot(np.arange(31), H[:31], "o-", color=C["orange"], ms=2.5, lw=0.9)
        self.ax3.set_title("|H[k]| for k = 0 ... 30")
        self.canvas.draw_idle()
        self.result.configure(text=(f"Peak {analysis.day_label(np.argmax(y))} ({y.max():.1f})  ·  "
                                    f"trough {analysis.day_label(np.argmin(y))} ({y.min():.1f})\n"
                                    f"Residual std {np.std(x - y):.2f} °C\n"
                                    f"Roughness {cd.roughness(y):.3f} °C per day"))
        try:
            xs = np.array([float(t) for t in self.x_s.get().split(",") if t.strip()])
            hs = np.array([float(t) for t in self.h_s.get().split(",") if t.strip()])
            Nn = max(len(xs), len(hs))
            out = cd.circ_dft(np.r_[xs, np.zeros(Nn - len(xs))], np.r_[hs, np.zeros(Nn - len(hs))])
            self.calc.configure(text=f"{Nn}-point result: [" + ", ".join(f"{t + 0.0:.4g}" for t in np.round(out, 10)) + "]")
        except ValueError:
            self.calc.configure(text="Invalid sequence")

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
    parser.add_argument("--var", type=int, default=0)
    parser.add_argument("--width", type=int, default=config.MA_DAYS)
    parser.add_argument("--screenshot", help="save a screenshot to this path and exit")
    args = parser.parse_args()
    app = NormalsApp()
    app.var_var.set(VARS[args.var])
    app.m_var.set(args.width)
    app.update_plot()
    if args.screenshot:
        app.capture(args.screenshot)
    app.mainloop()


if __name__ == "__main__":
    main()
