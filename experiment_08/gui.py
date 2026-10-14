"""
gui.py
------
Interactive "spectrum analyser" for Experiment 8.

    python gui.py

Choose the record length (whole years or any number of months), the trend
order, the window and the zero-padding factor, and pick which harmonics the
IDFT keeps.  The spectrum, the reconstructed seasonal cycle, the line
amplitude and the agreement with NOAA's deseasonalized series update
instantly.  A small calculator computes the DFT of any typed sequence.
All processing is delegated to ``dft``.
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
import dft as D
import plots

FS = config.SAMPLES_PER_YEAR


class SpectrumApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DSP Lab · Experiment 8 · DFT and IDFT")
        self.geometry("1280x700+10+10")
        self.minsize(1100, 640)
        self.configure(bg="#f6f6f3")
        self.w = data_loader.whole_years(data_loader.load_dataset())
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
        ttk.Label(panel, text="Spectrum Analyser", style="Header.TLabel").pack(anchor="w")
        ttk.Label(panel, text=f"Mauna Loa CO₂, {len(self.w)} months", foreground="#898781").pack(anchor="w", pady=(0, 4))
        ttk.Label(panel, text="DFT SETTINGS", style="Section.TLabel").pack(anchor="w")
        self.N_var = tk.IntVar(value=len(self.w))
        self._scale(panel, "Record length N (months)", self.N_var, 24, len(self.w), 1)
        self.order_var = tk.IntVar(value=config.TREND_ORDER)
        self._scale(panel, "Trend order removed", self.order_var, 0, 4, 1)
        self.pad_var = tk.IntVar(value=1)
        self._scale(panel, "Zero-padding factor", self.pad_var, 1, 16, 1)
        self.win_var = tk.StringVar(value="Rectangular")
        row = ttk.Frame(panel)
        row.pack(anchor="w", pady=(2, 0))
        for v in ("Rectangular", "Hann"):
            ttk.Radiobutton(row, text=v, value=v, variable=self.win_var, command=self.update_plot).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(panel, text="IDFT: HARMONICS KEPT (cycles per year)", style="Section.TLabel").pack(anchor="w", pady=(6, 0))
        self.h_vars = {}
        row = ttk.Frame(panel)
        row.pack(anchor="w")
        for h in (1, 2, 3, 4):
            v = tk.BooleanVar(value=h in config.HARMONICS)
            ttk.Checkbutton(row, text=str(h), variable=v, command=self.update_plot).pack(side=tk.LEFT, padx=(0, 6))
            self.h_vars[h] = v
        ttk.Label(panel, text="DFT CALCULATOR", style="Section.TLabel").pack(anchor="w", pady=(8, 0))
        self.seq_var = tk.StringVar(value="0.1, 2.5, 0.6, -3.2")
        e = ttk.Entry(panel, textvariable=self.seq_var, width=34)
        e.pack(anchor="w")
        e.bind("<Return>", lambda ev: self.update_plot())
        self.calc = ttk.Label(panel, text="", wraplength=290, foreground="#52514e")
        self.calc.pack(anchor="w", pady=(2, 0))
        self.result = ttk.Label(panel, text="", style="Big.TLabel", width=34, anchor="w", wraplength=290)
        self.result.pack(anchor="w", fill=tk.X, pady=(6, 4))
        ttk.Button(panel, text="Save figure…", command=self.save_figure).pack(anchor="w")

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
        N = int(self.N_var.get())
        w = self.w.iloc[:N]
        x = w["co2"].to_numpy()
        r, trend, _ = D.detrend(x, int(self.order_var.get()))
        win = D.hann(N) if self.win_var.get() == "Hann" else np.ones(N)
        P = int(self.pad_var.get())
        Xw = D.fft(r * win, P * N)
        A = D.amplitude_spectrum(Xw) * P / np.mean(win)
        f = D.frequencies(P * N, FS)[: P * N // 2 + 1]
        X = D.fft(r)
        k_year = N / FS
        bins = [int(round(h * k_year)) for h, v in self.h_vars.items() if v.get()]
        seas = D.idft(D.keep_bins(X, bins)).real if bins else np.zeros(N)
        C = plots.C
        self.ax1.clear()
        self.ax2.clear()
        self.ax1.plot(f, A, color=C["blue"], lw=0.9)
        self.ax1.set_xlim(0, 4.5)
        self.ax1.set_xlabel("Frequency (cycles per year)")
        self.ax1.set_ylabel("ppm")
        self.ax1.set_title(f"Amplitude spectrum, N = {N}, {self.win_var.get()}, padding ×{P}")
        t = w["t"].to_numpy()
        sel = slice(max(0, N - 60), N)
        self.ax2.plot(t[sel], r[sel], color=C["axis"], lw=1.4, label="detrended r[n]")
        self.ax2.plot(t[sel], seas[sel], color=C["orange"], lw=1.2, label=f"IDFT of bins {bins}")
        self.ax2.set_xlabel("Year")
        self.ax2.set_title("IDFT reconstruction (last 5 years shown)")
        plots.top_legend(self.ax2, ncol=2)
        k1 = int(np.argmax(A[f < 1.5] * (f[f < 1.5] > 0.5)))
        on_bin = abs(k_year - round(k_year)) < 1e-9
        rmse = np.sqrt(np.mean((x - seas - w["co2_deseason"].to_numpy()) ** 2))
        self.result.configure(text=(f"Annual line: {A[k1]:.3f} ppm at {f[k1]:.4f} cy/yr\n"
                                    f"N/12 = {k_year:.3f} → {'on a bin (no leakage)' if on_bin else 'between bins (leakage)'}\n"
                                    f"Resolution fs/N = {FS / N:.4f} cy/yr\n"
                                    f"RMSE of x − model vs NOAA: {rmse:.3f} ppm"))
        try:
            s = np.array([float(v) for v in self.seq_var.get().split(",") if v.strip()])
            Xs = np.round(D.dft(s), 10) + 0.0          # hide round-off such as -2e-16
            self.calc.configure(text="X[k] = " + ", ".join(f"{z.real + 0.0:.3g}{z.imag + 0.0:+.3g}j" for z in Xs))
        except ValueError:
            self.calc.configure(text="Invalid sequence")
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
    parser.add_argument("--N", type=int, default=0, help="record length in months (0 = all)")
    parser.add_argument("--hann", action="store_true")
    parser.add_argument("--screenshot", help="save a screenshot to this path and exit")
    args = parser.parse_args()
    app = SpectrumApp()
    if args.N:
        app.N_var.set(args.N)
    if args.hann:
        app.win_var.set("Hann")
    app.update_plot()
    if args.screenshot:
        app.capture(args.screenshot)
    app.mainloop()


if __name__ == "__main__":
    main()
