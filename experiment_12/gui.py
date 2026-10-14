"""
gui.py
------
Interactive "IIR filter designer" for Experiment 12.

    python gui.py

Choose the filter family (Butterworth, Chebyshev I, elliptic), the order, the
pass band, the ripple and stopband attenuation, causal or zero-phase
filtering, and whether to apply the 60 Hz notch.  The GUI shows the
magnitude and group delay, the pole-zero plot, the filtered EEG for eyes open
and eyes closed, and the alpha-power ratio.  All processing is delegated to
``iir``.
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
import iir
import plots

FS = config.FS


class IIRDesignerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DSP Lab · Experiment 12 · IIR Filter Design")
        self.geometry("1280x700+10+10")
        self.minsize(1100, 640)
        self.configure(bg="#f6f6f3")
        self.D = data_loader.load()
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
        ttk.Label(parent, text=label).pack(anchor="w", pady=(1, 0))
        s = tk.Scale(parent, variable=var, orient=tk.HORIZONTAL, length=270, from_=lo, to=hi, resolution=res,
                     bg="#f6f6f3", highlightthickness=0, troughcolor="#e1e0d9", font=("Segoe UI", 8))
        s.bind("<ButtonRelease-1>", lambda e: self.update_plot())
        s.pack(anchor="w")

    def _build_controls(self):
        panel = ttk.Frame(self, padding=(16, 8))
        panel.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(panel, text="IIR Filter Designer", style="Header.TLabel").pack(anchor="w")
        ttk.Label(panel, text=f"EEG {config.CHANNEL}, eyes open / closed, 160 Hz", foreground="#898781").pack(anchor="w", pady=(0, 4))
        ttk.Label(panel, text="FAMILY", style="Section.TLabel").pack(anchor="w")
        self.fam = tk.StringVar(value="butterworth")
        row = ttk.Frame(panel)
        row.pack(anchor="w")
        for fam in config.FAMILIES:
            ttk.Radiobutton(row, text=plots.FNAME[fam], value=fam, variable=self.fam,
                            command=self.update_plot).pack(side=tk.LEFT, padx=(0, 6))
        self.order = tk.IntVar(value=config.ORDER)
        self._scale(panel, "Prototype order", self.order, 1, 10, 1)
        self.f1, self.f2 = tk.DoubleVar(value=config.ALPHA_BAND[0]), tk.DoubleVar(value=config.ALPHA_BAND[1])
        self._scale(panel, "Lower edge (Hz)", self.f1, 1, 40, 0.5)
        self._scale(panel, "Upper edge (Hz)", self.f2, 2, 45, 0.5)
        self.rp, self.rs = tk.DoubleVar(value=config.CHEBY_RIPPLE_DB), tk.DoubleVar(value=config.ELLIP_STOP_DB)
        self._scale(panel, "Passband ripple (dB)", self.rp, 0.1, 3, 0.1)
        self._scale(panel, "Stopband attenuation (dB, elliptic)", self.rs, 20, 80, 1)
        self.zero = tk.BooleanVar(value=False)
        self.notch = tk.BooleanVar(value=True)
        ttk.Checkbutton(panel, text="Zero-phase (forward-backward)", variable=self.zero, command=self.update_plot).pack(anchor="w")
        ttk.Checkbutton(panel, text="60 Hz notch first", variable=self.notch, command=self.update_plot).pack(anchor="w")
        self.result = ttk.Label(panel, text="", style="Big.TLabel", width=34, anchor="w", wraplength=290)
        self.result.pack(anchor="w", fill=tk.X, pady=(6, 4))
        ttk.Button(panel, text="Save SOS coefficients…", command=self.save).pack(anchor="w")

    def _build_plot_area(self):
        right = ttk.Frame(self, padding=(6, 10, 14, 10))
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(8.5, 5.6), dpi=100)
        gs = self.fig.add_gridspec(2, 3, height_ratios=[1, 1.15], hspace=0.45, wspace=0.32)
        self.ax_m = self.fig.add_subplot(gs[0, 0])
        self.ax_g = self.fig.add_subplot(gs[0, 1])
        self.ax_z = self.fig.add_subplot(gs[0, 2])
        self.ax_y = self.fig.add_subplot(gs[1, :])
        self.fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.08)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        toolbar = NavigationToolbar2Tk(self.canvas, right, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_plot(self):
        lo, hi = sorted((float(self.f1.get()), float(self.f2.get())))
        if hi - lo < 0.5:
            hi = lo + 0.5
        d = iir.design_bandpass(self.fam.get(), (lo, hi), FS, int(self.order.get()),
                                float(self.rp.get()), float(self.rs.get()))
        self.sos = d["sos"]
        z, p, _ = d["zpk"]
        C = plots.C
        for ax in (self.ax_m, self.ax_g, self.ax_z, self.ax_y):
            ax.clear()
        f, H = iir.freq_response_sos(self.sos, FS)
        self.ax_m.plot(f, plots.db(H), color=C["blue"], lw=1.1)
        self.ax_m.set_ylim(-90, 5)
        self.ax_m.set_xlim(0, 60)
        self.ax_m.set_title("|H(f)| (dB)")
        fg, tau = iir.group_delay_sos(self.sos, FS)
        m = fg < 60
        self.ax_g.plot(fg[m], tau[m] / FS, color=C["orange"], lw=1.1)
        self.ax_g.set_ylim(0, max(0.2, min(2.0, 1.2 * np.nanmax(tau[(fg > lo) & (fg < hi)]) / FS)))
        self.ax_g.set_title("group delay (s)")
        t = np.linspace(0, 2 * np.pi, 300)
        self.ax_z.plot(np.cos(t), np.sin(t), color=C["axis"], lw=0.8)
        self.ax_z.plot(z.real, z.imag, "o", mfc="none", mec=C["aqua"], ms=4)
        self.ax_z.plot(p.real, p.imag, "x", color=C["red"], ms=4.5)
        self.ax_z.set_aspect("equal")
        self.ax_z.set_title("z-plane")
        bn, an, _ = iir.notch(config.MAINS_HZ, config.NOTCH_Q, FS)
        powers = {}
        for k, col, off in (("open", plots.OPEN, 120), ("closed", plots.CLOSED, -120)):
            x = self.D[k]["x"] - self.D[k]["x"].mean()
            if self.notch.get():
                x = iir.iir_filter(bn, an, x)
            y = iir.filtfilt_sos(self.sos, x) if self.zero.get() else iir.sos_filter(self.sos, x)
            powers[k] = np.mean(y[320:] ** 2)
            tt = np.arange(len(y)) / FS
            mm = (tt >= 20) & (tt < 26)
            self.ax_y.plot(tt[mm], y[mm] + off, color=col, lw=0.7, label=f"eyes {k}")
        self.ax_y.set_yticks([])
        self.ax_y.set_xlabel("Time (s)")
        self.ax_y.set_title("Filtered EEG")
        plots.top_legend(self.ax_y, ncol=2)
        self.canvas.draw_idle()
        inb = (fg >= lo) & (fg <= hi)
        self.result.configure(text=(f"Order {2 * int(self.order.get())} band-pass, {len(self.sos)} biquads\n"
                                    f"Largest pole radius {iir.max_pole_radius(p):.4f}\nIn-band delay {tau[inb].min() / FS:.2f} to "
                                    f"{tau[inb].max() / FS:.2f} s{' (zero-phase applied)' if self.zero.get() else ''}\n"
                                    f"Band power closed / open = {powers['closed'] / powers['open']:.1f}"))

    def save(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            np.savetxt(path, self.sos, delimiter=",")

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
    parser.add_argument("--family", default="elliptic", choices=config.FAMILIES)
    parser.add_argument("--screenshot", help="save a screenshot to this path and exit")
    args = parser.parse_args()
    app = IIRDesignerApp()
    app.fam.set(args.family)
    app.update_plot()
    if args.screenshot:
        app.capture(args.screenshot)
    app.mainloop()


if __name__ == "__main__":
    main()
