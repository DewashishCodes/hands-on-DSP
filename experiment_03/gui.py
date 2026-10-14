"""
gui.py
------
Interactive "sampling lab" for Experiment 3.

    python gui.py

Choose a sampling rate, an anti-aliasing prefilter (none, moving average of any
length, or FIR) and a reconstruction method; the time view, spectrum and SNR
update instantly, and the Play buttons let you *hear* aliasing.
All processing is delegated to ``sampling``.
"""

import argparse
import io
import tkinter as tk
from tkinter import filedialog, ttk

import matplotlib

matplotlib.use("TkAgg")
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy.io import wavfile

import config
import data_loader
import plots
import sampling as sp

RATE_CHOICES = [24_000, 16_000, 12_000, 8_000, 6_000, 4_000, 3_000, 2_000]


class SamplingLabApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DSP Lab · Experiment 3 · Sampling and Reconstruction")
        self.geometry("1280x700+10+10")
        self.minsize(1100, 640)
        self.configure(bg="#f6f6f3")
        self.fs, self.x = data_loader.load_reference()
        self.N = len(self.x)
        self.i_mid = int(np.argmax(np.abs(self.x[4000:-4000]))) + 4000
        self._build_style()
        self._build_controls()
        self._build_plot_area()
        self.update_plot()

    # ------------------------------------------------------------------ layout
    def _build_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background="#f6f6f3", font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI Semibold", 13))
        style.configure("Section.TLabel", font=("Segoe UI Semibold", 10), foreground="#52514e")
        style.configure("Big.TLabel", font=("Segoe UI Semibold", 12), foreground="#2a78d6",
                        background="#ffffff", padding=(8, 4))
        style.configure("Treeview", font=("Segoe UI", 9), rowheight=22)
        style.configure("Treeview.Heading", font=("Segoe UI Semibold", 9))

    def _build_controls(self):
        panel = ttk.Frame(self, padding=(16, 12))
        panel.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(panel, text="Sampling Lab", style="Header.TLabel").pack(anchor="w")
        ttk.Label(panel, text=f"ESC-50 bird song · reference {self.fs // 1000} kHz · 5 s",
                  foreground="#898781").pack(anchor="w", pady=(0, 8))

        ttk.Label(panel, text="SAMPLING RATE", style="Section.TLabel").pack(anchor="w")
        self.rate_var = tk.StringVar(value="8000")
        box = ttk.Combobox(panel, textvariable=self.rate_var, values=[str(r) for r in RATE_CHOICES],
                           state="readonly", width=12)
        box.pack(anchor="w", pady=(2, 8))
        box.bind("<<ComboboxSelected>>", lambda e: self._sync_ma())

        ttk.Label(panel, text="ANTI-ALIASING PREFILTER", style="Section.TLabel").pack(anchor="w")
        self.pre_var = tk.StringVar(value="None")
        for p in config.PREFILTERS:
            ttk.Radiobutton(panel, text=p, value=p, variable=self.pre_var,
                            command=self.update_plot).pack(anchor="w")
        ttk.Label(panel, text="Moving-average length M (0 = match D)").pack(anchor="w", pady=(4, 0))
        self.m_var = tk.IntVar(value=0)
        tk.Scale(panel, variable=self.m_var, orient=tk.HORIZONTAL, length=270, from_=0, to=48,
                 command=lambda v: self.update_plot(), bg="#f6f6f3", highlightthickness=0,
                 troughcolor="#e1e0d9", font=("Segoe UI", 8)).pack(anchor="w")

        ttk.Label(panel, text="RECONSTRUCTION", style="Section.TLabel").pack(anchor="w", pady=(8, 0))
        self.rec_var = tk.StringVar(value=config.RECON_METHODS[-1])
        for m in config.RECON_METHODS:
            ttk.Radiobutton(panel, text=m, value=m, variable=self.rec_var,
                            command=self.update_plot).pack(anchor="w")

        self.snr_label = ttk.Label(panel, text="", style="Big.TLabel", width=30, anchor="w")
        self.snr_label.pack(anchor="w", fill=tk.X, pady=(10, 6))
        btns = ttk.Frame(panel)
        btns.pack(anchor="w")
        ttk.Button(btns, text="▶ Play original", command=lambda: self.play(self.x, self.fs)).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btns, text="▶ Play sampled", command=self.play_sampled).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(panel, text="Save figure…", command=self.save_figure).pack(anchor="w", pady=(8, 0))

    def _build_plot_area(self):
        right = ttk.Frame(self, padding=(6, 10, 14, 10))
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(8.5, 4.9), dpi=100)
        self.ax1 = self.fig.add_subplot(2, 1, 1)
        self.ax2 = self.fig.add_subplot(2, 1, 2)
        self.fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.1, hspace=0.45)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        toolbar = NavigationToolbar2Tk(self.canvas, right, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        cols = ("quantity", "value")
        self.table = ttk.Treeview(right, columns=cols, show="headings", height=3)
        self.table.heading("quantity", text="Quantity")
        self.table.heading("value", text="Value")
        self.table.column("quantity", width=380)
        self.table.column("value", width=300, anchor="center")
        self.table.pack(fill=tk.X, pady=(6, 0))

    # --------------------------------------------------------------- behaviour
    def _sync_ma(self):
        self.update_plot()

    def _process(self):
        rate = int(self.rate_var.get())
        D = self.fs // rate
        kind = self.pre_var.get()
        if kind == "Moving average" and self.m_var.get() > 0:
            xf = sp.moving_average(self.x, self.m_var.get())
            xs = xf[::D]
        else:
            xs = sp.sample(self.x, self.fs, rate, kind)
        y = sp.reconstruct(xs, D, self.N, self.rec_var.get())
        ref = sp.bandlimit(self.x, self.fs, rate / 2)
        return rate, D, xs, y, ref

    def update_plot(self):
        rate, D, xs, y, ref = self._process()
        snr = sp.snr_db(ref, y, 2000)
        C = plots.C
        a1, a2 = self.ax1, self.ax2
        a1.clear()
        a2.clear()
        sel = slice(self.i_mid - 120, self.i_mid + 120)
        t = np.arange(self.N)[sel] / self.fs * 1000
        a1.plot(t, ref[sel], color=C["axis"], lw=2.4, label="band-limited reference")
        a1.plot(t, y[sel], color=C["blue"], lw=1.2, label="reconstruction",
                drawstyle="steps-post" if self.rec_var.get().startswith("Zero") else "default")
        k = np.arange((self.i_mid - 120) // D + 1, (self.i_mid + 120) // D)
        plots.stem(a1, k * D / self.fs * 1000, xs[k], C["violet"], "samples", markersize=3)
        a1.set_title(f"5 ms of the chirp, sampled at {rate / 1000:g} kHz")
        a1.set_xlabel("Time (ms)")
        plots.top_legend(a1, ncol=3)
        fr, Sx = sp.spectrum_db(self.x, self.fs)
        _, Sy = sp.spectrum_db(y, self.fs)
        a2.plot(fr[::8], Sx[::8], color=C["ink2"], lw=0.5, label="original")
        a2.plot(fr[::8], Sy[::8], color=C["orange"], lw=0.6, label="reconstructed")
        a2.axvline(rate / 2, color=C["red"], ls="--", lw=1.0, label="Nyquist fs/2")
        a2.set_xlim(0, 12000)
        a2.set_ylim(-100, 3)
        plots.khz(a2)
        a2.set_xlabel("Frequency (kHz)")
        a2.set_ylabel("dB")
        a2.set_title("Spectrum")
        plots.top_legend(a2, ncol=3)
        self.canvas.draw_idle()
        self.snr_label.configure(text=f"SNR = {snr:.1f} dB")
        self.table.delete(*self.table.get_children())
        lost = sp.energy_above(self.x, self.fs, rate / 2)
        for q, v in (("Decimation factor D = fs_ref / fs", f"{D}"),
                     ("Signal energy above fs/2 (aliased if not filtered)", f"{100 * lost:.1f} %"),
                     ("Storage for one hour, 16-bit mono", f"{rate * 2 * 3600 / 1e6:.1f} MB")):
            self.table.insert("", tk.END, values=(q, v))

    def play(self, x, fs):
        import winsound
        buf = io.BytesIO()
        wavfile.write(buf, int(fs), (np.clip(x, -1, 1) * 32767).astype(np.int16))
        winsound.PlaySound(buf.getvalue(), winsound.SND_MEMORY | winsound.SND_ASYNC)

    def play_sampled(self):
        rate, D, xs, y, ref = self._process()
        self.play(xs, rate)

    def save_figure(self):
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path:
            self.fig.savefig(path, dpi=200)

    # ------------------------------------------------------- report screenshots
    def capture(self, path, delay_ms=1800):
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
    parser.add_argument("--rate", type=int, default=8000)
    parser.add_argument("--prefilter", type=int, default=0, help="index in the prefilter list")
    parser.add_argument("--recon", type=int, default=2, help="index in the reconstruction list")
    parser.add_argument("--screenshot", help="save a screenshot to this path and exit")
    args = parser.parse_args()
    app = SamplingLabApp()
    app.rate_var.set(str(args.rate))
    app.pre_var.set(config.PREFILTERS[args.prefilter])
    app.rec_var.set(config.RECON_METHODS[args.recon])
    app.update_plot()
    if args.screenshot:
        app.capture(args.screenshot)
    app.mainloop()


if __name__ == "__main__":
    main()
