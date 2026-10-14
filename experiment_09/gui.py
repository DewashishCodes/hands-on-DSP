"""
gui.py
------
Interactive "convolution reverb" for Experiment 9.

    python gui.py

Choose the DFT length (from L up to L + M - 1 and beyond), the method (single
DFT or overlap-add with any block length), and how much of the room impulse
response to use.  The output, the time-aliasing error and the timing update
instantly, and Play buttons let the student hear the dry speech, the
reverberant result and the aliased result.  All processing is delegated to
``fastconv``.
"""

import argparse
import io
import time
import tkinter as tk
from tkinter import filedialog, ttk

import matplotlib

matplotlib.use("TkAgg")
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy.io import wavfile

import data_loader
import fastconv as fc
import plots

METHODS = ["Single DFT of length N", "Overlap-add (block B)"]


class ReverbApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DSP Lab · Experiment 9 · Linear Convolution using the DFT")
        self.geometry("1280x700+10+10")
        self.minsize(1100, 640)
        self.configure(bg="#f6f6f3")
        D = data_loader.load()
        self.fs, self.x, self.h_full = D["fs"], D["dry"], D["rir"]
        self.y = None
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
        ttk.Label(panel, text="Convolution Reverb", style="Header.TLabel").pack(anchor="w")
        ttk.Label(panel, text=f"speech {len(self.x):,} samples · room {len(self.h_full):,} samples",
                  foreground="#898781").pack(anchor="w", pady=(0, 6))
        ttk.Label(panel, text="METHOD", style="Section.TLabel").pack(anchor="w")
        self.method_var = tk.StringVar(value=METHODS[0])
        for m in METHODS:
            ttk.Radiobutton(panel, text=m, value=m, variable=self.method_var, command=self.update_plot).pack(anchor="w")
        self.rir_ms = tk.IntVar(value=1000)
        self._scale(panel, "Impulse response used (ms)", self.rir_ms, 10, 1000, 10)
        self.pad_pct = tk.IntVar(value=100)
        self._scale(panel, "DFT length N: % of required padding M − 1", self.pad_pct, 0, 150, 5)
        self.block = tk.IntVar(value=4096)
        self._scale(panel, "Overlap-add block B (samples)", self.block, 128, 32768, 128)
        self.result = ttk.Label(panel, text="", style="Big.TLabel", width=34, anchor="w", wraplength=290)
        self.result.pack(anchor="w", fill=tk.X, pady=(8, 6))
        row = ttk.Frame(panel)
        row.pack(anchor="w")
        ttk.Button(row, text="▶ Dry", command=lambda: self.play(self.x)).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(row, text="▶ Output", command=lambda: self.play(self.y)).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(row, text="Save WAV…", command=self.save_wav).pack(side=tk.LEFT)
        ttk.Button(panel, text="Save figure…", command=self.save_figure).pack(anchor="w", pady=(6, 0))

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
        h = self.h_full[: max(1, int(self.rir_ms.get() * self.fs / 1000))]
        x = self.x
        L, M = len(x), len(h)
        ref = fc.conv_dft(x, h)                         # exact linear result
        t0 = time.perf_counter()
        if self.method_var.get().startswith("Single"):
            N = L + int(round((M - 1) * self.pad_pct.get() / 100))
            y = fc.conv_dft(x, h, N)
            info = f"N = {N:,}  (L + M − 1 = {L + M - 1:,})"
        else:
            y, N = fc.overlap_add(x, h, int(self.block.get()))
            info = f"FFT size {N:,}, latency {1000 * self.block.get() / self.fs:.0f} ms"
        dt = time.perf_counter() - t0
        n = min(len(y), len(ref))
        err = fc.rel_error(y[:n], ref[:n])
        self.y = y
        C = plots.C
        a1, a2 = self.ax1, self.ax2
        a1.clear()
        a2.clear()
        t = np.arange(len(y)) / self.fs
        a1.plot(t, y, color=C["orange"], lw=0.4)
        a1.set_title("Output y[n]")
        a1.set_xlabel("Time (s)")
        a2.plot(np.arange(n) / self.fs, y[:n] - ref[:n], color=C["red"] if err > 1e-9 else C["aqua"], lw=0.5)
        a2.set_title("Difference from exact linear convolution")
        a2.set_xlabel("Time (s)")
        self.canvas.draw_idle()
        verdict = "exact (linear)" if err < 1e-9 else "TIME-ALIASED"
        self.result.configure(text=f"{info}\nRelative error {err:.1e}: {verdict}\nComputed in {1000 * dt:.1f} ms; "
                                   f"direct would need {L * M / 1e6:,.0f} M multiplications")

    def play(self, s):
        import winsound
        if s is None:
            return
        buf = io.BytesIO()
        wavfile.write(buf, self.fs, (s / np.max(np.abs(s)) * 0.9 * 32767).astype(np.int16))
        winsound.PlaySound(buf.getvalue(), winsound.SND_MEMORY | winsound.SND_ASYNC)

    def save_wav(self):
        path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV", "*.wav")])
        if path and self.y is not None:
            wavfile.write(path, self.fs, (self.y / np.max(np.abs(self.y)) * 0.9 * 32767).astype(np.int16))

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
    parser.add_argument("--pad", type=int, default=100, help="percent of the required zero padding")
    parser.add_argument("--screenshot", help="save a screenshot to this path and exit")
    args = parser.parse_args()
    app = ReverbApp()
    app.pad_pct.set(args.pad)
    app.update_plot()
    if args.screenshot:
        app.capture(args.screenshot)
    app.mainloop()


if __name__ == "__main__":
    main()
