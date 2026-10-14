"""
gui.py
------
Interactive Tkinter front-end for Experiment 1.

    python gui.py

Choose two signals from the dataset, pick an operation, adjust its parameter
with the slider, and the plots and the statistics table update instantly.
All processing is delegated to ``signal_ops`` -- the GUI only wires widgets.
"""

import argparse
import tkinter as tk
from tkinter import filedialog, ttk

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import config
import data_loader
import plots
import signal_ops as ops

# Parameter attached to each operation: (key, label, from, to, resolution, default)
PARAMS = {
    "Amplitude scaling": ("A", "Amplitude factor A", -5.0, 5.0, 0.05, config.AMP_SCALE_A),
    "Time shift":        ("k", "Shift k  (+ delay / − advance)", -48, 48, 1, config.SHIFT_K),
    "Downsampling":      ("D", "Decimation factor D", 1, 8, 1, config.DECIMATION_D),
    "Upsampling":        ("L", "Expansion factor L", 1, 8, 1, config.INTERP_L),
    "Convolution (MA)":  ("M", "Moving-average length M", 1, 72, 1, config.MA_LENGTH_M),
}
TWO_INPUT_OPS = {"Addition", "Subtraction", "Multiplication"}
STAT_COLS = ("signal", "N", "support", "mean", "min", "max", "energy", "power")


class SignalLabApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DSP Lab · Experiment 1 · Discrete-Time Signal Operations")
        self.geometry("1280x700+10+10")
        self.minsize(1100, 640)
        self.configure(bg="#f6f6f3")

        self.df = data_loader.load_dataset()
        self.columns = list(config.SIGNAL_INFO)
        self.result = None

        self._build_style()
        self._build_controls()
        self._build_plot_area()
        self._on_operation_change()

    # ------------------------------------------------------------------ layout
    def _build_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background="#f6f6f3", font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI Semibold", 13))
        style.configure("Section.TLabel", font=("Segoe UI Semibold", 10), foreground="#52514e")
        style.configure("Eq.TLabel", font=("Cambria Math", 12), foreground="#2a78d6",
                        background="#ffffff", padding=(8, 4))
        style.configure("Treeview", font=("Segoe UI", 9), rowheight=22)
        style.configure("Treeview.Heading", font=("Segoe UI Semibold", 9))

    def _build_controls(self):
        panel = ttk.Frame(self, padding=(16, 14))
        panel.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(panel, text="Signal Operations Lab", style="Header.TLabel").pack(anchor="w")
        ttk.Label(panel, text=f"{config.LOCATION_NAME}\n{config.START_DATE} → {config.END_DATE}"
                              f" · {len(self.df)} hourly samples",
                  foreground="#898781").pack(anchor="w", pady=(0, 12))

        ttk.Label(panel, text="INPUT SIGNALS", style="Section.TLabel").pack(anchor="w")
        names = [f"{config.SIGNAL_INFO[c][0]} [{config.SIGNAL_INFO[c][1]}]" for c in self.columns]
        self.x1_var = tk.StringVar(value=names[0])
        self.x2_var = tk.StringVar(value=names[1])
        ttk.Label(panel, text="x₁[n]").pack(anchor="w", pady=(6, 0))
        self.x1_box = ttk.Combobox(panel, textvariable=self.x1_var, values=names, state="readonly", width=34)
        self.x1_box.pack(anchor="w")
        ttk.Label(panel, text="x₂[n]").pack(anchor="w", pady=(6, 0))
        self.x2_box = ttk.Combobox(panel, textvariable=self.x2_var, values=names, state="readonly", width=34)
        self.x2_box.pack(anchor="w")
        for box in (self.x1_box, self.x2_box):
            box.bind("<<ComboboxSelected>>", lambda e: self.update_plot())

        ttk.Label(panel, text="View window (samples)").pack(anchor="w", pady=(8, 0))
        win = ttk.Frame(panel)
        win.pack(anchor="w")
        self.n_lo = tk.IntVar(value=0)
        self.n_hi = tk.IntVar(value=71)
        for var, txt in ((self.n_lo, "from"), (self.n_hi, "to")):
            ttk.Label(win, text=txt).pack(side=tk.LEFT, padx=(0, 4))
            sp = ttk.Spinbox(win, from_=0, to=len(self.df) - 1, textvariable=var, width=6,
                             command=self.update_plot)
            sp.bind("<Return>", lambda e: self.update_plot())
            sp.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(panel, text="OPERATION", style="Section.TLabel").pack(anchor="w", pady=(12, 4))
        self.op_var = tk.StringVar(value="Addition")
        grid = ttk.Frame(panel)
        grid.pack(anchor="w")
        for i, op in enumerate(ops.OPERATIONS):
            ttk.Radiobutton(grid, text=op, value=op, variable=self.op_var,
                            command=self._on_operation_change).grid(row=i // 2, column=i % 2,
                                                                    sticky="w", padx=(0, 14), pady=1)

        self.eq_label = ttk.Label(panel, text="", style="Eq.TLabel", width=34, anchor="center")
        self.eq_label.pack(anchor="w", pady=(10, 6), fill=tk.X)

        self.param_label = ttk.Label(panel, text="")
        self.param_label.pack(anchor="w")
        self.param_var = tk.DoubleVar()
        self.param_scale = tk.Scale(panel, variable=self.param_var, orient=tk.HORIZONTAL, length=270,
                                    command=lambda v: self.update_plot(), bg="#f6f6f3",
                                    highlightthickness=0, troughcolor="#e1e0d9",
                                    activebackground="#2a78d6", font=("Segoe UI", 9))
        self.param_scale.pack(anchor="w")

        self.stem_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(panel, text="Stem plot (discrete view)", variable=self.stem_var,
                        command=self.update_plot).pack(anchor="w", pady=(8, 0))

        btns = ttk.Frame(panel)
        btns.pack(anchor="w", pady=(14, 0))
        ttk.Button(btns, text="Save figure…", command=self.save_figure).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btns, text="Export y[n] as CSV…", command=self.export_csv).pack(side=tk.LEFT)

    def _build_plot_area(self):
        right = ttk.Frame(self, padding=(6, 10, 14, 10))
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(8.5, 4.6), dpi=100)
        self.ax_in = self.fig.add_subplot(2, 1, 1)
        self.ax_out = self.fig.add_subplot(2, 1, 2)
        self.fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.1, hspace=0.42)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        toolbar = NavigationToolbar2Tk(self.canvas, right, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.table = ttk.Treeview(right, columns=STAT_COLS, show="headings", height=3)
        for col in STAT_COLS:
            self.table.heading(col, text=col.capitalize() if col != "N" else "N")
            self.table.column(col, width=200 if col == "signal" else 90, anchor="center")
        self.table.pack(fill=tk.X, pady=(8, 0))

    # --------------------------------------------------------------- behaviour
    def _selected_signal(self, combo_var, name):
        idx = [f"{config.SIGNAL_INFO[c][0]} [{config.SIGNAL_INFO[c][1]}]" for c in self.columns] \
            .index(combo_var.get())
        col = self.columns[idx]
        return ops.DTSignal(self.df["n"].to_numpy(), self.df[col].to_numpy(), name,
                            config.SIGNAL_INFO[col][1])

    def _on_operation_change(self):
        op = self.op_var.get()
        self.eq_label.configure(text=ops.OPERATIONS[op])
        self.x2_box.configure(state="readonly" if op in TWO_INPUT_OPS else "disabled")
        if op in PARAMS:
            key, label, lo, hi, res, default = PARAMS[op]
            self.param_label.configure(text=label)
            self.param_scale.configure(from_=lo, to=hi, resolution=res, state=tk.NORMAL)
            self.param_var.set(default)
        else:
            self.param_label.configure(text="No parameter for this operation")
            self.param_scale.configure(state=tk.DISABLED)
        self.update_plot()

    def _params(self):
        op = self.op_var.get()
        if op not in PARAMS:
            return {}
        key, *_ = PARAMS[op]
        v = self.param_var.get()
        return {key: v if key == "A" else int(round(v))}

    def _draw(self, ax, s, color, label, lo, hi):
        keep = (s.n >= lo) & (s.n <= hi)
        if self.stem_var.get():
            plots.stem(ax, s.n[keep], s.x[keep], color, label, markersize=3)
        else:
            ax.plot(s.n[keep], s.x[keep], color=color, label=label)

    def update_plot(self):
        op = self.op_var.get()
        x1 = self._selected_signal(self.x1_var, "x₁[n]")
        x2 = self._selected_signal(self.x2_var, "x₂[n]")
        try:
            y = ops.apply_operation(op, x1, x2, self._params())
        except ValueError as exc:
            self.eq_label.configure(text=str(exc))
            return
        self.result = y

        lo, hi = sorted((self.n_lo.get(), self.n_hi.get()))
        # Output window follows the operation (reversal maps n -> -n, decimation n -> n/D, ...)
        if op == "Time reversal":
            out_lo, out_hi = -hi, -lo
        elif op == "Downsampling":
            out_lo, out_hi = lo // self._params()["D"], hi // self._params()["D"]
        elif op == "Upsampling":
            out_lo, out_hi = lo * self._params()["L"], hi * self._params()["L"]
        elif op == "Time shift":
            k = self._params()["k"]
            out_lo, out_hi = min(lo, lo + k), max(hi, hi + k)
        else:
            out_lo, out_hi = lo, hi

        for ax in (self.ax_in, self.ax_out):
            ax.clear()
        self._draw(self.ax_in, x1, plots.C["blue"], f"x₁[n]: {self.x1_var.get()}", lo, hi)
        if op in TWO_INPUT_OPS:
            self._draw(self.ax_in, x2, plots.C["orange"], f"x₂[n]: {self.x2_var.get()}", lo, hi)
        self.ax_in.set_title("Input signal(s)")
        plots.top_legend(self.ax_in, ncol=2)
        if op == "Time shift":
            self._draw(self.ax_out, x1, plots.C["axis"], "x₁[n] (reference)", out_lo, out_hi)
        self._draw(self.ax_out, y, plots.C["violet"], y.name, out_lo, out_hi)
        self.ax_out.set_title(f"Output: {ops.OPERATIONS[op]}")
        plots.top_legend(self.ax_out, ncol=2)
        self.ax_out.set_xlabel("Sample index n  (1 sample = 1 hour)")
        self.ax_out.set_xlim(out_lo - 1, out_hi + 1)
        self.ax_in.set_xlim(lo - 1, hi + 1)
        self.canvas.draw_idle()

        self.table.delete(*self.table.get_children())
        shown = [x1, x2, y] if op in TWO_INPUT_OPS else [x1, y]
        for s in shown:
            st = ops.signal_stats(s)
            self.table.insert("", tk.END, values=[
                st["signal"], st["N"], st["support"], f"{st['mean']:.3f}", f"{st['min']:.3f}",
                f"{st['max']:.3f}", f"{st['energy']:.4g}", f"{st['power']:.4g}"])

    def save_figure(self):
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path:
            self.fig.savefig(path, dpi=200)

    def export_csv(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path and self.result is not None:
            import pandas as pd
            pd.DataFrame({"n": self.result.n, "y": self.result.x}).to_csv(path, index=False)

    # ------------------------------------------------------- report screenshots
    def capture(self, path, delay_ms=1500):
        """Render this window's own pixels into a PNG (used for the report screenshots).

        Uses the Win32 PrintWindow API, so only the app is captured -- never
        whatever happens to be on screen in front of or behind it.
        """
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
            user32.PrintWindow(hwnd, mem, 1 | 2)   # PW_CLIENTONLY | PW_RENDERFULLCONTENT

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
    try:  # crisp rendering on high-DPI Windows displays
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--op", default="Addition", choices=list(ops.OPERATIONS))
    parser.add_argument("--x1", type=int, default=0, help="index of x1 in the signal list")
    parser.add_argument("--x2", type=int, default=1, help="index of x2 in the signal list")
    parser.add_argument("--param", type=float, help="parameter value for the operation")
    parser.add_argument("--stem", action="store_true")
    parser.add_argument("--screenshot", help="save a screenshot to this path and exit")
    args = parser.parse_args()

    app = SignalLabApp()
    app.x1_box.current(args.x1)
    app.x2_box.current(args.x2)
    app.op_var.set(args.op)
    app.stem_var.set(args.stem)
    app._on_operation_change()
    if args.param is not None:
        app.param_var.set(args.param)
        app.update_plot()
    if args.screenshot:
        app.capture(args.screenshot)
    app.mainloop()


if __name__ == "__main__":
    main()
