"""
gui.py
------
Interactive Tkinter "system tester" for Experiment 2.

    python gui.py

Pick a system (or type your own difference equation), pick a test, adjust the
parameters, and the GUI shows both sides of the test, the error signal and a
verdict.  All processing is delegated to ``systems``.
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
import systems as sy

CUSTOM = "Custom difference equation"
TESTS = ["Linearity (superposition)", "Time invariance (shift)", "Impulse and step response",
         "Apply to PM2.5 data"]
# parameter slider for each built-in system: (key, label, from, to, resolution)
SYS_PARAM = {
    "S1": ("M", "Window length M (hours)", 1, 72, 1),
    "S2": ("alpha", "Smoothing factor α", 0.02, 1.0, 0.01),
    "S4": ("beta", "Drift β (per hour)", 0.0, 0.01, 0.0002),
}


class SystemTesterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DSP Lab · Experiment 2 · Is the System LTI?")
        self.geometry("1280x700+10+10")
        self.minsize(1100, 640)
        self.configure(bg="#f6f6f3")
        self.df = data_loader.load_dataset()
        self.x1 = self.df["pm25"].to_numpy()
        self.x2 = self.df["no2"].to_numpy()
        self.systems = sy.default_systems()
        self._build_style()
        self._build_controls()
        self._build_plot_area()
        self._on_system_change()

    # ------------------------------------------------------------------ layout
    def _build_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background="#f6f6f3", font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI Semibold", 13))
        style.configure("Section.TLabel", font=("Segoe UI Semibold", 10), foreground="#52514e")
        style.configure("Eq.TLabel", font=("Cambria Math", 11), foreground="#2a78d6",
                        background="#ffffff", padding=(8, 4))
        style.configure("Pass.TLabel", font=("Segoe UI Semibold", 12), foreground="#008300",
                        background="#ffffff", padding=(8, 4))
        style.configure("Fail.TLabel", font=("Segoe UI Semibold", 12), foreground="#c62828",
                        background="#ffffff", padding=(8, 4))

    def _slider(self, parent, label, var, lo, hi, res):
        ttk.Label(parent, text=label).pack(anchor="w")
        s = tk.Scale(parent, variable=var, orient=tk.HORIZONTAL, length=280, from_=lo, to=hi,
                     resolution=res, command=lambda v: self.update_plot(), bg="#f6f6f3",
                     highlightthickness=0, troughcolor="#e1e0d9", font=("Segoe UI", 8))
        s.pack(anchor="w")
        return s

    def _build_controls(self):
        panel = ttk.Frame(self, padding=(16, 12))
        panel.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(panel, text="System Tester", style="Header.TLabel").pack(anchor="w")
        ttk.Label(panel, text=f"Inputs: x₁ = PM2.5, x₂ = NO₂ ({len(self.df)} hourly samples)",
                  foreground="#898781").pack(anchor="w", pady=(0, 8))

        ttk.Label(panel, text="SYSTEM", style="Section.TLabel").pack(anchor="w")
        names = [f"{k}: {s.name}" for k, s in self.systems.items()] + [CUSTOM]
        self.sys_var = tk.StringVar(value=names[0])
        box = ttk.Combobox(panel, textvariable=self.sys_var, values=names, state="readonly", width=36)
        box.pack(anchor="w", pady=(2, 4))
        box.bind("<<ComboboxSelected>>", lambda e: self._on_system_change())
        self.eq_label = ttk.Label(panel, text="", style="Eq.TLabel", width=38, anchor="w")
        self.eq_label.pack(anchor="w", fill=tk.X, pady=(0, 4))

        self.custom = ttk.Frame(panel)
        self.b_var = tk.StringVar(value="0.5, 0.5")
        self.a_var = tk.StringVar(value="1, -0.3")
        for txt, var in (("b coefficients (x terms)", self.b_var), ("a coefficients (y terms)", self.a_var)):
            ttk.Label(self.custom, text=txt).pack(anchor="w")
            e = ttk.Entry(self.custom, textvariable=var, width=30)
            e.pack(anchor="w")
            e.bind("<Return>", lambda ev: self.update_plot())
        self.param_var = tk.DoubleVar()
        self.param_frame = ttk.Frame(panel)
        self.param_frame.pack(anchor="w")

        ttk.Label(panel, text="TEST", style="Section.TLabel").pack(anchor="w", pady=(8, 2))
        self.test_var = tk.StringVar(value=TESTS[0])
        for t in TESTS:
            ttk.Radiobutton(panel, text=t, value=t, variable=self.test_var,
                            command=self.update_plot).pack(anchor="w")

        tp = ttk.Frame(panel)
        tp.pack(anchor="w", pady=(6, 0))
        self.a_w, self.b_w = tk.DoubleVar(value=config.TEST_A), tk.DoubleVar(value=config.TEST_B)
        self.k_var = tk.IntVar(value=config.TEST_SHIFT_K)
        row = ttk.Frame(tp)
        row.pack(anchor="w")
        for txt, var in (("a", self.a_w), ("b", self.b_w)):
            ttk.Label(row, text=txt).pack(side=tk.LEFT, padx=(0, 4))
            sp = ttk.Spinbox(row, from_=-5, to=5, increment=0.5, textvariable=var, width=5,
                             command=self.update_plot)
            sp.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(row, text="shift k").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Spinbox(row, from_=0, to=72, textvariable=self.k_var, width=5,
                    command=self.update_plot).pack(side=tk.LEFT)

        self.verdict = ttk.Label(panel, text="", style="Pass.TLabel", width=38, anchor="w")
        self.verdict.pack(anchor="w", fill=tk.X, pady=(10, 4))
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

    # --------------------------------------------------------------- behaviour
    def _key(self):
        v = self.sys_var.get()
        return "custom" if v == CUSTOM else v.split(":")[0]

    def _on_system_change(self):
        for w in self.param_frame.winfo_children():
            w.destroy()
        key = self._key()
        if key == "custom":
            self.custom.pack(anchor="w", before=self.param_frame)
            self.eq_label.configure(text="Σ aₖ y[n−k] = Σ bₖ x[n−k]")
        else:
            self.custom.pack_forget()
            s = self.systems[key]
            self.eq_label.configure(text=s.equation)
            if key in SYS_PARAM:
                pkey, label, lo, hi, res = SYS_PARAM[key]
                self.param_var.set(s.params[pkey])
                self._slider(self.param_frame, label, self.param_var, lo, hi, res)
        self.update_plot()

    def _system(self):
        key = self._key()
        if key == "custom":
            b = [float(v) for v in self.b_var.get().split(",") if v.strip()]
            a = [float(v) for v in self.a_var.get().split(",") if v.strip()]
            return sy.System("Custom", sy.difference_equation, "", "", True, True, {"b": b, "a": a})
        s = self.systems[key]
        if key in SYS_PARAM:
            pkey = SYS_PARAM[key][0]
            val = self.param_var.get()
            s.params[pkey] = int(round(val)) if pkey == "M" else val
        return s

    def update_plot(self):
        try:
            s = self._system()
        except ValueError:
            self.verdict.configure(text="Invalid coefficients", style="Fail.TLabel")
            return
        test = self.test_var.get()
        a1, a2 = self.ax1, self.ax2
        a1.clear()
        a2.clear()
        C = plots.C
        if test.startswith("Linearity"):
            r = sy.linearity_test(s, self.x1, self.x2, self.a_w.get(), self.b_w.get())
            n = np.arange(len(r["lhs"]))
            a1.plot(n, r["lhs"], color=C["blue"], lw=1.8, label="T{a x₁ + b x₂}")
            a1.plot(n, r["rhs"], color=C["orange"], ls="--", label="a T{x₁} + b T{x₂}")
            a2.plot(n, r["lhs"] - r["rhs"], color=C["aqua"] if r["holds"] else C["red"])
            a1.set_title("Both sides of the superposition test")
            a2.set_title("Error signal  e_L[n]")
            self._verdict("LINEAR" if r["holds"] else "NONLINEAR", r["error"], r["holds"])
        elif test.startswith("Time"):
            r = sy.time_invariance_test(s, self.x1, self.k_var.get())
            n = np.arange(len(r["lhs"]))
            a1.plot(n, r["lhs"], color=C["blue"], lw=1.8, label=f"T{{x[n − {r['k']}]}}")
            a1.plot(n, r["rhs"], color=C["orange"], ls="--", label=f"y[n − {r['k']}]")
            a2.plot(n, r["lhs"] - r["rhs"], color=C["aqua"] if r["holds"] else C["red"])
            a1.set_title("Response to the shifted input vs. shifted response")
            a2.set_title("Error signal  e_T[n]")
            self._verdict("TIME-INVARIANT" if r["holds"] else "TIME-VARYING", r["error"], r["holds"])
        elif test.startswith("Impulse"):
            L = config.RESPONSE_LENGTH
            plots.stem(a1, np.arange(L), sy.impulse_response(s, L), C["violet"], "h[n]", markersize=3)
            a2.plot(np.arange(L), sy.step_response(s, L), color=C["blue"], label="s[n]")
            a1.set_title("Impulse response  h[n] = T{δ[n]}")
            a2.set_title("Step response  s[n] = T{u[n]}")
            dc = float(np.sum(sy.impulse_response(s, 2000)))
            self.verdict.configure(text=f"DC gain Σh[n] = {dc:.4f}", style="Pass.TLabel")
        else:
            n = np.arange(len(self.x1))
            a1.plot(n, self.x1, color=C["ink2"], lw=0.9, label="x₁[n]  PM2.5")
            a1.plot(n, s(self.x1), color=C["blue"], lw=1.6, label="y[n]")
            a1.axhline(config.NAAQS_PM25_24H, color=C["red"], lw=0.8, ls="--")
            a1.set_title("System applied to the real PM2.5 signal")
            h = sy.impulse_response(s, len(self.x1))
            err = sy.relative_error(np.convolve(self.x1, h)[: len(n)], s(self.x1))
            a2.plot(n, np.convolve(self.x1, h)[: len(n)] - s(self.x1), color=C["aqua"] if err < 1e-9 else C["red"])
            a2.set_title("x[n] * h[n] minus y[n]  (zero only for an LTI system)")
            self._verdict("y = x * h  (LTI)" if err < 1e-9 else "y ≠ x * h", err, err < 1e-9)
        a2.set_xlabel("Sample index n  (1 sample = 1 hour)")
        for ax in (a1, a2):
            if ax.get_legend_handles_labels()[0]:
                plots.top_legend(ax, ncol=2)
        self.canvas.draw_idle()

    def _verdict(self, text, err, ok):
        self.verdict.configure(text=f"{text}   (ε = {err:.2e})", style="Pass.TLabel" if ok else "Fail.TLabel")

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
    parser.add_argument("--system", type=int, default=0, help="index in the system list (4 = custom)")
    parser.add_argument("--test", type=int, default=0, help="index in the test list")
    parser.add_argument("--screenshot", help="save a screenshot to this path and exit")
    args = parser.parse_args()

    app = SystemTesterApp()
    names = [f"{k}: {s.name}" for k, s in app.systems.items()] + [CUSTOM]
    app.sys_var.set(names[args.system])
    app.test_var.set(TESTS[args.test])
    app._on_system_change()
    if args.screenshot:
        app.capture(args.screenshot)
    app.mainloop()


if __name__ == "__main__":
    main()
