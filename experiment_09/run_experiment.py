"""
run_experiment.py
-----------------
Batch entry point for Experiment 9.

    python run_experiment.py            # run everything, save figures + results
    python run_experiment.py --show     # also open the figures in windows

Outputs (in ``outputs/``):
    figures/*.png        every figure used in the report
    reverberant.wav      the convolution result (listen to it)
    results.json         every number quoted in the report
"""

import argparse
import json
import sys

import matplotlib
import numpy as np
from scipy.io import wavfile

import config


def summarise(R) -> dict:
    A = {k: v for k, v in R["acc"].items() if k != "y"}
    room = {k: v for k, v in R["room"].items() if k in ("T20", "T30", "EDT", "DRR", "peak_ms")}
    dec = {"rows": R["deconv"]["rows"], "wet_sisnr": R["deconv"]["wet_sisnr"]}
    return {"files": R["desc"], "accuracy_speed": A, "aliasing": R["alias"]["rows"], "ola": R["ola"],
            "room": room, "deconvolution": dec}


def main():
    sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles default to cp1252
    parser = argparse.ArgumentParser(description="Experiment 9: linear convolution using the DFT")
    parser.add_argument("--show", action="store_true", help="display figures interactively")
    args = parser.parse_args()
    if not args.show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import analysis
    import plots

    R = analysis.run_all()
    A = R["acc"]
    print(f"L = {A['L']}, M = {A['M']}, N = L + M - 1 = {A['Nlin']}")
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for name, fig in plots.make_all_figures(R).items():
        fig.savefig(config.FIGURE_DIR / name)
        print(f"  saved {name}")
        if not args.show:
            plt.close(fig)

    y = A["y"] / np.max(np.abs(A["y"])) * 0.9
    wavfile.write(config.OUTPUT_DIR / "reverberant.wav", R["fs"], (y * 32767).astype(np.int16))
    with open(config.OUTPUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(summarise(R), f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o),
                  ensure_ascii=False)
    print(f"  DFT vs published: {A['snr_vs_published']:.1f} dB;  speed-up {A['speedup']:.0f}x;  "
          f"RT60 (T20) {R['room']['T20']:.2f} s")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
