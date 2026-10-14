"""
run_experiment.py
-----------------
Batch entry point for Experiment 11.

    python run_experiment.py            # run everything, save figures + results
    python run_experiment.py --show     # also open the figures in windows

Outputs (in ``outputs/``):
    figures/*.png        every figure used in the report
    pulse_filter.csv     the designed band-pass coefficients (deployable)
    results.json         every number quoted in the report
"""

import argparse
import json
import sys

import matplotlib
import numpy as np

import config


def summarise(R) -> dict:
    sc = lambda d: {k: v for k, v in d.items() if isinstance(v, (int, float, str, bool))}
    return {"windows": {w: sc(v) for w, v in R["win"].items()}, "kaiser": sc(R["kaiser"]),
            "gains_db": R["F"]["gains"], "application": sc(R["app"]), "methods": R["methods"]}


def main():
    sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles default to cp1252
    parser = argparse.ArgumentParser(description="Experiment 11: FIR filter design by windowing")
    parser.add_argument("--show", action="store_true", help="display figures interactively")
    args = parser.parse_args()
    if not args.show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import analysis
    import plots

    R = analysis.run_all()
    print(f"Loaded {len(R['D']['ppg'])} PPG samples at {config.FS:g} Hz ({config.RECORD})")
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for name, fig in plots.make_all_figures(R).items():
        fig.savefig(config.FIGURE_DIR / name)
        print(f"  saved {name}")
        if not args.show:
            plt.close(fig)
    np.savetxt(config.OUTPUT_DIR / "pulse_filter.csv", R["F"]["pulse"]["b"], delimiter=",")
    with open(config.OUTPUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(summarise(R), f, indent=2, default=float, ensure_ascii=False)
    A = R["app"]
    print(f"  HR {A['hr_mean']:.1f} vs monitor {A['hr_ref_mean']:.1f} bpm (MAE {A['hr_mae']:.2f}); "
          f"RR {A['rr_mean']:.1f} vs annotated {A['rr_ref_mean']:.1f} (MAE {A['rr_mae']:.2f})")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
