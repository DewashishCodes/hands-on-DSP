"""
run_experiment.py
-----------------
Batch entry point for Experiment 8.

    python run_experiment.py            # run everything, save figures + results
    python run_experiment.py --show     # also open the figures in windows

Outputs (in ``outputs/``):
    figures/*.png        every figure used in the report
    results.json         every number quoted in the report
"""

import argparse
import json
import sys

import matplotlib

import config


def _scal(d):
    return {k: v for k, v in d.items() if isinstance(v, (int, float, str, list, dict)) and not hasattr(v, "shape")}


def summarise(R) -> dict:
    return {
        "info": R["info"], "methods": R["methods"],
        "spectrum": {k: v for k, v in _scal(R["spec"]).items()},
        "leakage": _scal(R["leak"]), "seasonal": _scal(R["season"]),
        "decades": R["decades"], "growth": R["growth"]["by_epoch"], "timing": R["timing"],
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles default to cp1252
    parser = argparse.ArgumentParser(description="Experiment 8: DFT and IDFT")
    parser.add_argument("--show", action="store_true", help="display figures interactively")
    args = parser.parse_args()
    if not args.show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import analysis
    import plots

    R = analysis.run_all()
    print(f"Loaded {R['info']['months']} months; DFT length N = {R['info']['N']} ({R['info']['years']} whole years)")
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for name, fig in plots.make_all_figures(R).items():
        fig.savefig(config.FIGURE_DIR / name)
        print(f"  saved {name}")
        if not args.show:
            plt.close(fig)

    summary = summarise(R)
    with open(config.OUTPUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float, ensure_ascii=False)
    A1 = R["spec"]["lines"][1][0]
    print(f"  annual line {A1:.3f} ppm, peak {R['season']['max_month']}, trough {R['season']['min_month']}, "
          f"RMSE vs NOAA {R['season']['rmse_noaa']:.3f} ppm")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
