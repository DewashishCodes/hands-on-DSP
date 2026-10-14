"""
run_experiment.py
-----------------
Batch entry point for Experiment 10.

    python run_experiment.py            # run everything, save figures + results
    python run_experiment.py --show     # also open the figures in windows

Outputs (in ``outputs/``):
    figures/*.png        every figure used in the report
    normals.csv          the smoothed daily normals (a usable product)
    results.json         every number quoted in the report
"""

import argparse
import json
import sys

import matplotlib
import pandas as pd

import config


def summarise(R) -> dict:
    sm = {k: {a: b for a, b in v.items() if not hasattr(b, "shape")} for k, v in R["smooth"].items()}
    return {"info": {k: str(v) for k, v in R["info"].items()}, "smoothing": sm,
            "noise_expected": R["noise_expected"],
            "methods": {k: v for k, v in R["methods"].items() if not hasattr(v, "shape")},
            "edge": R["edge_rows"], "timing": R["timing"], "scaling": R["scaling"]}


def main():
    sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles default to cp1252
    parser = argparse.ArgumentParser(description="Experiment 10: circular convolution using the DFT")
    parser.add_argument("--show", action="store_true", help="display figures interactively")
    args = parser.parse_args()
    if not args.show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import analysis
    import plots

    R = analysis.run_all()
    print(f"Loaded {R['info']['rows']} daily rows for {config.STATION_NAME}")
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for name, fig in plots.make_all_figures(R).items():
        fig.savefig(config.FIGURE_DIR / name)
        print(f"  saved {name}")
        if not args.show:
            plt.close(fig)

    box = R["smooth"][f"Boxcar {config.MA_DAYS} d"]
    pd.DataFrame({"date": R["clim"]["date"].dt.strftime("%d-%b"), "tmax_normal": box["y"].round(2),
                  "tmin_normal": R["tmin_s"].round(2)}).to_csv(config.OUTPUT_DIR / "normals.csv", index=False)
    with open(config.OUTPUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(summarise(R), f, indent=2, default=float, ensure_ascii=False)
    print(f"  circular via DFT vs direct: {R['methods']['direct_vs_fft']:.1e};  "
          f"linear edge error up to {R['edge_rows'][2]['max_err']:.1f} °C")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
