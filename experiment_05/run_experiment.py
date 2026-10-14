"""
run_experiment.py
-----------------
Batch entry point for Experiment 5.

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


def summarise(R) -> dict:
    A = R["acf"]
    return {
        "info": R["info"],
        "acf": {k: v for k, v in A.items() if k != "rho"},
        "methods": {k: v for k, v in R["methods"].items() if k != "full"},
        "noise": {k: {a: b for a, b in v.items() if a != "rho"} for k, v in R["noise"].items() if k != "smooth"},
        "windows": [{"span": w["span"], "period_years": w["period_years"], "peak": w["peak"]} for w in R["windows"]],
        "robust": [{"level": r["level"], "snr_db": r["snr_db"], "period_years": r["period_years"],
                    "peak": r["peak"]} for r in R["robust"]],
        "forecast": R["forecast"],
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles default to cp1252
    parser = argparse.ArgumentParser(description="Experiment 5: autocorrelation")
    parser.add_argument("--show", action="store_true", help="display figures interactively")
    args = parser.parse_args()
    if not args.show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import analysis
    import plots

    R = analysis.run_all()
    print(f"Loaded {R['info']['months']} months ({R['info']['start']} to {R['info']['end']})")
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for name, fig in plots.make_all_figures(R).items():
        fig.savefig(config.FIGURE_DIR / name)
        print(f"  saved {name}")
        if not args.show:
            plt.close(fig)

    summary = summarise(R)
    with open(config.OUTPUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float, ensure_ascii=False)
    print(f"Dominant period: {summary['acf']['period_years']:.2f} years (rho = {summary['acf']['peak_value']:.3f})")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
