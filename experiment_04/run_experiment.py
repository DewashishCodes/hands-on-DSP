"""
run_experiment.py
-----------------
Batch entry point for Experiment 4.

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
    sm = R["smoothing"]
    return {
        "info": {k: str(v) for k, v in R["info"].items()},
        "seasons": {k: {"mode_deg": v["mode_deg"], "n": v["n"], "speed_mean": v["speed_mean"]}
                    for k, v in R["seasons"].items()},
        "smoothing": {k: sm[k] for k in ("sum_p", "sum_circ", "sum_lin", "seam_err", "seam_bins", "interior_err")},
        "methods": R["methods"],
        "lin_circ": {k: R["lin_circ"][k] for k in ("wrap_ok", "pad_ok")},
        "runway": {str(lim): {k: v for k, v in d.items() if k != "coverage"}
                   for lim, d in R["runway"].items() if isinstance(lim, float)},
        "runway_brute_force_err": R["runway"]["brute_force_err"],
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles default to cp1252
    parser = argparse.ArgumentParser(description="Experiment 4: circular convolution")
    parser.add_argument("--show", action="store_true", help="display figures interactively")
    args = parser.parse_args()
    if not args.show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import analysis
    import plots

    R = analysis.run_all()
    print(f"Loaded {R['info']['reports']} METAR reports for {config.STATION} {config.YEAR}")
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for name, fig in plots.make_all_figures(R).items():
        fig.savefig(config.FIGURE_DIR / name)
        print(f"  saved {name}")
        if not args.show:
            plt.close(fig)

    summary = summarise(R)
    with open(config.OUTPUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary["runway"], indent=1))

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
