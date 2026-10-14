"""
run_experiment.py
-----------------
Batch entry point for Experiment 7.

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
    keep = ("max_pole", "stable", "noise_gain", "lag", "lag_measured", "roughness", "rmse", "ops", "lfilter_err")
    return {
        "info": {k: str(v) for k, v in R["info"].items()},
        "filters": {n: {k: v for k, v in f.items() if k in keep} for n, f in R["filters"].items()},
        "tradeoff": R["tradeoff"].to_dict(orient="records"),
        "initial_conditions": {k: v for k, v in R["ic"].items() if not hasattr(v, "shape")},
        "macd": {k: v for k, v in R["macd"].items() if isinstance(v, (int, float))},
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles default to cp1252
    parser = argparse.ArgumentParser(description="Experiment 7: difference-equation smoothing")
    parser.add_argument("--show", action="store_true", help="display figures interactively")
    args = parser.parse_args()
    if not args.show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import analysis
    import plots

    R = analysis.run_all()
    print(f"Loaded {R['info']['trading_days']} trading days of {config.SERIES}")
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for name, fig in plots.make_all_figures(R).items():
        fig.savefig(config.FIGURE_DIR / name)
        print(f"  saved {name}")
        if not args.show:
            plt.close(fig)

    summary = summarise(R)
    with open(config.OUTPUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float, ensure_ascii=False)
    for n, f_ in summary["filters"].items():
        print(f"  {n:<26} lag {f_['lag']:5.2f}  ramp lag {f_['lag_measured']:5.2f}  noise gain {f_['noise_gain']:.4f}  ops {f_['ops']}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
