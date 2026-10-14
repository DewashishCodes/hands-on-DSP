"""
run_experiment.py
-----------------
Batch entry point for Experiment 6.

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


def _scalars(d):
    return {k: v for k, v in d.items() if isinstance(v, (int, float, str, list, tuple))}


def summarise(R) -> dict:
    return {
        "raw": {s: _scalars(p) for s, p in R["pairs"].items()},
        "tidal": {s: _scalars(p) for s, p in R["tidal"].items()},
        "chain_tidal": R["chain"]["steps"], "chain_raw": R["chain_raw"]["steps"],
        "speed_tidal": R["speed"], "speed_raw": R["speed_raw"],
        "subtidal": {s: _scalars(v) for s, v in R["subtidal"].items() if s != "x"},
        "properties": _scalars(R["props"]),
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles default to cp1252
    parser = argparse.ArgumentParser(description="Experiment 6: cross-correlation")
    parser.add_argument("--show", action="store_true", help="display figures interactively")
    args = parser.parse_args()
    if not args.show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import analysis
    import plots

    R = analysis.run_all()
    print(f"Loaded {len(R['df'])} hourly samples at {len(config.STATIONS)} gauges")
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for name, fig in plots.make_all_figures(R).items():
        fig.savefig(config.FIGURE_DIR / name)
        print(f"  saved {name}")
        if not args.show:
            plt.close(fig)

    summary = summarise(R)
    with open(config.OUTPUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float, ensure_ascii=False)
    for s, p in summary["tidal"].items():
        print(f"  {p['name']:<16} {p['dist_km']:6.1f} km  delay {p['lag_h']:5.2f} h  rho {p['peak']:.3f}")
    print(f"  wave speed {summary['speed_tidal']['c_ms']:.2f} m/s, depth {summary['speed_tidal']['depth_m']:.2f} m")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
