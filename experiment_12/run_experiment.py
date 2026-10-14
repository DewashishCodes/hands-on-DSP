"""
run_experiment.py
-----------------
Batch entry point for Experiment 12.

    python run_experiment.py            # run everything, save figures + results
    python run_experiment.py --show     # also open the figures in windows

Outputs (in ``outputs/``):
    figures/*.png        every figure used in the report
    alpha_sos.csv        the Butterworth alpha band-pass as second-order sections
    results.json         every number quoted in the report
"""

import argparse
import json
import sys
import warnings

import matplotlib
import numpy as np

import config


def summarise(R) -> dict:
    sc = lambda d: {k: v for k, v in d.items()
                    if isinstance(v, (int, float, str, bool, tuple)) and k not in ("zpk", "ba")}
    return {"designs": {k: sc(v) for k, v in R["designs"].items()}, "fir_equivalent": R["fir_eq"],
            "prewarp": R["prewarp"], "numerics": R["numerics"], "notch": sc(R["notch"]),
            "phase": sc(R["phase"]), "alpha": {k: v for k, v in R["alpha"].items() if isinstance(v, float)}}


def main():
    sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles default to cp1252
    warnings.filterwarnings("ignore", category=RuntimeWarning)   # the deliberately unstable demo overflows
    parser = argparse.ArgumentParser(description="Experiment 12: IIR filter design")
    parser.add_argument("--show", action="store_true", help="display figures interactively")
    args = parser.parse_args()
    if not args.show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import analysis
    import plots

    R = analysis.run_all()
    print(f"Loaded channel {config.CHANNEL}: {len(R['D']['open']['x'])} samples per run at {R['D']['fs']:g} Hz")
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for name, fig in plots.make_all_figures(R).items():
        fig.savefig(config.FIGURE_DIR / name)
        print(f"  saved {name}")
        if not args.show:
            plt.close(fig)
    np.savetxt(config.OUTPUT_DIR / "alpha_sos.csv", R["designs"]["butterworth"]["sos"], delimiter=",")
    with open(config.OUTPUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(summarise(R), f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o),
                  ensure_ascii=False)
    print(f"  alpha power eyes closed / open = {R['alpha']['ratio']:.1f}  (t = {R['alpha']['t']:.1f})")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
