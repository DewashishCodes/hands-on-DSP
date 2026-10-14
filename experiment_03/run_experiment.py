"""
run_experiment.py
-----------------
Batch entry point for Experiment 3.

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
    return {
        "bandwidth_hz": R["clip"]["bandwidth"],
        "peak_freq_hz": R["clip"]["peak_freq"],
        "energy_above_nyquist": {str(k): v for k, v in R["clip"]["above"].items()},
        "prefilter_snr_db": {str(r): {k: v["snr"] for k, v in row.items()} for r, row in R["prefilter"].items()},
        "recon_snr_db": {str(r): {m: row[m]["snr"] for m in config.RECON_METHODS} for r, row in R["recon"].items()},
        "moving_average": {str(M): {k: v for k, v in d.items() if k not in ("y",)}
                           for M, d in R["ma"].items() if isinstance(M, int)},
        "storage_mb_per_hour": {str(k): v for k, v in R["storage_mb_per_h"].items()},
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles default to cp1252
    parser = argparse.ArgumentParser(description="Experiment 3: moving average, sampling and reconstruction")
    parser.add_argument("--show", action="store_true", help="display figures interactively")
    args = parser.parse_args()
    if not args.show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import analysis
    import plots

    R = analysis.run_all()
    print(f"Loaded {len(R['x'])} samples at {R['fs']} Hz from {config.CLIP_FILE.name}")

    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for name, fig in plots.make_all_figures(R).items():
        fig.savefig(config.FIGURE_DIR / name)
        print(f"  saved {name}")
        if not args.show:
            plt.close(fig)

    summary = summarise(R)
    with open(config.OUTPUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary["prefilter_snr_db"], indent=1))
    print(json.dumps(summary["recon_snr_db"], indent=1))

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
