"""
run_experiment.py
-----------------
Batch entry point for Experiment 1.

    python run_experiment.py            # run everything, save figures + results
    python run_experiment.py --show     # also open the figures in windows

Outputs (in ``outputs/``):
    figures/*.png        every figure used in the report
    results.json         every number quoted in the report
    signal_stats.csv     statistics table of all input / output signals
"""

import argparse
import json
import sys

import matplotlib

import config


def summarise(results) -> dict:
    """Pull the scalar results out of the analysis dictionaries (JSON-serialisable)."""
    out = {}
    for section, values in results.items():
        out[section] = {k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in values.items()
                        if isinstance(v, (int, float, str, list)) and not isinstance(v, bool)}
    return out


def stats_table(results):
    import pandas as pd
    import signal_ops as ops

    rows = [
        ("Addition", results["addition"]["ghi"]),
        ("Subtraction", results["subtraction"]["dpd"]),
        ("Multiplication", results["multiplication"]["power"]),
        ("Amplitude scaling", results["scaling"]["scaled"]),
        ("Time shift (delay)", results["shift"]["forecast"]),
        ("Time reversal", results["reversal"]["folded"]),
        ("Downsampling", results["time_scaling"]["down"]),
        ("Upsampling", results["time_scaling"]["up"]),
        ("Convolution", results["convolution"]["y"]),
    ]
    table = pd.DataFrame([{"operation": op, **ops.signal_stats(s)} for op, s in rows])
    return table


def main():
    sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles default to cp1252
    parser = argparse.ArgumentParser(description="Experiment 1: basic DT signal operations")
    parser.add_argument("--show", action="store_true", help="display figures interactively")
    args = parser.parse_args()
    if not args.show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import analysis
    import data_loader
    import plots

    df = data_loader.load_dataset()
    print(f"Loaded {len(df)} samples from {config.DATA_FILE.name}")

    results = analysis.run_all(df)

    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for name, fig in plots.make_all_figures(df, results).items():
        fig.savefig(config.FIGURE_DIR / name)
        print(f"  saved {name}")
        if not args.show:
            plt.close(fig)

    summary = summarise(results)
    summary["multiplication"]["daily_kwh"] = {
        str(d): {"ideal": round(r.ideal, 4), "actual": round(r.actual, 4)}
        for d, r in results["multiplication"]["daily_kwh"].iterrows()}
    summary["time_scaling"]["daily_max"] = {
        str(d): {"true": float(t), "synoptic": float(s)}
        for d, t, s in zip(results["time_scaling"]["daily_max_true"].index,
                           results["time_scaling"]["daily_max_true"],
                           results["time_scaling"]["daily_max_synoptic"])}
    summary["dataset"] = data_loader.describe_dataset(df).round(3).reset_index() \
        .rename(columns={"index": "column"}).to_dict(orient="records")
    with open(config.OUTPUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str, ensure_ascii=False)

    table = stats_table(results)
    table.to_csv(config.OUTPUT_DIR / "signal_stats.csv", index=False)
    print("\n" + table.round(3).to_string(index=False))

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
