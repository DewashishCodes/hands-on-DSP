"""
run_experiment.py
-----------------
Batch entry point for Experiment 2.

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
    out = {"battery": {}, "responses": {}, "application": {}}
    for key, res in R["battery"].items():
        s = res["system"]
        out["battery"][key] = {
            "system": s.name, "equation": s.equation,
            "linearity_error": res["linear"]["error"], "linear": res["linear"]["holds"],
            "ti_error": res["ti"]["error"], "time_invariant": res["ti"]["holds"],
            "lti": res["lti"], "matches_expectation": res["matches_expectation"],
        }
    for key, res in R["responses"].items():
        out["responses"][key] = {k: (float(v) if not hasattr(v, "__len__") else None)
                                 for k, v in res.items() if not hasattr(v, "__len__")}
    out["application"] = {k: (v if isinstance(v, (int, float, str, dict)) else str(v))
                          for k, v in R["application"].items() if not hasattr(v, "shape")}
    out["aqi_examples"] = R["aqi_examples"]
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles default to cp1252
    parser = argparse.ArgumentParser(description="Experiment 2: LTI system tests")
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
    R = analysis.run_all(df)

    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for name, fig in plots.make_all_figures(df, R).items():
        fig.savefig(config.FIGURE_DIR / name)
        print(f"  saved {name}")
        if not args.show:
            plt.close(fig)

    summary = summarise(R)
    with open(config.OUTPUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str, ensure_ascii=False)

    print("\nSystem                  linearity error   TI error    verdict")
    for key, b in summary["battery"].items():
        verdict = "LTI" if b["lti"] else ("nonlinear" if not b["linear"] else "time-varying")
        print(f"{key} {b['system']:<22} {b['linearity_error']:>12.2e}   {b['ti_error']:>9.2e}   {verdict}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
