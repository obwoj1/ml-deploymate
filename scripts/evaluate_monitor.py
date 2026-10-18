"""Evaluate the monitor: precision, recall, F1, false-alarm rate, detection delay.

    python scripts/evaluate_monitor.py            # 10 runs per scenario (~2 min)
    python scripts/evaluate_monitor.py --runs 3   # quicker

Saves assets/monitor_results.json. Needs the trained model (scripts/train.py).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from deploymate.data.load import load_dataset  # noqa: E402
from deploymate.monitor.evaluate import evaluate  # noqa: E402
from deploymate.monitor.pipeline import Monitor  # noqa: E402
from deploymate.simulate.scenarios import simulate  # noqa: E402

CALIBRATION_SEED = 100_000  # calibration batches; evaluation seeds are 1000*scenario + run, so they never overlap


def fmt(v, pct=False):
    if v is None:
        return "-"
    return f"{v:.1%}" if pct else f"{v:.3f}" if isinstance(v, float) else str(v)


def main(runs: int) -> None:
    start = time.time()
    model_path = ROOT / "models" / "model.joblib"
    if not model_path.exists():
        sys.exit("No trained model. Run: python scripts/train.py")
    X_train, X_test, _, y_test = load_dataset()
    model = joblib.load(model_path)

    print("Calibrating on 200 healthy batches...")
    healthy = [b.X for b in simulate("healthy", X_test, y_test, n_batches=200, seed=CALIBRATION_SEED)]
    monitor = Monitor.from_data(model, X_train.sample(5000, random_state=0), healthy)

    print(f"Running every scenario {runs} times (40 batches x 400 rows each)...")
    results = evaluate(monitor, X_test, y_test, seeds=range(runs))

    combined = results["combined"]
    print(f"\nDeployMate monitor (combined), {runs} runs per scenario\n")
    print(f"{'scenario':<16}{'precision':>10}{'recall':>8}{'F1':>7}{'false alarms':>14}{'delay':>7}{'detected':>10}")
    for scenario, s in combined.items():
        print(f"{scenario:<16}{fmt(s['precision']):>10}{fmt(s['recall']):>8}{fmt(s['f1']):>7}"
              f"{fmt(s['false_alarm_rate'], pct=True):>14}{fmt(s['mean_detection_delay']):>7}{s['runs_detected']:>10}")

    print("\nWhat each half contributes (overall)\n")
    print(f"{'detector':<14}{'precision':>10}{'recall':>8}{'F1':>7}{'false alarms':>14}")
    for name, per in results.items():
        o = per["overall"]
        print(f"{name:<14}{fmt(o['precision']):>10}{fmt(o['recall']):>8}{fmt(o['f1']):>7}"
              f"{fmt(o['false_alarm_rate'], pct=True):>14}")

    out = ROOT / "assets" / "monitor_results.json"
    out.write_text(json.dumps({"runs_per_scenario": runs, "batches_per_run": 40,
                               "batch_size": 400, "results": results}, indent=2))
    print(f"\nSaved assets/monitor_results.json ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the DeployMate monitor.")
    parser.add_argument("--runs", type=int, default=10, help="runs per scenario")
    main(parser.parse_args().runs)
