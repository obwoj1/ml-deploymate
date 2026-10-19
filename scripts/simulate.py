"""Watch the full monitor work through a drift scenario, batch by batch.

    python scripts/simulate.py sudden_shift
    python scripts/simulate.py --all          # one-line summary per scenario
    python scripts/simulate.py sudden_shift --explain   # + AI explanation of the first alert

For every batch: was it really broken (truth), what the drift tests flagged,
what the anomaly detector flagged, and the final alert + severity.
Needs the trained model: run scripts/train.py first.
"""

import argparse
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from deploymate.data.load import load_dataset  # noqa: E402
from deploymate.llm.explain import explain_alert  # noqa: E402
from deploymate.monitor.pipeline import Monitor  # noqa: E402
from deploymate.simulate.scenarios import SCENARIOS, simulate  # noqa: E402

CALIBRATION_SEED = 100_000  # healthy batches used to learn "normal" (same as evaluate_monitor.py)


def build_monitor():
    model_path = ROOT / "models" / "model.joblib"
    if not model_path.exists():
        sys.exit("No trained model. Run: python scripts/train.py")
    X_train, X_test, _, y_test = load_dataset()
    model = joblib.load(model_path)
    print("Calibrating the monitor on 200 healthy batches...")
    healthy = [b.X for b in simulate("healthy", X_test, y_test, n_batches=200, seed=CALIBRATION_SEED)]
    monitor = Monitor.from_data(model, X_train.sample(5000, random_state=0), healthy)
    return monitor, X_test, y_test


def watch(name, monitor, X_test, y_test, explain=False):
    print(f"\nScenario: {name}: {SCENARIOS[name][0]}\n")
    print(f"{'batch':>5}  {'truth':<7}{'accuracy':>9}{'confidence':>11}  {'ALERT':<7}"
          f"{'severity':<9}drift / anomaly")
    first_alert = None
    for b in simulate(name, X_test, y_test):
        r = monitor.check(b.X, b.y, batch_id=b.batch_id)
        if r["alert"] and first_alert is None:
            first_alert = r
        a = r["anomaly"]
        reasons = r["drifted_features"] + [f"{m}*" for m in a["flagged_metrics"]]
        if a["isolation_forest"]:
            reasons.append("isolation-forest*")
        print(f"{b.batch_id:>5}  {'BROKEN' if b.drifted else 'ok':<7}"
              f"{r['metrics']['accuracy']:>9.3f}{r['metrics']['mean_confidence']:>11.3f}  "
              f"{'YES' if r['alert'] else '-':<7}{r['severity']:<9}{', '.join(reasons) or '-'}")
    print("\n* = anomaly detector (info only, doesn't trigger alerts); others = drift tests")

    if explain and first_alert:
        e = explain_alert(first_alert)
        print(f"\nExplanation of the first alert (batch {first_alert['batch_id']}, source: {e['source']})")
        print(f"  What happened:    {e['summary']}")
        print(f"  Likely cause:     {e['likely_cause']}")
        print(f"  Suggested action: {e['suggested_action']}")


def summary(monitor, X_test, y_test):
    print(f"\n{'scenario':<16}{'broken':>7}{'drift':>7}{'anomaly':>9}{'caught':>8}{'false alarms':>14}")
    for name in SCENARIOS:
        broken = drift = anomaly = caught = false = 0
        for b in simulate(name, X_test, y_test):
            r = monitor.check(b.X)
            broken += b.drifted
            drift += bool(r["drifted_features"]) and b.drifted
            anomaly += r["anomaly"]["anomalous"] and b.drifted
            caught += r["alert"] and b.drifted
            false += r["alert"] and not b.drifted
        print(f"{name:<16}{broken:>7}{drift:>7}{anomaly:>9}{caught:>8}{false:>14}")
    print("\ndrift / anomaly = broken batches each part flagged on its own; caught = alerts (drift tests)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch the monitor work through drift scenarios.")
    parser.add_argument("scenario", nargs="?", choices=list(SCENARIOS), default="sudden_shift")
    parser.add_argument("--all", action="store_true", help="summary of every scenario")
    parser.add_argument("--explain", action="store_true", help="explain the first alert")
    args = parser.parse_args()

    monitor, X_test, y_test = build_monitor()
    if args.all:
        summary(monitor, X_test, y_test)
    else:
        watch(args.scenario, monitor, X_test, y_test, explain=args.explain)
