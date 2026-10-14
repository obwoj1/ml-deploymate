"""Watch a drift scenario play out, batch by batch.

    python scripts/simulate.py sudden_shift
    python scripts/simulate.py --all          # one-line summary per scenario

For every batch: was it really broken (truth), the model's accuracy on it, and
which features the drift tests flagged. Needs the trained model:
run scripts/train.py first.
"""

import argparse
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from deploymate.data.load import load_dataset  # noqa: E402
from deploymate.monitor.drift import detect_drift  # noqa: E402
from deploymate.simulate.scenarios import SCENARIOS, simulate  # noqa: E402


def load():
    model_path = ROOT / "models" / "model.joblib"
    if not model_path.exists():
        sys.exit("No trained model. Run: python scripts/train.py")
    X_train, X_test, _, y_test = load_dataset()
    return joblib.load(model_path), X_train, X_test, y_test


def watch(name, model, reference, X_test, y_test):
    print(f"\nScenario: {name}: {SCENARIOS[name][0]}\n")
    print(f"{'batch':>5}  {'truth':<7} {'accuracy':>8}  flagged features")
    for b in simulate(name, X_test, y_test):
        accuracy = (model.predict(b.X) == b.y).mean()
        flagged = [d.feature for d in detect_drift(reference, b.X) if d.drifted]
        truth = "BROKEN" if b.drifted else "ok"
        print(f"{b.batch_id:>5}  {truth:<7} {accuracy:>8.3f}  {', '.join(flagged) or '-'}")


def summary(model, reference, X_test, y_test):
    print(f"\n{'scenario':<16}{'broken':>7}{'caught':>8}{'false alarms':>14}{'acc ok→broken':>16}")
    for name in SCENARIOS:
        caught = false_alarms = broken = 0
        acc = {True: [], False: []}
        for b in simulate(name, X_test, y_test):
            alert = any(d.drifted for d in detect_drift(reference, b.X))
            broken += b.drifted
            caught += alert and b.drifted
            false_alarms += alert and not b.drifted
            acc[b.drifted].append((model.predict(b.X) == b.y).mean())
        ok_acc = sum(acc[False]) / len(acc[False])
        bad_acc = f"{sum(acc[True]) / len(acc[True]):.3f}" if acc[True] else "  -  "
        print(f"{name:<16}{broken:>7}{caught:>8}{false_alarms:>14}{ok_acc:>10.3f} → {bad_acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch drift scenarios play out.")
    parser.add_argument("scenario", nargs="?", choices=list(SCENARIOS), default="sudden_shift")
    parser.add_argument("--all", action="store_true", help="summary of every scenario")
    args = parser.parse_args()

    model, X_train, X_test, y_test = load()
    reference = X_train.sample(5000, random_state=0)  # "normal" = a slice of training data
    if args.all:
        summary(model, reference, X_test, y_test)
    else:
        watch(args.scenario, model, reference, X_test, y_test)
