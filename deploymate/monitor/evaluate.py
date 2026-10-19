"""Score the monitor against the simulator's ground truth.

Every simulated batch knows whether it was really broken, so each monitor
decision is one of:
    caught (TP)       broken batch, alert fired
    missed (FN)       broken batch, no alert
    false alarm (FP)  healthy batch, alert fired
    quiet (TN)        healthy batch, no alert

From those: precision (when it alerts, is it right?), recall (what share of
broken batches does it catch?), F1 (balance of the two), false-alarm rate, and
detection delay (how many broken batches go by before the first alert).

We score detectors from the same reports, to show what each part adds:
    drift_only    the per-feature drift tests (what DeployMate alerts on)
    anomaly_only  the model-behavior anomaly detector (info signal)
    either        alert if either fires (the design we evaluated and rejected)
"""

from __future__ import annotations

from deploymate.simulate.scenarios import SCENARIOS, simulate

DETECTORS = {
    "drift_only": lambda r: bool(r["drifted_features"]),
    "anomaly_only": lambda r: r["anomaly"]["anomalous"],
    "either": lambda r: bool(r["drifted_features"]) or r["anomaly"]["anomalous"],
}


def _scores(tp, fp, fn, tn, delays, runs_with_drift):
    # Precision only means something when there were broken batches to find.
    precision = tp / (tp + fp) if (tp + fp) and (tp + fn) else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall else (0.0 if precision is not None and recall is not None else None))
    r = lambda v: None if v is None else round(v, 4)  # noqa: E731
    return {
        "precision": r(precision),
        "recall": r(recall),
        "f1": r(f1),
        "false_alarm_rate": r(fp / (fp + tn) if fp + tn else None),
        "mean_detection_delay": r(sum(delays) / len(delays)) if delays else None,
        "runs_detected": f"{len(delays)}/{runs_with_drift}" if runs_with_drift else "-",
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def evaluate(monitor, X, y, seeds=range(10), n_batches: int = 40, batch_size: int = 400,
             scenarios=None) -> dict:
    """Run every scenario once per seed and score each detector.

    Returns {detector: {scenario: scores, ..., "overall": scores}}.
    """
    scenarios = list(scenarios or SCENARIOS)
    # counts[detector][scenario] = [tp, fp, fn, tn, delays, runs_with_drift]
    counts = {d: {s: [0, 0, 0, 0, [], 0] for s in scenarios} for d in DETECTORS}

    for s_index, scenario in enumerate(scenarios):
        for seed in seeds:
            # A different seed per scenario: with a shared seed, every scenario
            # replays the SAME healthy batches before its break, so one false
            # alarm would be counted once per scenario.
            run_seed = 1000 * s_index + seed
            batches = list(simulate(scenario, X, y, n_batches, batch_size, seed=run_seed))
            reports = [monitor.check(b.X) for b in batches]
            first_broken = next((i for i, b in enumerate(batches) if b.drifted), None)

            for name, fires in DETECTORS.items():
                c = counts[name][scenario]
                for b, rep in zip(batches, reports):
                    alert = fires(rep)
                    if b.drifted:
                        c[0 if alert else 2] += 1
                    else:
                        c[1 if alert else 3] += 1
                if first_broken is not None:
                    c[5] += 1
                    hit = next((i for i in range(first_broken, len(batches)) if fires(reports[i])), None)
                    if hit is not None:
                        c[4].append(hit - first_broken)  # 0 = caught on the very first broken batch

    results = {}
    for name in DETECTORS:
        per = {s: _scores(*counts[name][s]) for s in scenarios}
        totals = [sum(counts[name][s][k] for s in scenarios) for k in range(4)]
        delays = [d for s in scenarios for d in counts[name][s][4]]
        runs = sum(counts[name][s][5] for s in scenarios)
        per["overall"] = _scores(*totals, delays, runs)
        results[name] = per
    return results
