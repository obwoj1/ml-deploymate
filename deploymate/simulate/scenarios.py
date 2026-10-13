"""Turn the held-out test set into simulated live traffic, some of it broken on purpose.

    from deploymate.simulate.scenarios import simulate
    for batch in simulate("sudden_shift", X_test, y_test):
        batch.X, batch.y, batch.drifted, batch.strength

Each batch carries the TRUE answer to "was this batch broken?" (batch.drifted).
We did the breaking, so we know. That ground truth is what lets us measure the
detector's precision, recall and detection delay.
"""

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np
import pandas as pd

DEFAULT_BATCHES = 40
DEFAULT_BATCH_SIZE = 400


@dataclass
class Batch:
    batch_id: int        # 1-based, in arrival order
    scenario: str
    X: pd.DataFrame
    y: pd.Series
    drifted: bool        # ground truth: was this batch deliberately changed?
    strength: float      # 0 = untouched, 1 = full-strength change


# --- How each scenario breaks a batch --------------------------------------
# Each function gets a copy of the batch, a strength in (0, 1], and a random
# generator, and returns the modified batch. Real-world stories in comments.

def _gradual_shift(X, y, strength, rng):
    # Customer base slowly aging: up to +15 years by the last batch.
    X["age"] = (X["age"] + 15 * strength).clip(17, 90)
    return X, y


def _sudden_shift(X, y, strength, rng):
    # A policy change or new data source: everyone suddenly logs ~15 more hours.
    X["hours_per_week"] = (X["hours_per_week"] + 15 * strength).clip(1, 99)
    return X, y


def _category_shift(X, y, strength, rng):
    # Launch to a new market: half the batch now comes from 3 occupations.
    rows = rng.random(len(X)) < 0.5 * strength
    X.loc[rows, "occupation"] = rng.choice(
        ["Exec-managerial", "Prof-specialty", "Tech-support"], rows.sum())
    return X, y


def _missing_values(X, y, strength, rng):
    # A broken upstream pipeline: 40% of these fields arrive empty.
    for col in ["workclass", "occupation", "native_country"]:
        rows = rng.random(len(X)) < 0.4 * strength
        X.loc[rows, col] = "Unknown"
    return X, y


def _noise(X, y, strength, rng):
    # Sensor / data-entry errors: random jitter on numeric fields.
    X["age"] = (X["age"] + rng.normal(0, 12 * strength, len(X))).round().clip(17, 90)
    X["hours_per_week"] = (X["hours_per_week"]
                           + rng.normal(0, 15 * strength, len(X))).round().clip(1, 99)
    return X, y


# name -> (description, corrupt function, fraction of the run before it starts)
SCENARIOS: dict[str, tuple[str, Callable | None, float]] = {
    "healthy": ("Normal traffic: nothing changes.", None, 1.0),
    "gradual_shift": ("Ages slowly creep upward batch after batch.", _gradual_shift, 0.3),
    "sudden_shift": ("Hours-per-week jumps suddenly halfway through.", _sudden_shift, 0.5),
    "category_shift": ("The occupation mix changes sharply.", _category_shift, 0.5),
    "missing_values": ("An upstream pipeline breaks and fields go blank.", _missing_values, 0.5),
    "noise": ("Random noise is added to numeric fields.", _noise, 0.5),
}


def simulate(name: str, X: pd.DataFrame, y: pd.Series, n_batches: int = DEFAULT_BATCHES,
             batch_size: int = DEFAULT_BATCH_SIZE, seed: int = 0) -> Iterator[Batch]:
    """Yield n_batches of live-looking traffic for one scenario.

    Batches before the scenario's start point are untouched (healthy), so every
    run has both normal and broken batches, which is needed to measure false alarms.
    """
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario {name!r}. Choose from {list(SCENARIOS)}")
    _, corrupt, start_fraction = SCENARIOS[name]
    rng = np.random.default_rng(seed)

    total = n_batches * batch_size
    order = rng.choice(len(X), total, replace=total > len(X))
    start = int(n_batches * start_fraction)

    for i in range(n_batches):
        rows = order[i * batch_size:(i + 1) * batch_size]
        X_batch = X.iloc[rows].reset_index(drop=True).copy()
        y_batch = y.iloc[rows].reset_index(drop=True).copy()

        strength = 0.0
        if corrupt is not None and i >= start:
            if name == "gradual_shift":   # ramps from small to full over the remaining batches
                strength = (i - start + 1) / (n_batches - start)
            else:                         # full strength immediately
                strength = 1.0
            X_batch, y_batch = corrupt(X_batch, y_batch, strength, rng)

        yield Batch(i + 1, name, X_batch, y_batch, strength > 0, round(strength, 3))
