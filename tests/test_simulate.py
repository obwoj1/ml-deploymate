import numpy as np
import pandas as pd
import pytest

from deploymate.data.load import CATEGORICAL, FEATURES, NUMERIC
from deploymate.simulate.scenarios import SCENARIOS, simulate

rng = np.random.default_rng(1)
N = 3000
X = pd.DataFrame({
    "age": rng.integers(17, 90, N), "education_num": rng.integers(1, 16, N),
    "capital_gain": np.zeros(N), "capital_loss": np.zeros(N),
    "hours_per_week": rng.integers(20, 60, N),
})
for col in CATEGORICAL:
    X[col] = rng.choice(["a", "b", "c"], N)
X = X[FEATURES]
y = pd.Series(rng.integers(0, 2, N))


def run(name, **kw):
    return list(simulate(name, X, y, n_batches=20, batch_size=100, **kw))


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_batch_count_size_and_columns(name):
    batches = run(name)
    assert len(batches) == 20
    assert all(len(b.X) == 100 and len(b.y) == 100 for b in batches)
    assert all(list(b.X.columns) == FEATURES for b in batches)
    assert [b.batch_id for b in batches] == list(range(1, 21))


def test_healthy_is_never_drifted():
    assert not any(b.drifted for b in run("healthy"))


@pytest.mark.parametrize("name", [n for n in SCENARIOS if n != "healthy"])
def test_scenarios_start_healthy_then_drift(name):
    batches = run(name)
    assert not batches[0].drifted
    assert batches[-1].drifted
    first = next(i for i, b in enumerate(batches) if b.drifted)
    assert all(b.drifted for b in batches[first:])  # once broken, stays broken


def test_gradual_shift_ramps_up():
    strengths = [b.strength for b in run("gradual_shift") if b.drifted]
    assert strengths == sorted(strengths) and strengths[-1] == 1.0


def test_sudden_shift_raises_hours():
    batches = run("sudden_shift")
    assert batches[-1].X["hours_per_week"].mean() > batches[0].X["hours_per_week"].mean() + 10


def test_missing_values_adds_unknowns():
    batches = run("missing_values")
    assert (batches[-1].X["workclass"] == "Unknown").mean() > 0.25
    assert (batches[0].X["workclass"] == "Unknown").mean() == 0


def test_corruption_does_not_touch_source_data():
    before = X.copy()
    run("noise")
    pd.testing.assert_frame_equal(X, before)


def test_same_seed_same_batches():
    a, b = run("category_shift", seed=3), run("category_shift", seed=3)
    assert all(x.X.equals(y_.X) for x, y_ in zip(a, b))


def test_numeric_columns_stay_in_valid_range():
    for b in run("noise") + run("gradual_shift"):
        assert b.X["age"].between(17, 90).all()
        assert b.X["hours_per_week"].between(1, 99).all()
    assert all(X[c].dtype.kind in "if" for c in NUMERIC)


def test_unknown_scenario_raises():
    with pytest.raises(ValueError):
        run("nope")
