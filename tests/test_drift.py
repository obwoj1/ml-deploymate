import numpy as np
import pandas as pd

from deploymate.monitor.drift import detect_drift, psi

rng = np.random.default_rng(42)


def make_frame(n, age_shift=0.0, job_probs=(0.5, 0.3, 0.2)):
    return pd.DataFrame({
        "age": rng.normal(40 + age_shift, 10, n),
        "job": rng.choice(["tech", "sales", "admin"], n, p=job_probs),
    })


def by_feature(results):
    return {r.feature: r for r in results}


def test_psi_is_near_zero_for_same_distribution():
    assert psi(rng.normal(0, 1, 5000), rng.normal(0, 1, 5000)) < 0.05


def test_psi_grows_with_bigger_shifts():
    ref = rng.normal(0, 1, 5000)
    small = psi(ref, rng.normal(0.3, 1, 5000))
    large = psi(ref, rng.normal(1.5, 1, 5000))
    assert 0 < small < large


def test_no_drift_on_healthy_batch():
    results = detect_drift(make_frame(5000), make_frame(1000))
    assert not any(r.drifted for r in results)


def test_numeric_shift_is_flagged():
    results = by_feature(detect_drift(make_frame(5000), make_frame(1000, age_shift=10)))
    assert results["age"].drifted
    assert results["age"].kind == "numeric"
    assert not results["job"].drifted


def test_categorical_shift_is_flagged():
    results = by_feature(detect_drift(make_frame(5000), make_frame(1000, job_probs=(0.1, 0.2, 0.7))))
    assert results["job"].drifted
    assert results["job"].kind == "categorical"
    assert not results["age"].drifted


def test_unseen_category_in_live_data_is_handled():
    ref = make_frame(5000)
    cur = make_frame(1000)
    cur.loc[:400, "job"] = "brand_new_job"
    assert by_feature(detect_drift(ref, cur))["job"].drifted


def test_only_shared_columns_are_checked():
    cur = make_frame(1000).drop(columns="job")
    assert [r.feature for r in detect_drift(make_frame(5000), cur)] == ["age"]


def test_rare_categories_do_not_cause_false_alarms():
    # Like native_country: one dominant value plus 40 rare ones (0.25% each).
    countries = ["US"] + [f"rare_{i}" for i in range(40)]
    probs = [0.9] + [0.1 / 40] * 40
    ref = pd.DataFrame({"country": rng.choice(countries, 20000, p=probs)})
    alarms = 0
    for _ in range(30):  # 30 healthy batches of 400 rows
        cur = pd.DataFrame({"country": rng.choice(countries, 400, p=probs)})
        alarms += by_feature(detect_drift(ref, cur))["country"].drifted
    assert alarms <= 1
