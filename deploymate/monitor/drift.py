"""Data drift tests: compare a live batch of data against the training reference.

Numeric features  -> Population Stability Index (PSI) + Kolmogorov-Smirnov test
Categorical       -> Chi-squared test + Cramér's V (effect size)

With big batches, p-values flag tiny, meaningless shifts as "significant", so the
drift decision uses effect sizes (PSI, Cramér's V). The p-values are reported
for context only.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

# Common industry rule of thumb: PSI < 0.1 stable, 0.1-0.25 moderate, > 0.25 major shift.
PSI_THRESHOLD = 0.2
CRAMERS_V_THRESHOLD = 0.1
# Categories rarer than this in the reference get pooled into "Other". With many
# rare categories (native_country has 41), a 400-row batch looks "different" by
# pure chance and inflates Cramér's V. Pooling removes those false alarms.
RARE_CATEGORY_SHARE = 0.01
_EPS = 1e-6  # avoids log(0) / divide-by-zero for empty bins


@dataclass
class FeatureDrift:
    feature: str
    kind: str          # "numeric" or "categorical"
    statistic: float   # PSI for numeric, Cramér's V for categorical
    p_value: float     # KS p-value for numeric, chi-squared p-value for categorical
    drifted: bool


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two numeric samples.

    Bin edges come from the reference quantiles, so each bin holds ~10% of the
    training data. A larger PSI means the live distribution moved further.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:  # constant feature: drifted only if live values differ
        return 0.0 if np.allclose(current, reference[0]) else float("inf")
    edges[0], edges[-1] = -np.inf, np.inf

    ref_pct = np.histogram(reference, edges)[0] / len(reference) + _EPS
    cur_pct = np.histogram(current, edges)[0] / len(current) + _EPS
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def numeric_drift(name: str, reference: pd.Series, current: pd.Series) -> FeatureDrift:
    ref, cur = reference.dropna().to_numpy(), current.dropna().to_numpy()
    score = psi(ref, cur)
    p_value = float(stats.ks_2samp(ref, cur).pvalue)
    return FeatureDrift(name, "numeric", score, p_value, score >= PSI_THRESHOLD)


def _pool_rare(reference: pd.Series, current: pd.Series):
    """Map categories that are rare in the reference (and brand-new ones) to "Other"."""
    shares = reference.value_counts(normalize=True)
    common = set(shares[shares >= RARE_CATEGORY_SHARE].index)
    pool = lambda s: s.where(s.isin(common) | s.isna(), "Other")  # noqa: E731
    return pool(reference), pool(current)


def categorical_drift(name: str, reference: pd.Series, current: pd.Series) -> FeatureDrift:
    reference, current = _pool_rare(reference, current)
    categories = sorted(set(reference.dropna()) | set(current.dropna()), key=str)
    table = np.array([
        reference.value_counts().reindex(categories, fill_value=0).to_numpy(),
        current.value_counts().reindex(categories, fill_value=0).to_numpy(),
    ])
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2:  # only one category seen anywhere: nothing can drift
        return FeatureDrift(name, "categorical", 0.0, 1.0, False)

    chi2, p_value, _, _ = stats.chi2_contingency(table)
    cramers_v = float(np.sqrt(chi2 / (table.sum() * (min(table.shape) - 1))))
    return FeatureDrift(name, "categorical", cramers_v, float(p_value),
                        cramers_v >= CRAMERS_V_THRESHOLD)


def detect_drift(reference: pd.DataFrame, current: pd.DataFrame) -> list[FeatureDrift]:
    """Run the right drift test on every column the two frames share."""
    results = []
    for col in reference.columns.intersection(current.columns):
        if pd.api.types.is_numeric_dtype(reference[col]):
            results.append(numeric_drift(col, reference[col], current[col]))
        else:
            results.append(categorical_drift(col, reference[col], current[col]))
    return results
