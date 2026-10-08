"""Load and clean the UCI Adult dataset (see docs/DATASET.md).

    from deploymate.data.load import load_dataset
    X_train, X_test, y_train, y_test = load_dataset()

y is 1 for income >50K, 0 for <=50K.
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income",
]
TARGET = "income"

# Dropped columns:
#   fnlwgt    - a census sampling weight (how many people the row represents),
#               not a fact about the person, so it shouldn't drive predictions.
#   education - the same information as education_num, just as text.
DROPPED = ["fnlwgt", "education"]

NUMERIC = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL = ["workclass", "marital_status", "occupation", "relationship",
               "race", "sex", "native_country"]
FEATURES = NUMERIC + CATEGORICAL


def read_raw(path: Path, is_test: bool = False) -> pd.DataFrame:
    """Read one raw file and fix its quirks. Returns features + target column."""
    df = pd.read_csv(
        path,
        names=COLUMNS,
        skipinitialspace=True,      # values come as " State-gov"
        na_values="?",              # missing values are written as "?"
        skiprows=1 if is_test else 0,  # adult.test starts with a junk line
    )
    df = df.dropna(how="all")       # trailing blank lines
    df[TARGET] = df[TARGET].str.rstrip(".")  # adult.test labels look like ">50K."
    df[CATEGORICAL] = df[CATEGORICAL].fillna("Unknown")
    return df.drop(columns=DROPPED)


def load_dataset(raw_dir: Path = RAW_DIR):
    """Return (X_train, X_test, y_train, y_test) using the dataset's own split."""
    raw_dir = Path(raw_dir)
    if not (raw_dir / "adult.data").exists():
        raise FileNotFoundError(
            f"No dataset in {raw_dir}. Run: python scripts/download_data.py"
        )

    train = read_raw(raw_dir / "adult.data")
    test = read_raw(raw_dir / "adult.test", is_test=True)

    def split(df):
        return df[FEATURES].reset_index(drop=True), (df[TARGET] == ">50K").astype(int).reset_index(drop=True)

    X_train, y_train = split(train)
    X_test, y_test = split(test)
    return X_train, X_test, y_train, y_test
