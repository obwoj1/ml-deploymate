import pytest

from deploymate.data.load import CATEGORICAL, FEATURES, NUMERIC, RAW_DIR, load_dataset

TRAIN_ROWS = """\
39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K
50, Self-emp-not-inc, 83311, Bachelors, 13, Married-civ-spouse, Exec-managerial, Husband, White, Male, 0, 0, 13, United-States, >50K
54, ?, 180211, Some-college, 10, Married-civ-spouse, ?, Husband, Asian-Pac-Islander, Male, 0, 0, 60, ?, >50K

"""
TEST_ROWS = """\
|1x3 Cross validator
25, Private, 226802, 11th, 7, Never-married, Machine-op-inspct, Own-child, Black, Male, 0, 0, 40, United-States, <=50K.
44, Private, 160323, Some-college, 10, Married-civ-spouse, Machine-op-inspct, Husband, Black, Male, 7688, 0, 40, United-States, >50K.
"""


@pytest.fixture
def raw_dir(tmp_path):
    (tmp_path / "adult.data").write_text(TRAIN_ROWS)
    (tmp_path / "adult.test").write_text(TEST_ROWS)
    return tmp_path


def test_shapes_and_columns(raw_dir):
    X_train, X_test, y_train, y_test = load_dataset(raw_dir)
    assert len(X_train) == 3 and len(X_test) == 2  # blank line + junk header dropped
    assert list(X_train.columns) == FEATURES
    assert "fnlwgt" not in X_train and "education" not in X_train


def test_labels_are_binary_and_test_periods_stripped(raw_dir):
    _, _, y_train, y_test = load_dataset(raw_dir)
    assert y_train.tolist() == [0, 1, 1]
    assert y_test.tolist() == [0, 1]


def test_missing_values_become_unknown(raw_dir):
    X_train, *_ = load_dataset(raw_dir)
    row = X_train.iloc[2]
    assert row["workclass"] == "Unknown"
    assert row["occupation"] == "Unknown"
    assert row["native_country"] == "Unknown"
    assert not X_train.isna().any().any()


def test_whitespace_stripped_and_numbers_are_numeric(raw_dir):
    X_train, *_ = load_dataset(raw_dir)
    assert X_train.loc[0, "workclass"] == "State-gov"
    for col in NUMERIC:
        assert X_train[col].dtype.kind in "if", col


def test_missing_dataset_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data.py"):
        load_dataset(tmp_path)


@pytest.mark.skipif(not (RAW_DIR / "adult.data").exists(), reason="real dataset not downloaded")
def test_real_dataset_sizes():
    X_train, X_test, y_train, y_test = load_dataset()
    assert len(X_train) == 32561 and len(X_test) == 16281
    assert 0.23 < y_train.mean() < 0.25
    assert not X_train[CATEGORICAL].isna().any().any()
