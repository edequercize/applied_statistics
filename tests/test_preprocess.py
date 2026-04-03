"""Tests unitaires pour le module de preprocessing."""

import pandas as pd

from src.data.preprocess import (
    add_basic_features,
    build_preprocessor,
    clean_ordinal_columns,
    split_data,
)
from src.models.evaluate import mae_ordinale

# ── Tests preprocessing ─────────────────────────────────────────────────────


class TestAddBasicFeatures:
    def test_bmi_computed(self):
        df = pd.DataFrame({"Weight": [80.0], "Height": [1.80]})
        result = add_basic_features(df)
        expected_bmi = 80.0 / (1.80**2)
        assert "BMI" in result.columns
        assert abs(result["BMI"].iloc[0] - expected_bmi) < 1e-6

    def test_no_weight_height(self):
        df = pd.DataFrame({"Age": [25], "Gender": ["Male"]})
        result = add_basic_features(df)
        assert "BMI" not in result.columns


class TestCleanOrdinalColumns:
    def test_rounds_correctly(self):
        df = pd.DataFrame({"FAF": [1.7, 2.3], "FCVC": [2.9, 1.1]})
        result = clean_ordinal_columns(df, ["FAF", "FCVC"])
        assert result["FAF"].tolist() == [2, 2]
        assert result["FCVC"].tolist() == [3, 1]

    def test_missing_column_ignored(self):
        df = pd.DataFrame({"FAF": [1.5]})
        result = clean_ordinal_columns(df, ["FAF", "NONEXISTENT"])
        assert result["FAF"].iloc[0] == 2


class TestSplitData:
    def test_split_shapes(self):
        df = pd.DataFrame({
            "A": range(100),
            "B": ["cat"] * 50 + ["dog"] * 50,
            "target": ["a"] * 50 + ["b"] * 50,
        })
        X_train, X_test, y_train, y_test = split_data(df, "target", test_size=0.2)
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert "target" not in X_train.columns


class TestBuildPreprocessor:
    def test_output_shape(self):
        X = pd.DataFrame({
            "Age": [25.0, 30.0, 35.0],
            "Gender": ["Male", "Female", "Male"],
            "FAF": [1, 2, 3],
        })
        preproc = build_preprocessor(X, ord_cols=["FAF"], ohe_min_frequency=1)
        preproc.fit(X)
        result = preproc.transform(X)
        # 1 num (Age scaled) + 2 cat (Gender OHE) + 1 ord (FAF) = 4
        assert result.shape[1] == 4


# ── Tests métriques ──────────────────────────────────────────────────────────


class TestMaeOrdinale:
    def test_perfect_prediction(self):
        y = ["Obesity_Type_I", "Normal_Weight"]
        assert mae_ordinale(y, y) == 0.0

    def test_off_by_one(self):
        y_true = ["Obesity_Type_I", "Obesity_Type_I"]
        y_pred = ["Obesity_Type_II", "Obesity_Type_II"]
        assert mae_ordinale(y_true, y_pred) == 1.0

    def test_numeric_inputs(self):
        assert mae_ordinale([0, 1, 2], [0, 1, 2]) == 0.0
        assert mae_ordinale([0, 0, 0], [3, 3, 3]) == 3.0
