import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features such as BMI.
    """
    result = df.copy()

    if "Weight" in result.columns and "Height" in result.columns:
        result["BMI"] = result["Weight"] / (result["Height"] ** 2)

    return result


def clean_ordinal_columns(df: pd.DataFrame, ordinal_cols: list[str]) -> pd.DataFrame:
    """
    Round ordinal columns to nearest integer.
    Missing columns are ignored.
    """
    result = df.copy()

    for col in ordinal_cols:
        if col in result.columns:
            result[col] = result[col].round().astype(int)

    return result

def split_data(
    df: pd.DataFrame,
    target: str,
    drop_columns: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split dataset into train/test sets.

    Args:
        df: Full dataframe
        target: Target column name
        drop_columns: Optional columns to remove before splitting
        test_size: Test proportion
        random_state: Random seed

    Returns:
        X_train, X_test, y_train, y_test
    """
    work_df = df.copy()

    if drop_columns:
        existing_cols = [c for c in drop_columns if c in work_df.columns]
        work_df = work_df.drop(columns=existing_cols)

    X = work_df.drop(columns=[target])
    y = work_df[target]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

def build_preprocessor(
    X: pd.DataFrame,
    ord_cols: list[str],
    ohe_min_frequency: int = 10,
) -> ColumnTransformer:
    """
    Build preprocessing pipeline with:
    - scaled numeric columns
    - one-hot categorical columns
    - passthrough ordinal columns
    """
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # remove ordinal cols from numeric pipeline
    num_cols = [c for c in num_cols if c not in ord_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=ohe_min_frequency,
                    sparse_output=False,
                ),
                cat_cols,
            ),
            ("ord", "passthrough", ord_cols),
        ]
    )

    return preprocessor