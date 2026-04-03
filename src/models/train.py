from lightgbm import LGBMClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.models.evaluate import mae_ordinale


def train_lightgbm(
    X_train,
    y_train,
    preprocessor,
    param_grid: dict,
    cv: int = 3,
):
    """
    Train LightGBM pipeline with GridSearchCV.
    """
    pipeline = Pipeline([("preprocessor", preprocessor),
                            ("clf", LGBMClassifier(
                                random_state=42,
                                verbosity=-1,
                            )),
                        ])

    scorer = make_scorer(mae_ordinale, greater_is_better=False)

    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scorer,
        n_jobs=-1,
    )

    grid.fit(X_train, y_train)

    return grid
