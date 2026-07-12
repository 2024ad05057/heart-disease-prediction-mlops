"""
preprocess_backup.py

Contains reusable preprocessing pipeline for the
Heart Disease Prediction project.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


NUMERICAL_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"
]

CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal"
]


def create_preprocessor():
    """
    Creates a preprocessing pipeline.

    Returns
    -------
    ColumnTransformer
        Preprocessing pipeline
    """

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                NUMERICAL_FEATURES
            ),
            (
                "cat",
                categorical_transformer,
                CATEGORICAL_FEATURES
            )
        ]
    )

    return preprocessor