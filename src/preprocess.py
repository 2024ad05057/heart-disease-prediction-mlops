"""
preprocess.py

Creates the preprocessing pipeline for the
Heart Disease Prediction project.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Numerical features
NUMERICAL_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"
]

# Categorical features
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
    Create preprocessing pipeline.

    Returns
    -------
    ColumnTransformer
        Configured preprocessing pipeline.
    """

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
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