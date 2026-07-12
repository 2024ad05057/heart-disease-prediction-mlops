"""
train_no_mlfow.py

Train machine learning models for Heart Disease Prediction.
"""

import os
import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.preprocess import create_preprocessor
from src.evaluate import evaluate_model


def train_model():

    # Load dataset
    df = pd.read_csv("../data/processed/heart_clean.csv")

    X = df.drop("target", axis=1)
    y = df["target"]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Preprocessing
    preprocessor = create_preprocessor()

    # -----------------------------
    # Logistic Regression
    # -----------------------------

    lr_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000,
                                          random_state=42))
    ])

    lr_pipeline.fit(X_train, y_train)

    y_pred_lr = lr_pipeline.predict(X_test)
    y_prob_lr = lr_pipeline.predict_proba(X_test)[:, 1]

    lr_metrics = evaluate_model(
        y_test,
        y_pred_lr,
        y_prob_lr
    )

    # -----------------------------
    # Random Forest
    # -----------------------------

    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",
         RandomForestClassifier(random_state=42))
    ])

    param_grid = {

        "classifier__n_estimators": [100, 200],

        "classifier__max_depth": [5, 10, None],

        "classifier__min_samples_split": [2, 5]

    }

    grid_search = GridSearchCV(
        rf_pipeline,
        param_grid,
        cv=5,
        scoring="roc_auc"
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    y_pred_rf = best_model.predict(X_test)

    y_prob_rf = best_model.predict_proba(X_test)[:, 1]

    rf_metrics = evaluate_model(
        y_test,
        y_pred_rf,
        y_prob_rf
    )

    # Save Model
    os.makedirs("../models", exist_ok=True)

    joblib.dump(
        best_model,
        "../models/best_model.pkl"
    )

    return {

        "Logistic Regression": lr_metrics,

        "Random Forest": rf_metrics,

        "Best Model": best_model,

        "Best Parameters": grid_search.best_params_

    }
if __name__ == "__main__":
    results = train_model()
    print(results)
