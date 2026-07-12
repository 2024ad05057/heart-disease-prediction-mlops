"""
train_withoutpathlib.py

Train machine learning models for Heart Disease Prediction.
"""

from pathlib import Path
import joblib
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.preprocess import create_preprocessor
from src.evaluate import evaluate_model



#PROJECT_ROOT = Path(__file__).resolve().parent.parent



#print("Tracking URI:", mlflow.get_tracking_uri())

# =====================================================
# Project Paths
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "heart_clean.csv"

MODELS_DIR = PROJECT_ROOT / "models"

# =====================================================
# MLflow Experiment
# =====================================================

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")

mlflow.set_experiment("Heart Disease Prediction")


def train_model(data_path=DATA_PATH):
    """
    Train machine learning models and log experiment using MLflow.

    Parameters
    ----------
    data_path : Path
        Path to cleaned dataset.

    Returns
    -------
    dict
        Model metrics and best parameters.
    """

    with mlflow.start_run():

        # =====================================================
        # Load Dataset
        # =====================================================

        df = pd.read_csv(data_path)

        X = df.drop("target", axis=1)
        y = df["target"]

        # =====================================================
        # Train Test Split
        # =====================================================

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        # =====================================================
        # Preprocessor
        # =====================================================

        preprocessor = create_preprocessor()

        # =====================================================
        # Logistic Regression
        # =====================================================

        lr_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            )
        ])

        lr_pipeline.fit(X_train, y_train)

        y_pred_lr = lr_pipeline.predict(X_test)

        y_prob_lr = lr_pipeline.predict_proba(X_test)[:, 1]

        lr_metrics = evaluate_model(
            y_test,
            y_pred_lr,
            y_prob_lr
        )

        # =====================================================
        # Random Forest
        # =====================================================

        rf_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=42
                )
            )
        ])

        param_grid = {

            "classifier__n_estimators": [100, 200],

            "classifier__max_depth": [5, 10, None],

            "classifier__min_samples_split": [2, 5]

        }

        grid_search = GridSearchCV(
            estimator=rf_pipeline,
            param_grid=param_grid,
            cv=5,
            scoring="roc_auc"
        )

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        # =====================================================
        # Log Parameters
        # =====================================================

        mlflow.log_params(grid_search.best_params_)

        # =====================================================
        # Predictions
        # =====================================================

        y_pred_rf = best_model.predict(X_test)

        y_prob_rf = best_model.predict_proba(X_test)[:, 1]

        rf_metrics = evaluate_model(
            y_test,
            y_pred_rf,
            y_prob_rf
        )


        # =====================================================
        # Log Metrics
        # =====================================================

        mlflow.log_metric("accuracy", rf_metrics["Accuracy"])
        mlflow.log_metric("precision", rf_metrics["Precision"])
        mlflow.log_metric("recall", rf_metrics["Recall"])
        mlflow.log_metric("f1_score", rf_metrics["F1-Score"])
        mlflow.log_metric("roc_auc", rf_metrics["ROC-AUC"])

        # =====================================================
        # Save Model
        # =====================================================

        MODELS_DIR.mkdir(exist_ok=True)

        model_path = MODELS_DIR / "best_model.pkl"

        joblib.dump(
            best_model,
            model_path
        )

        print(f"Model saved to: {model_path}")

        # =====================================================
        # Log Model
        # =====================================================

        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="best_model"
        )

        print("Model logged to MLflow successfully!")

        # =====================================================
        # Return Results
        # =====================================================

        return {

            "Logistic Regression": lr_metrics,

            "Random Forest": rf_metrics,

            "Best Model": "Random Forest",
            #"Best Model": best_model_name,

            "Best Parameters": grid_search.best_params_

        }


if __name__ == "__main__":

    results = train_model()

    print("\n========== Training Completed ==========\n")

    print(results)