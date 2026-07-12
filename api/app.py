from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI

from api.schemas import HeartDiseaseInput
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0"
)
Instrumentator().instrument(app).expose(app)
import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"

model = joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return {
        "message": "Heart Disease Prediction API is running!"
    }


# @app.post("/predict")
# def predict(data: HeartDiseaseInput):
#
#     input_df = pd.DataFrame([data.model_dump()])
#
#     prediction = model.predict(input_df)[0]
#
#     probability = float(
#         model.predict_proba(input_df)[0][1]
#     )
#
#     return {
#
#         "prediction": int(prediction),
#
#         "probability": round(probability, 4)
#
#     }


@app.post("/predict")
def predict(data: HeartDiseaseInput):
    logger.info("Received prediction request")

    input_df = pd.DataFrame([data.model_dump()])

    prediction = model.predict(input_df)[0]

    probability = float(
        model.predict_proba(input_df)[0][1]
    )

    # Prediction Label
    if prediction == 1:
        diagnosis = "Heart Disease Detected"
    else:
        diagnosis = "No Heart Disease"

    # Log prediction
    logger.info(
        f"Input: {data.model_dump()} | "
        f"Prediction: {prediction} | "
        f"Diagnosis: {diagnosis} | "
        f"Probability: {probability:.4f}"
    )

    return {

        "prediction": int(prediction),

        "diagnosis": diagnosis,

        "probability": round(probability, 4)

    }