"""FastAPI service that serves the trained Diabetes prediction model.

The service loads the scikit-learn pipeline that was trained on Google Colab
(see notebook/Diabetes_Model_Training.ipynb) and exposes it over HTTP.
"""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

from app.schemas import PatientData, PredictionResponse

# The model was saved next to the project root during training.
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "model.joblib"

# The exact feature order the model expects.
FEATURE_ORDER = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# Load the model once when the application starts.
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Diabetes Prediction API",
    description="Predicts the likelihood of diabetes from medical measurements.",
    version="1.0.0",
)


@app.get("/")
def root():
    """Small welcome message so we know the API is reachable."""
    return {"message": "Diabetes Prediction API. See /docs for usage."}


@app.get("/health")
def health_check():
    """Health check used by Docker and monitoring tools."""
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PatientData):
    """Make a prediction for a single patient."""
    # Build a one row DataFrame with the same column order used in training.
    features = pd.DataFrame([data.model_dump()])[FEATURE_ORDER]

    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][prediction])

    return PredictionResponse(
        prediction=prediction,
        label="Diabetes" if prediction == 1 else "No diabetes",
        probability=round(probability, 4),
    )