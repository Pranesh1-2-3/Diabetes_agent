"""FastAPI endpoint for diabetes prediction with SHAP explanations."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path
import numpy as np
import shap
import xgboost as xgb

app = FastAPI(
    title="Diabetes Prediction API",
    description="API for predicting diabetes risk with SHAP explanations",
    version="1.0.0"
)

# Load model and feature names
model_dir = Path(__file__).parent.parent.parent.parent / "models"
model_path = model_dir / "xgb.pkl"  # Using the existing XGBoost model
feature_names_path = model_dir / "feature_names.joblib"

model = joblib.load(model_path)
feature_names = joblib.load(feature_names_path)['features']


# -------------------------
# Data Models
# -------------------------
class PredictionInput(BaseModel):
    """Input schema for prediction endpoint."""
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float


class PredictionOutput(BaseModel):
    """Output schema for prediction endpoint."""
    probability: float
    label: int
    top_features_shap: list


# -------------------------
# API Endpoints
# -------------------------
@app.get("/")
def root():
    """Root endpoint."""
    return {"msg": "Diabetes Agent API is running"}

import requests

def predict_risk(patient_data: dict) -> dict:
    """
    Predict diabetes risk by calling the /predict API endpoint.
    Returns actual probability, label, and top features.
    """
    url = "http://127.0.0.1:8001/predict"  # Change if your server runs on another host/port

    # Prepare payload matching model features exactly
    payload = {
        "Pregnancies": patient_data.get("Pregnancies", 0),
        "Glucose": patient_data.get("Glucose", 0),
        "BloodPressure": patient_data.get("BloodPressure", 0),
        "SkinThickness": patient_data.get("SkinThickness", 0),
        "Insulin": patient_data.get("Insulin", 0),
        "BMI": patient_data.get("BMI", 0),
        "DiabetesPedigreeFunction": patient_data.get("DiabetesPedigreeFunction", 0.0),
        "Age": patient_data.get("Age", 0)
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()

        # Format the output nicely
        return {
            "prediction": "high_risk" if result["label"] == 1 else "low_risk",
            "probability": result["probability"],
            "top_features": [f["name"] for f in result.get("top_features_shap", [])]
        }

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error calling /predict endpoint: {e}")


@app.post("/predict", response_model=PredictionOutput)
async def predict(data: PredictionInput):
    """Make prediction and return probability, label and SHAP explanations."""
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([data.dict()])

        # Ensure columns are in correct order
        input_df = input_df[feature_names]

        # Get prediction probability
        probability = model.predict_proba(input_df)[0, 1]

        # Get predicted label
        label = int(probability >= 0.5)

        # Initialize top_features
        top_features = []

        # SHAP explanation
        try:
            if isinstance(model, xgb.XGBClassifier):
                # For XGBoost, use TreeExplainer
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_df)

                # Pair features with absolute SHAP importance
                feature_importance = list(zip(feature_names, np.abs(shap_values[0])))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                top_features = [
                    {"name": feat, "importance": float(imp)}
                    for feat, imp in feature_importance[:3]
                ]

            else:
                # For other models, use KernelExplainer
                explainer = shap.KernelExplainer(
                    model.predict_proba,
                    input_df.sample(min(50, len(input_df)))  # background dataset
                )
                shap_values = explainer.shap_values(input_df)

                if isinstance(shap_values, list):  # binary classification
                    shap_values = shap_values[1]

                feature_importance = list(zip(feature_names, np.abs(shap_values[0])))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                top_features = [
                    {"name": feat, "importance": float(imp)}
                    for feat, imp in feature_importance[:3]
                ]
        except Exception:
            # Fallback if SHAP fails
            top_features = [{"name": "N/A", "importance": 0.0}] * 3

        return {
            "probability": float(probability),
            "label": label,
            "top_features_shap": top_features
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )
