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
    """Make prediction and return probability, label, and SHAP explanations."""
    try:
        input_data = data.dict()

        # --- Input validation & clipping ---
        # Prevent out-of-range or unseen values that can confuse the model
        input_data["Glucose"] = np.clip(input_data["Glucose"], 0, 300)
        input_data["BloodPressure"] = np.clip(input_data["BloodPressure"], 0, 200)
        input_data["SkinThickness"] = np.clip(input_data["SkinThickness"], 0, 100)
        input_data["Insulin"] = np.clip(input_data["Insulin"], 0, 900)
        input_data["BMI"] = np.clip(input_data["BMI"], 0, 70)
        input_data["Age"] = np.clip(input_data["Age"], 0, 120)
        input_data["Pregnancies"] = np.clip(input_data["Pregnancies"], 0, 20)

        # --- Smarter fallback for DiabetesPedigreeFunction ---
        if input_data.get("DiabetesPedigreeFunction", 0) == 0:
            dpf = (
                0.4
                + (input_data["Glucose"] / 400)
                + (input_data["BMI"] / 150)
                + (input_data["Age"] / 300)
            )
            input_data["DiabetesPedigreeFunction"] = round(min(dpf, 2.0), 3)

        # --- Prepare DataFrame ---
        input_df = pd.DataFrame([input_data])[feature_names]

        # --- Model Prediction ---
        probability = float(model.predict_proba(input_df)[0, 1])

        # Adjusted threshold: 0.4 gives more clinically aligned results
        threshold = 0.4
        label = int(probability >= threshold)

        # --- SHAP Explanations ---
        try:
            if isinstance(model, xgb.XGBClassifier):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_df)
            else:
                explainer = shap.KernelExplainer(model.predict_proba, input_df.sample(1))
                shap_values = explainer.shap_values(input_df)

            if isinstance(shap_values, list):  # binary classification case
                shap_values = shap_values[1]

            feature_importance = sorted(
                [
                    {"name": feat, "importance": float(abs(val))}
                    for feat, val in zip(feature_names, shap_values[0])
                ],
                key=lambda x: x["importance"],
                reverse=True,
            )
            top_features = feature_importance[:3]

        except Exception:
            top_features = [{"name": "N/A", "importance": 0.0}] * 3

        return {
            "probability": probability,
            "label": label,
            "top_features_shap": top_features,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")
