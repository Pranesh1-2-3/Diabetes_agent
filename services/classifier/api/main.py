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

@app.get("/")
def root():
    """Root endpoint."""
    return {"msg": "Diabetes Agent API is running"}

async def predict_risk(patient_data: dict) -> dict:
    """Wrapper function to predict diabetes risk from patient data."""
    # Convert patient data to model input format
    model_input = {
        "Pregnancies": 0,  # default for now
        "Glucose": patient_data.get("glucose", 0),
        "BloodPressure": float(patient_data.get("blood_pressure", "0/0").split("/")[0]),  # systolic
        "SkinThickness": 0,  # not provided in patient data
        "Insulin": 0,  # not provided in patient data
        "BMI": patient_data.get("bmi", 0),
        "DiabetesPedigreeFunction": 1.0 if patient_data.get("family_history") == "yes" else 0.0,
        "Age": patient_data.get("age", 0)
    }
    
    # Make prediction using the API endpoint
    input_data = PredictionInput(**model_input)
    result = await predict(input_data)   # <-- result is a dict
    
    # Format response
    return {
        "prediction": "high_risk" if result["label"] == 1 else "low_risk",
        "probability": result["probability"],
        "top_features": [f["name"] for f in result.get("top_features_shap", [])]
    }


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
