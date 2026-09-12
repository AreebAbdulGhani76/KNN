import os
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Literal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Initialize FastAPI App
app = FastAPI(
    title="Loan Approval Prediction API (KNN)",
    description="REST API serving a trained and tuned K-Nearest Neighbors classifier to predict loan approvals.",
    version="1.0.0"
)

# Enable CORS for Next.js and frontend dev servers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load artifacts
ARTIFACTS_PATH = os.path.join(os.path.dirname(__file__), "loan_knn_model.joblib")
if not os.path.exists(ARTIFACTS_PATH):
    # Fallback to current directory
    ARTIFACTS_PATH = "loan_knn_model.joblib"

if not os.path.exists(ARTIFACTS_PATH):
    raise RuntimeError(f"Model artifacts not found at {ARTIFACTS_PATH}. Please run train_knn.py first.")

artifacts = joblib.load(ARTIFACTS_PATH)
model = artifacts['model']
default_model = artifacts['default_model']
scaler = artifacts['scaler']
feature_cols = artifacts['feature_cols']
category_mappings = artifacts['category_mappings']
imputation_values = artifacts['imputation_values']
metrics_info = artifacts['metrics']

class ApplicantData(BaseModel):
    Gender: Literal["Male", "Female"] = Field(..., example="Male")
    Married: Literal["Yes", "No"] = Field(..., example="Yes")
    Dependents: Literal["0", "1", "2", "3+"] = Field(..., example="1")
    Education: Literal["Graduate", "Not Graduate"] = Field(..., example="Graduate")
    Self_Employed: Literal["Yes", "No"] = Field(..., example="No")
    ApplicantIncome: float = Field(..., ge=0, example=5000)
    CoapplicantIncome: float = Field(0.0, ge=0, example=1500)
    LoanAmount: float = Field(..., gt=0, example=150)  # in thousands
    Loan_Amount_Term: float = Field(360.0, gt=0, example=360)  # in months
    Credit_History: float = Field(..., ge=0, le=1, example=1.0)
    Property_Area: Literal["Urban", "Semiurban", "Rural"] = Field(..., example="Urban")
    model_choice: Optional[Literal["tuned", "default"]] = "tuned"

class PredictionResponse(BaseModel):
    status: str
    prediction: Literal["Approved", "Rejected"]
    is_approved: bool
    confidence: float
    probability_approved: float
    probability_rejected: float
    model_used: str
    risk_level: str
    key_factors: list[str]

@app.get("/")
def read_root():
    return {
        "message": "Loan Approval Prediction API is up and running!",
        "model": "K-Nearest Neighbors (KNN)",
        "endpoints": {
            "predict": "POST /predict",
            "model_info": "GET /model-info",
            "health": "GET /health"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "loan-knn-api"}

@app.get("/model-info")
def get_model_info():
    return {
        "model_type": "K-Nearest Neighbors (KNN)",
        "default_k": int(default_model.n_neighbors),
        "default_accuracy": round(metrics_info['default_accuracy'] * 100, 2),
        "tuned_parameters": metrics_info['best_params'],
        "tuned_accuracy": round(metrics_info['tuned_accuracy'] * 100, 2),
        "feature_count": len(feature_cols),
        "feature_names": feature_cols,
        "k_experiments": metrics_info['k_experiments']
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_loan_status(applicant: ApplicantData):
    try:
        # 1. Encode Categorical variables
        encoded_data = {
            'Gender': category_mappings['Gender'].get(applicant.Gender, 1),
            'Married': category_mappings['Married'].get(applicant.Married, 1),
            'Dependents': category_mappings['Dependents'].get(applicant.Dependents, 0),
            'Education': category_mappings['Education'].get(applicant.Education, 1),
            'Self_Employed': category_mappings['Self_Employed'].get(applicant.Self_Employed, 0),
            'Property_Area': category_mappings['Property_Area'].get(applicant.Property_Area, 2),
            'ApplicantIncome': applicant.ApplicantIncome,
            'CoapplicantIncome': applicant.CoapplicantIncome,
            'LoanAmount': applicant.LoanAmount,
            'Loan_Amount_Term': applicant.Loan_Amount_Term,
            'Credit_History': applicant.Credit_History,
        }

        # 2. Build feature vector aligned with feature_cols as DataFrame
        df_input = pd.DataFrame([encoded_data])[feature_cols]

        # 3. Standardize using fitted scaler
        vector_scaled = scaler.transform(df_input)

        # 4. Select model
        selected_model = best_model if applicant.model_choice == "tuned" else default_model
        model_name = f"Tuned KNN (k={selected_model.n_neighbors}, weights='{selected_model.weights}')" if applicant.model_choice == "tuned" else f"Default KNN (k={selected_model.n_neighbors})"

        # 5. Predict class and probabilities
        pred_int = int(selected_model.predict(vector_scaled)[0])
        probabilities = selected_model.predict_proba(vector_scaled)[0]
        prob_rejected = float(probabilities[0])
        prob_approved = float(probabilities[1])

        is_approved = (pred_int == 1)
        confidence = prob_approved if is_approved else prob_rejected

        # Risk assessment & key factors
        factors = []
        if applicant.Credit_History == 1.0:
            factors.append("Positive credit history verified")
        else:
            factors.append("No/poor credit history detected (High risk indicator)")

        total_income = applicant.ApplicantIncome + applicant.CoapplicantIncome
        if total_income > 0:
            loan_to_income = (applicant.LoanAmount * 1000) / total_income
            if loan_to_income > 2.5:
                factors.append(f"High loan-to-monthly income ratio (~{loan_to_income:.1f}x)")
            else:
                factors.append("Healthy loan-to-income ratio")
        
        if applicant.Property_Area == "Semiurban":
            factors.append("Property situated in Semiurban zone (highest empirical approval rate)")
        
        if prob_approved >= 0.75:
            risk = "Low"
        elif prob_approved >= 0.50:
            risk = "Moderate"
        else:
            risk = "High"

        return PredictionResponse(
            status="success",
            prediction="Approved" if is_approved else "Rejected",
            is_approved=is_approved,
            confidence=round(confidence * 100, 2),
            probability_approved=round(prob_approved * 100, 2),
            probability_rejected=round(prob_rejected * 100, 2),
            model_used=model_name,
            risk_level=risk,
            key_factors=factors
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Make sure best_model is aliased to model
best_model = model

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
