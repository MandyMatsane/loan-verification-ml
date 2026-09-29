from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Loan Approval API")

model = joblib.load("loan_model.pkl")


class LoanRequest(BaseModel):
    no_of_dependents: int
    education: str
    self_employed: str
    income_annum: float
    loan_amount: float
    loan_term: int
    cibil_score: int
    residential_assets_value: float
    commercial_assets_value: float
    luxury_assets_value: float
    bank_asset_value: float


@app.get("/")
def home():
    return {
        "status": "running",
        "service": "Loan Approval API"
    }


@app.post("/api/ml/predict")
def predict(data: LoanRequest):

    features = pd.DataFrame([{
        "no_of_dependents": data.no_of_dependents,
        "education": 1 if data.education == "Graduate" else 0,
        "self_employed": 1 if data.self_employed == "Yes" else 0,
        "income_annum": data.income_annum,
        "loan_amount": data.loan_amount,
        "loan_term": data.loan_term,
        "cibil_score": data.cibil_score,
        "residential_assets_value": data.residential_assets_value,
        "commercial_assets_value": data.commercial_assets_value,
        "luxury_assets_value": data.luxury_assets_value,
        "bank_asset_value": data.bank_asset_value,
    }])

    prediction = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]

    credit_risk = "low"

    if data.cibil_score < 500:
        credit_risk = "high"
    elif data.cibil_score < 650:
        credit_risk = "medium"

    confidence_score = round(float(probabilities.max()) * 100, 2)

    reasoning_notes = (
        f"Decision: {prediction}. "
        f"CIBIL score: {data.cibil_score}. "
        f"Model confidence: {confidence_score}%."
    )

    return {
        "eligibility_outcome": prediction,
        "confidence_score": confidence_score,
        "fraud_risk_score": credit_risk,
        "reasoning_notes": reasoning_notes
    }