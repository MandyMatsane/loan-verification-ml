import joblib
import pandas as pd

model = joblib.load("loan_model.pkl")

data = pd.DataFrame([{
    "no_of_dependents": 2,
    "education": 1,
    "self_employed": 0,
    "income_annum": 500,
    "loan_amount": 140,
    "loan_term": 10,
    "cibil_score": 0,
    "residential_assets_value": 500,
    "commercial_assets_value": 500,
    "luxury_assets_value": 100,
    "bank_asset_value": 300
}])

prediction = model.predict(data)[0]
probability = model.predict_proba(data)[0]

confidence = max(probability) * 100

print(f"Decision: {prediction}")
print(f"Confidence: {confidence:.2f}%")