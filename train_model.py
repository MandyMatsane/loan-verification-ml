import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import joblib

# ---- 1. Load the dataset ----
df = pd.read_csv("loan_approval_dataset.csv")

# This dataset is known to have extra spaces in column names and text values,
# so we clean that up first.
df.columns = df.columns.str.strip()
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

print("Columns found:", list(df.columns))
print(df.head())

# ---- 2. Select features and target ----
# Target: loan_status (Approved / Rejected)
target = "loan_status"

feature_cols = [
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]

X = df[feature_cols].copy()
y = df[target]

# ---- 3. Encode categorical columns (education, self_employed) ----
X["education"] = X["education"].map({"Graduate": 1, "Not Graduate": 0})
X["self_employed"] = X["self_employed"].map({"Yes": 1, "No": 0})

# ---- 4. Train/test split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- 5. Train the model ----
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ---- 6. Evaluate ----
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

precision_rejected = precision_score(y_test, y_pred, pos_label="Rejected")
recall_rejected = recall_score(y_test, y_pred, pos_label="Rejected")
print(f"Precision (Rejected class): {precision_rejected:.4f}")
print(f"Recall (Rejected class): {recall_rejected:.4f}")

print("\nFull classification report:")
print(classification_report(y_test, y_pred))

# ---- 7. Feature importance (for FR-10 / admin insights page) ----
importances = pd.Series(model.feature_importances_, index=feature_cols)
importances = importances.sort_values(ascending=False)
print("\nFeature importance ranking:")
print(importances)

# ---- 8. Save the trained model ----
joblib.dump(model, "loan_model.pkl")
print("\nModel saved as loan_model.pkl")