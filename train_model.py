import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib


# import os
# from pathlib import Path

# project_root = Path(os.environ["PROJECT_ROOT"])

# data_path = project_root / "data" / "churn_data.csv"
# scaler_path = project_root / "models" / "scaler.pkl"
# model_path = project_root / "models" / "model.pkl"

# Load Dataset
data = pd.read_csv("churn_ data.csv")

# Encode Gender
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])

# Features & Target
X = data.drop(["CustomerID", "Churn"], axis=1)
y = data["Churn"]

# Scale
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model & scaler
joblib.dump(scaler, "scaler.pkl")
joblib.dump(model, "churn_model.pkl")


print("Model saved successfully!")
