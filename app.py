from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

app = Flask(__name__)

# Load model and scaler
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load dataset for metrics
data = pd.read_csv("churn_data.csv")

@app.route("/")
def home():
    return render_template("index.html")

# -------------------------------
# Endpoint 1: Model Metrics
# -------------------------------
@app.route("/metrics", methods=["GET"])
def metrics():
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    data['Gender'] = le.fit_transform(data['Gender'])

    X = data.drop(["CustomerID", "Churn"], axis=1)
    y = data["Churn"]

    X = scaler.transform(X)
    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred).tolist()

    return jsonify({
        "accuracy": round(acc, 4),
        "confusion_matrix": cm
    })

# -------------------------------
# Endpoint 2: Predict Customer
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    input_data = request.json

    features = [
        input_data["Age"],
        input_data["Gender"],
        input_data["MonthlySpend"],
        input_data["Tenure"],
        input_data["VisitsPerMonth"]
    ]

    features = scaler.transform([features])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    return jsonify({
        "prediction": int(prediction),
        "probability": round(float(probability), 3),
        "risk_level": (
            "High" if probability > 0.7 else
            "Medium" if probability > 0.4 else
            "Low"
        )
    })


# -------------------------------
# Endpoint 3: Dataset for Charts
# -------------------------------
@app.route("/data", methods=["GET"])
def get_data():
    from sklearn.preprocessing import LabelEncoder
    
    df = pd.read_csv("churn_data.csv")

    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])

    # Send only required columns
    result = df[["Age", "Gender", "MonthlySpend", "Tenure", "VisitsPerMonth", "Churn"]]

    return jsonify(result.to_dict(orient="records"))




# -------------------------------
# Endpoint: Feature Importance
# -------------------------------
@app.route("/feature-importance", methods=["GET"])
def feature_importance():

    feature_names = [
        "Age",
        "Gender",
        "MonthlySpend",
        "Tenure",
        "VisitsPerMonth"
    ]

    importances = model.feature_importances_

    result = [
        {"feature": name, "importance": float(score)}
        for name, score in zip(feature_names, importances)
    ]

    return jsonify(result)




# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)