from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

# Load dataset
df = pd.read_csv("elastic_net_regression.csv")
X = df.drop("Price", axis=1)
y = df["Price"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Elastic Net model
model = ElasticNet(alpha=0.1, l1_ratio=0.5)
model.fit(X_scaled, y)

# Save model & scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form values
        area = float(request.form["area"])
        bedrooms = int(request.form["bedrooms"])
        bathrooms = int(request.form["bathrooms"])
        stories = int(request.form["stories"])
        age = int(request.form["age"])

        # Load model & scaler
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")

        # Prepare input
        features = pd.DataFrame([[area, bedrooms, bathrooms, stories, age]],
                                columns=["Area", "Bedrooms", "Bathrooms", "Stories", "Age"])
        features_scaled = scaler.transform(features)

        # Prediction
        prediction = model.predict(features_scaled)[0]
        return render_template("index.html", prediction_text=f"Predicted Price: ${prediction:,.2f}")
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
