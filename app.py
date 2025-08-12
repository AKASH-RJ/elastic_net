# app.py
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained pipeline
pipeline = joblib.load("pipeline.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = None
    if request.method == "POST":
        try:
            area = float(request.form.get("area"))
            bedrooms = int(request.form.get("bedrooms"))
            bathrooms = int(request.form.get("bathrooms"))
            stories = int(request.form.get("stories"))
            age = float(request.form.get("age"))

            features = np.array([[area, bedrooms, bathrooms, stories, age]])
            pred = pipeline.predict(features)[0]
            prediction_text = f"Predicted House Price: ${pred:,.2f}"
        except Exception as e:
            prediction_text = f"Error: {str(e)}"

    return render_template("index.html", prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
