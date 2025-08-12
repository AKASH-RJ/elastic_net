# model.py
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
import joblib

# Load dataset
df = pd.read_csv("ds.csv")  # <-- ensure the filename matches

# Features and target
X = df.drop(columns=["Price"])
y = df["Price"]

# Build pipeline with scaling and ElasticNetCV (automatically finds best alpha/l1_ratio)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("elastic_net_cv", ElasticNetCV(l1_ratio=[0.1,0.3,0.5,0.7,0.9], alphas=None, cv=5, random_state=42, max_iter=5000))
])

# Train
pipeline.fit(X, y)

# Save pipeline
joblib.dump(pipeline, "pipeline.pkl")
print("Model pipeline trained and saved as pipeline.pkl")
