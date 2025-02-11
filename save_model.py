import joblib
import tarfile
import os

# Load trained model
model = joblib.load("car_price_predictor.pkl")

# Save model inside a folder
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/car_price_predictor.pkl")

# Compress model folder into a .tar.gz file
with tarfile.open("model.tar.gz", "w:gz") as tar:
    tar.add("model", arcname=".")

print("Model successfully saved as model.tar.gz")
