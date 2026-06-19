from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

# Create FastAPI app
app = FastAPI()

# Load trained model/pipeline
with open("RidgeModel.pkl", "rb") as f:
    model = pickle.load(f)

# Input schema
class HouseData(BaseModel):
    location: str
    total_sqft: float
    bath: int
    BHK: int

# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Bangalore House Price Prediction API"
    }

# Prediction endpoint
@app.post("/predict")
def predict(data: HouseData):

    input_df = pd.DataFrame({
        "location": [data.location],
        "total_sqft": [data.total_sqft],
        "bath": [data.bath],
        "BHK": [data.BHK]
    })

    prediction = model.predict(input_df)

    return {
        "predicted_price_lakh": round(float(prediction[0]), 2)
    }