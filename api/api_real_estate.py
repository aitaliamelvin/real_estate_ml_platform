# ==============================
# 1. IMPORTS
# ==============================

from fastapi import FastAPI

import joblib
import pandas as pd
from pydantic import BaseModel
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "real_estate_model.pkl"

# ==============================
# 2. LOAD MODEL
# ==============================

model = joblib.load(MODEL_PATH)

# ==============================
# 3. FASTAPI
# ==============================

app = FastAPI(
    title="Real Estate ML API",
    description="API de prédiction de prix immobilier",
    version="1.0.0",
)


class PropertyFeatures(BaseModel):
    surface: float
    rooms: int
    floor: int
    balcony: int
    parking: int
    age: int
    district: str


# ==============================
# 4. HOME
# ==============================


@app.get("/")
def root():
    return {"message": "Real Estate ML API", "status": "running"}


# ==============================
# 5. PREDICT
# ==============================


@app.post("/predict")
def predict_price(property_data: PropertyFeatures):

    input_data = pd.DataFrame(
        [
            {
                "surface": property_data.surface,
                "rooms": property_data.rooms,
                "floor": property_data.floor,
                "balcony": property_data.balcony,
                "parking": property_data.parking,
                "age": property_data.age,
                "district": property_data.district,
            }
        ]
    )

    # --------------------------
    # PREDICTION
    # --------------------------

    prediction = model.predict(input_data)[0]

    # --------------------------
    # RETURN RESULT
    # --------------------------

    return {"predicted_price": round(float(prediction), 2), "currency": "EUR"}
