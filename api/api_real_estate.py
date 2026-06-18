# ==============================
# 1. IMPORTS
# ==============================

from fastapi import FastAPI

import joblib
import pandas as pd

# ==============================
# 2. LOAD MODEL
# ==============================

model = joblib.load("ML/real_estate_ml/api/real_estate_model.pkl")

# ==============================
# 3. FASTAPI
# ==============================

app = FastAPI()

# ==============================
# 4. HOME
# ==============================

@app.get("/")
def home():

    return {
        "message": "API Real Estate ML fonctionne"
    }

# ==============================
# 5. PREDICT
# ==============================

@app.get("/predict")

def predict(
    surface: float,
    rooms: int,
    floor: int,
    balcony: int,
    parking: int,
    age: int,

    district_calme: int,
    district_centre: int,
    district_luxe: int,
    district_populaire: int
):

    # --------------------------
    # CREATE DATAFRAME
    # --------------------------

    data = pd.DataFrame([{
        "surface": surface,
        "rooms": rooms,
        "floor": floor,
        "balcony": balcony,
        "parking": parking,
        "age": age,

        "district_calme": district_calme,
        "district_centre": district_centre,
        "district_luxe": district_luxe,
        "district_populaire": district_populaire
    }])

    # --------------------------
    # PREDICTION
    # --------------------------

    prediction = model.predict(data)

    # --------------------------
    # RETURN RESULT
    # --------------------------

    return {
        "prix_estime": round(prediction[0], 2)
    }