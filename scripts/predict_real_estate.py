# ==============================
# 1. IMPORTS
# ==============================

import joblib
import pandas as pd

# ==============================
# 2. LOAD MODEL
# ==============================

model = joblib.load("real_estate_model.pkl")

# ==============================
# 3. NEW DATA
# ==============================

new_house = pd.DataFrame([{
    "surface": 120,
    "rooms": 5,
    "floor": 3,
    "balcony": 1,
    "parking": 1,
    "age": 10,

    "district_calme": 0,
    "district_centre": 0,
    "district_luxe": 1,
    "district_populaire": 0
}])

# ==============================
# 4. PREDICTION
# ==============================

prediction = model.predict(new_house)

print("\nPrix estimé :")

print(prediction[0])