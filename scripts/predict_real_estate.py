# ==============================
# 1. IMPORTS
# ==============================

import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "real_estate_model.pkl"


# ==============================
# 2. LOAD MODEL
# ==============================

model = joblib.load(MODEL_PATH)

# ==============================
# 3. NEW DATA
# ==============================

sample = pd.DataFrame(
    [
        {
            "surface": 100,
            "rooms": 4,
            "floor": 2,
            "balcony": 1,
            "parking": 1,
            "age": 10,
            "district": "centre",
        }
    ]
)

# ==============================
# 4. PREDICTION
# ==============================

prediction = model.predict(sample)

print(f"Prix estimé : {prediction[0]:,.0f} €")
