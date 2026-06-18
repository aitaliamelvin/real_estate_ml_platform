# ==============================
# 1. IMPORTS
# ==============================

import pandas as pd
import numpy as np

# ==============================
# 2. RANDOM SEED
# ==============================

np.random.seed(42)

# ==============================
# 3. NOMBRE DE LIGNES
# ==============================

n = 10000

# ==============================
# 4. FEATURES
# ==============================

surface = np.random.randint(20, 250, n)

rooms = np.random.randint(1, 8, n)

floor = np.random.randint(0, 15, n)

balcony = np.random.choice([0, 1], n)

parking = np.random.choice([0, 1], n)

age = np.random.randint(0, 80, n)

district = np.random.choice(
    ["centre", "calme", "populaire", "luxe"],
    n
)

# ==============================
# 5. PRIX RÉALISTE
# ==============================

district_bonus = {
    "centre": 80000,
    "calme": 40000,
    "populaire": -20000,
    "luxe": 150000
}

price = (
    surface * 3500
    + rooms * 10000
    + floor * 2000
    + balcony * 15000
    + parking * 20000
    - age * 1000
).astype(float)

# ajout bonus quartier
price += np.array([district_bonus[d] for d in district])

# bruit réaliste
noise = np.random.normal(0, 30000, n)

price += noise

# ==============================
# 6. DATAFRAME
# ==============================

df = pd.DataFrame({
    "surface": surface,
    "rooms": rooms,
    "floor": floor,
    "balcony": balcony,
    "parking": parking,
    "age": age,
    "district": district,
    "price": price
})

# ==============================
# 7. SAVE CSV
# ==============================

df.to_csv("real_estate_dataset.csv", index=False)

print(df.head())

print("\nDataset créé :", df.shape)