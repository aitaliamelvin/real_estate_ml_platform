# ==============================
# 1. IMPORTS
# ==============================

import pandas as pd
import os 
dossier = os.path.dirname(__file__)
fichier = os.path.join(dossier, "real_estate_dataset.csv")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# ==============================
# 2. LOAD DATASET
# ==============================

df = pd.read_csv(fichier)

# ==============================
# 3. FEATURES / TARGET
# ==============================

X = df.drop("price", axis=1)

y = df["price"]

# ==============================
# 4. ENCODING DISTRICT
# ==============================

encoder = OneHotEncoder(sparse_output=False)

district_encoded = encoder.fit_transform(X[["district"]])

district_df = pd.DataFrame(
    district_encoded,
    columns=encoder.get_feature_names_out(["district"])
)

# ==============================
# 5. REMOVE OLD DISTRICT
# ==============================

X = X.drop("district", axis=1)

# ==============================
# 6. CONCAT NEW COLUMNS
# ==============================

X = pd.concat(
    [X.reset_index(drop=True),
     district_df.reset_index(drop=True)],
    axis=1
)

# ==============================
# 7. TRAIN / TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==============================
# 8. VERIFICATIONS
# ==============================

print("\n===== X TRAIN =====")
print(X_train.head())

print("\nShape X_train :", X_train.shape)
print("Shape X_test :", X_test.shape)

print("\nColonnes :")
print(X.columns)