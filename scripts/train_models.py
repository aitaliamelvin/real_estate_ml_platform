# ==============================
# 1. IMPORTS
# ==============================

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "real_estate_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "real_estate_model.pkl"

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import r2_score

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ==============================
# 2. LOAD DATASET
# ==============================

df = pd.read_csv(DATA_PATH)

# ==============================
# 3. FEATURES / TARGET
# ==============================

X = df[
    [
        "surface",
        "rooms",
        "floor",
        "balcony",
        "parking",
        "age",
        "district",
    ]
]

y = df["price"]

categorical_features = ["district"]

numeric_features = [
    "surface",
    "rooms",
    "floor",
    "balcony",
    "parking",
    "age",
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "district_encoder",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        )
    ],
    remainder="passthrough",
)

# ==============================
# 4. ENCODING
# ==============================


# 5. TRAIN / TEST
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 6. MODÈLES
# ==============================

pipeline_lr = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression()),
    ]
)

pipeline_dt = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", DecisionTreeRegressor(random_state=42)),
    ]
)

pipeline_rf = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(random_state=42)),
    ]
)

# ==============================
# 7. ENTRAÎNEMENT
# ==============================

pipeline_lr.fit(X_train, y_train)

pipeline_dt.fit(X_train, y_train)

pipeline_rf.fit(X_train, y_train)

# ==============================
# 8. PREDICTIONS
# ==============================

pred_lr = pipeline_lr.predict(X_test)

pred_dt = pipeline_dt.predict(X_test)

pred_rf = pipeline_rf.predict(X_test)

# ==============================
# 9. ÉVALUATION
# ==============================

print("\n===== LINEAR REGRESSION =====")

print("MAE :", mean_absolute_error(y_test, pred_lr))

print("MSE :", mean_squared_error(y_test, pred_lr))

print("R² :", r2_score(y_test, pred_lr))

# ------------------------------

print("\n===== DECISION TREE =====")

print("MAE :", mean_absolute_error(y_test, pred_dt))

print("MSE :", mean_squared_error(y_test, pred_dt))

print("R² :", r2_score(y_test, pred_dt))

# ------------------------------

print("\n===== RANDOM FOREST =====")

print("MAE :", mean_absolute_error(y_test, pred_rf))

print("MSE :", mean_squared_error(y_test, pred_rf))

print("R² :", r2_score(y_test, pred_rf))

# ==============================
# 10. VISUALISATION
# ==============================

plt.figure(figsize=(8, 6))

plt.scatter(y_test, pred_lr)

plt.xlabel("Vrais prix")

plt.ylabel("Prix prédits")

plt.title("Linear Regression : vrais prix vs prédictions")

plt.show()

plt.figure(figsize=(8, 6))

plt.scatter(y_test, pred_rf)

plt.xlabel("Vrais prix")

plt.ylabel("Prix prédits")

plt.title("Random Forest : vrais prix vs prédictions")

plt.show()

# ==============================
# 11. SAVE MODEL
# ==============================

joblib.dump(pipeline_lr, MODEL_PATH)

print("\nModèle sauvegardé !")
