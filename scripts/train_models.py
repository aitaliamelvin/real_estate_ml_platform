# ==============================
# 1. IMPORTS
# ==============================

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import os 
dossier = os.path.dirname(__file__)
fichier = os.path.join(dossier, "real_estate_dataset.csv")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

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
# 4. ENCODING
# ==============================

encoder = OneHotEncoder(sparse_output=False)

district_encoded = encoder.fit_transform(X[["district"]])

district_df = pd.DataFrame(
    district_encoded,
    columns=encoder.get_feature_names_out(["district"])
)

X = X.drop("district", axis=1)

X = pd.concat(
    [X.reset_index(drop=True),
     district_df.reset_index(drop=True)],
    axis=1
)

# ==============================
# 5. TRAIN / TEST
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#==============================
# 6. MODÈLES
# ==============================

model_lr = LinearRegression()

model_dt = DecisionTreeRegressor(
    max_depth=10,
    random_state=42
)

model_rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

# ==============================
# 7. ENTRAÎNEMENT
# ==============================

model_lr.fit(X_train, y_train)

model_dt.fit(X_train, y_train)

model_rf.fit(X_train, y_train)

# ==============================
# 8. PREDICTIONS
# ==============================

pred_lr = model_lr.predict(X_test)

pred_dt = model_dt.predict(X_test)

pred_rf = model_rf.predict(X_test)

# ==============================
# 9. ÉVALUATION
# ==============================

print("\n===== LINEAR REGRESSION =====")

print("MAE :", mean_absolute_error(y_test, pred_lr))

print("R2 :", r2_score(y_test, pred_lr))

# ------------------------------

print("\n===== DECISION TREE =====")

print("MAE :", mean_absolute_error(y_test, pred_dt))

print("R2 :", r2_score(y_test, pred_dt))

# ------------------------------

print("\n===== RANDOM FOREST =====")

print("MAE :", mean_absolute_error(y_test, pred_rf))

print("R2 :", r2_score(y_test, pred_rf))

# ==============================
# 10. VISUALISATION
# ==============================

plt.figure(figsize=(8,6))

plt.scatter(y_test, pred_lr)

plt.xlabel("Vrais prix")

plt.ylabel("Prix prédits")

plt.title("Linear Regression : vrais prix vs prédictions")

plt.show()

plt.figure(figsize=(8,6))

plt.scatter(y_test, pred_rf)

plt.xlabel("Vrais prix")

plt.ylabel("Prix prédits")

plt.title("Random Forest : vrais prix vs prédictions")

plt.show()

# ==============================
# 11. SAVE MODEL
# ==============================

joblib.dump(model_lr, "real_estate_model.pkl")

print("\nModèle sauvegardé !")