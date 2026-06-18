# ==============================
# 1. IMPORTS
# ==============================

import matplotlib.pyplot as plt
import pandas as pd
import os 
dossier = os.path.dirname(__file__)
fichier = os.path.join(dossier, "real_estate_dataset.csv")

# ==============================
# 2. LOAD DATASET
# ==============================

df = pd.read_csv(fichier)

# ==============================
# 3. APERÇU GÉNÉRAL
# ==============================

print("\n===== HEAD =====")
print(df.head())

# ==============================
# 4. INFORMATIONS DATASET
# ==============================

print("\n===== INFO =====")
print(df.info())

# ==============================
# 5. STATISTIQUES
# ==============================

print("\n===== DESCRIBE =====")
print(df.describe())

# ==============================
# 6. VALEURS MANQUANTES
# ==============================

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# ==============================
# 7. TYPES DE DONNÉES
# ==============================

print("\n===== DATA TYPES =====")
print(df.dtypes)

# ==============================
# 8. DISTRIBUTION QUARTIERS
# ==============================

print("\n===== DISTRICTS =====")
print(df["district"].value_counts())

# ==============================
# 9. DISTRIBUTION DES PRIX
# ==============================

plt.figure(figsize=(8,5))

plt.hist(df["price"], bins=30)

plt.title("Distribution des prix")

plt.xlabel("Prix")

plt.ylabel("Nombre de biens")

plt.show()

# ==============================
# 10. SURFACE VS PRIX
# ==============================

plt.figure(figsize=(8,5))

plt.scatter(df["surface"], df["price"])

plt.title("Surface vs Prix")

plt.xlabel("Surface")

plt.ylabel("Prix")

plt.show()

# ==============================
# 11. PRIX PAR QUARTIER
# ==============================

plt.figure(figsize=(8,5))

df.groupby("district")["price"].mean().plot(kind="bar")

plt.title("Prix moyen par quartier")

plt.xlabel("Quartier")

plt.ylabel("Prix moyen")

plt.show()

# ==============================
# 12. CORRÉLATIONS
# ==============================

correlation = df.corr(numeric_only=True)

print("\n===== CORRELATION =====")

print(correlation)