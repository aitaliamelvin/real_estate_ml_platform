# ==============================
# 1. IMPORTS
# ==============================

import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import joblib
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# ==============================
# 2. LOAD MODEL
# ==============================

model = joblib.load("../models/real_estate_model.pkl")
df = pd.read_csv("../data/real_estate_dataset.csv")

# ==============================
# 3. TITLE
# ==============================

st.sidebar.title("🏠 Real Estate Price Prediction")

st.title("Estimation immobilière avec Machine Learning")

st.markdown(
    """
    Cette application utilise :
    
    - Machine Learning
    - IA explicative
    - GPT Analysis
    
    pour analyser des biens immobiliers.
    """
)

# ==============================
# 4. USER INPUTS
# ==============================

surface = st.sidebar.slider("Surface", 20, 300, 100)

rooms = st.sidebar.slider("Nombre de pièces", 1, 10, 4)

floor = st.sidebar.slider("Étage", 0, 20, 2)

balcony = st.sidebar.selectbox("Balcon", [0, 1])

parking = st.sidebar.selectbox("Parking", [0, 1])

age = st.sidebar.slider("Âge du bien", 0, 100, 10)

district = st.sidebar.selectbox(
    "Quartier",
    ["calme", "centre", "luxe", "populaire"]
)

# ==============================
# 5. ENCODING DISTRICT
# ==============================

district_calme = 0
district_centre = 0
district_luxe = 0
district_populaire = 0

if district == "calme":
    district_calme = 1

elif district == "centre":
    district_centre = 1

elif district == "luxe":
    district_luxe = 1

elif district == "populaire":
    district_populaire = 1

# ==============================
# 6. CREATE DATAFRAME
# ==============================

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

# ==============================
# 7. PREDICTION
# ==============================

prediction = model.predict(data)

# ==============================
# 8. DISPLAY RESULT
# ==============================

st.subheader("💰 Prix estimé")

st.metric(
    label="💰 Prix estimé",
    value=f"{prediction[0]:,.0f} €"
)

st.divider()

# ==============================
# 9. AI EXPLANATION
# ==============================

st.subheader("🤖 Analyse IA")

explanations = []

# surface
if surface > 150:
    explanations.append(
        "grande surface"
    )

# quartier
if district == "luxe":
    explanations.append(
        "quartier luxe"
    )

elif district == "centre":
    explanations.append(
        "quartier central"
    )

# parking
if parking == 1:
    explanations.append(
        "présence d'un parking"
    )

# balcon
if balcony == 1:
    explanations.append(
        "présence d'un balcon"
    )

# âge
if age < 10:
    explanations.append(
        "bien récent"
    )

elif age > 50:
    explanations.append(
        "bien ancien"
    )

# ==============================
# FINAL TEXT
# ==============================

if len(explanations) > 0:

    explanation_text = ", ".join(explanations)

    st.info(
        f"🤖 Analyse IA : le prix est principalement influencé par {explanation_text}."
    )

else:

    st.info(
        "🤖 Analyse IA : le bien possède des caractéristiques standards."
    )

# ==============================
# 10. GPT ANALYSIS
# ==============================

st.subheader("🧠 Analyse IA avancée")

try:

    prompt = f"""
    Analyse ce bien immobilier.

    Surface : {surface} m²
    Pièces : {rooms}
    Étage : {floor}
    Balcon : {balcony}
    Parking : {parking}
    Âge : {age}
    Quartier : {district}

    Prix estimé : {prediction[0]:,.0f} €

    Donne une analyse professionnelle courte et claire.
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    analysis = response.choices[0].message.content


    st.write(analysis)

except Exception as e:

    st.warning(
        "⚠️ API OpenAI indisponible ou quota dépassé."
    )

    st.code(str(e))

# ==============================
# 11. GRAPHS
# ==============================

st.divider()

st.subheader("📈 Prix selon surface")

fig, ax = plt.subplots()

ax.scatter(
    df["surface"],
    df["price"],
    alpha=0.3
)

ax.set_xlabel("Surface")

ax.set_ylabel("Prix")

st.pyplot(fig)

st.divider()

st.subheader("🏘️ Répartition des quartiers")

district_counts = df["district"].value_counts()

fig2, ax2 = plt.subplots()

ax2.bar(
    district_counts.index,
    district_counts.values
)

ax2.set_xlabel("Quartier")

ax2.set_ylabel("Nombre de biens")

st.pyplot(fig2)

st.divider()

market_average = df["price"].mean()

difference = prediction[0] - market_average

st.subheader("📊 Position sur le marché")

st.write(
    f"Prix moyen du marché : {market_average:,.0f} €"
)

st.write(
    f"Différence avec le bien estimé : {difference:,.0f} €"
)