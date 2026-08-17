# ==============================
# 1. IMPORTS
# ==============================

from pathlib import Path
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.ticker import FuncFormatter

from dotenv import load_dotenv
from openai import OpenAI

# ==============================
# 2. PATHS
# ==============================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "real_estate_model.pkl"
DATA_PATH = BASE_DIR / "data" / "real_estate_dataset.csv"


# ==============================
# 3. ENVIRONMENT
# ==============================

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ==============================
# 4. PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="Real Estate ML",
    page_icon="🏠",
    layout="wide",
)


# ==============================
# 5. CUSTOM CSS
# ==============================

st.markdown(
    """
    <style>

    /* GENERAL */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    h1 {
        font-size: 2.35rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
        margin-bottom: 0.5rem !important;
    }

    h2, h3 {
        font-weight: 700 !important;
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #f7f8fa;
        border-right: 1px solid #e6e8eb;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    /* METRIC CARDS */
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e7e9ed;
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.04);
    }

    div[data-testid="stMetric"] label {
        font-size: 0.95rem;
        color: #6b7280;
    }

    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }

    /* INFO CARDS */
    .custom-card {
        background: #ffffff;
        border: 1px solid #e7e9ed;
        border-radius: 16px;
        padding: 20px;
        min-height: 165px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.04);
    }

    .custom-card h4 {
        margin-top: 0;
        margin-bottom: 12px;
        font-size: 1.05rem;
    }

    .custom-card ul {
        padding-left: 1.2rem;
        margin-bottom: 0;
    }

    /* SECTION */
    .section-title {
        font-size: 1.55rem;
        font-weight: 750;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }

    /* HERO */
    .hero-subtitle {
        font-size: 1.05rem;
        color: #5f6368;
        line-height: 1.6;
        max-width: 900px;
        margin-bottom: 1.5rem;
    }

    /* BUTTON */
    div.stButton > button,
    div.stFormSubmitButton > button {
        border-radius: 12px;
        height: 46px;
        font-weight: 650;
    }

    /* DISCLAIMER */
    .disclaimer {
        font-size: 0.82rem;
        color: #777;
        margin-top: 1.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================
# 6. LOAD MODEL & DATA
# ==============================


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


model = load_model()
df = load_data()


# ==============================
# 7. HEADER
# ==============================

st.markdown(
    """
    <h1>🏠 Estimation immobilière avec Machine Learning</h1>

    <div class="hero-subtitle">
        Une application de démonstration qui estime le prix d’un bien immobilier,
        analyse son positionnement par rapport au marché simulé et propose une
        interprétation complémentaire grâce à l’intelligence artificielle.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="
        display:flex;
        gap:8px;
        margin-top:8px;
        margin-bottom:8px;
        flex-wrap:wrap;
    ">
        <span style="padding:5px 10px;border:1px solid #e5e7eb;border-radius:999px;font-size:0.8rem;">
            Machine Learning
        </span>
        <span style="padding:5px 10px;border:1px solid #e5e7eb;border-radius:999px;font-size:0.8rem;">
            FastAPI
        </span>
        <span style="padding:5px 10px;border:1px solid #e5e7eb;border-radius:999px;font-size:0.8rem;">
            Streamlit
        </span>
        <span style="padding:5px 10px;border:1px solid #e5e7eb;border-radius:999px;font-size:0.8rem;">
            OpenAI
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()


# ==============================
# 8. SIDEBAR FORM
# ==============================

st.sidebar.markdown("## 🏡 Caractéristiques du bien")

with st.sidebar.form("property_form"):

    surface = st.slider(
        "Surface (m²)",
        min_value=20,
        max_value=300,
        value=100,
    )

    rooms = st.slider(
        "Nombre de pièces",
        min_value=1,
        max_value=10,
        value=4,
    )

    floor = st.slider(
        "Étage",
        min_value=0,
        max_value=20,
        value=2,
    )

    balcony = st.selectbox(
        "Balcon",
        options=[0, 1],
        format_func=lambda x: "Oui" if x == 1 else "Non",
    )

    parking = st.selectbox(
        "Parking",
        options=[0, 1],
        format_func=lambda x: "Oui" if x == 1 else "Non",
    )

    age = st.slider(
        "Âge du bien",
        min_value=0,
        max_value=100,
        value=10,
    )

    district = st.selectbox(
        "Quartier",
        ["calme", "centre", "luxe", "populaire"],
    )

    estimate_button = st.form_submit_button(
        "Estimer le bien",
        use_container_width=True,
    )


# ==============================
# 9. PREDICTION
# ==============================

if estimate_button:

    property_data = pd.DataFrame(
        [
            {
                "surface": surface,
                "rooms": rooms,
                "floor": floor,
                "balcony": balcony,
                "parking": parking,
                "age": age,
                "district": district,
            }
        ]
    )

    prediction = model.predict(property_data)[0]

    st.session_state["prediction"] = float(prediction)

    st.session_state["property_data"] = {
        "surface": surface,
        "rooms": rooms,
        "floor": floor,
        "balcony": balcony,
        "parking": parking,
        "age": age,
        "district": district,
    }


# ==============================
# 10. RESULTS
# ==============================

if "prediction" in st.session_state:

    prediction = st.session_state["prediction"]
    property_info = st.session_state["property_data"]

    surface = property_info["surface"]
    rooms = property_info["rooms"]
    floor = property_info["floor"]
    balcony = property_info["balcony"]
    parking = property_info["parking"]
    age = property_info["age"]
    district = property_info["district"]

    price_per_m2 = prediction / surface
    market_average = df["price"].mean()
    difference = prediction - market_average

    st.markdown(
        '<div class="section-title">💰 Estimation du bien</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Prix estimé",
            f"{prediction:,.0f} €",
        )

    with col2:
        st.metric(
            "Prix estimé / m²",
            f"{price_per_m2:,.0f} €",
        )

    with col3:
        st.metric(
            "Écart avec la moyenne",
            f"{difference:+,.0f} €",
        )

    st.divider()

    # ==============================
    # 11. ANALYSIS
    # ==============================

    positive_factors = []
    negative_factors = []

    if surface >= 150:
        positive_factors.append("Grande surface")

    if district == "luxe":
        positive_factors.append("Quartier haut de gamme")

    elif district == "centre":
        positive_factors.append("Localisation centrale")

    if parking == 1:
        positive_factors.append("Parking disponible")

    if balcony == 1:
        positive_factors.append("Présence d'un balcon")

    if age < 10:
        positive_factors.append("Bien récent")

    if age > 50:
        negative_factors.append("Bien relativement ancien")

    st.markdown(
        '<div class="section-title">🔍 Analyse du bien</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        positive_html = ""

        if positive_factors:
            positive_html = "".join(f"<li>{factor}</li>" for factor in positive_factors)
        else:
            positive_html = (
                "<li>Aucun facteur particulièrement valorisant détecté.</li>"
            )

        st.markdown(
            f"""
            <div class="custom-card">
                <h4>✅ Points valorisants</h4>
                <ul>{positive_html}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        negative_html = ""

        if negative_factors:
            negative_html = "".join(f"<li>{factor}</li>" for factor in negative_factors)
        else:
            negative_html = "<li>Aucun point défavorable majeur détecté.</li>"

        st.markdown(
            f"""
            <div class="custom-card">
                <h4>⚠️ Points à surveiller</h4>
                <ul>{negative_html}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ==============================
    # 12. MARKET POSITION
    # ==============================

    district_data = df[df["district"] == district]
    district_average = district_data["price"].mean()

    st.markdown(
        '<div class="section-title">📊 Positionnement sur le marché</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Prix moyen global",
            f"{market_average:,.0f} €",
        )

    with col2:
        st.metric(
            f"Prix moyen — quartier {district}",
            f"{district_average:,.0f} €",
        )

    if prediction > district_average:
        st.info(
            f"Le bien est estimé au-dessus du prix moyen du quartier « {district} »."
        )
    else:
        st.info(
            f"Le bien est estimé en dessous du prix moyen du quartier « {district} »."
        )

    st.divider()

    # ==============================
    # 13. GRAPH 1
    # ==============================

    st.markdown(
        '<div class="section-title">📈 Prix en fonction de la surface</div>',
        unsafe_allow_html=True,
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(
        df["surface"],
        df["price"],
        alpha=0.12,
        s=18,
    )

    ax.scatter(
        surface,
        prediction,
        s=170,
        marker="X",
        label="Bien estimé",
        zorder=5,
    )

    ax.annotate(
        f"{prediction:,.0f} €",
        (surface, prediction),
        xytext=(12, 12),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
    )

    ax.set_xlabel("Surface (m²)")
    ax.set_ylabel("Prix")

    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:.0f} k€"))

    ax.grid(alpha=0.12)
    ax.legend(frameon=False)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)

    # ==============================
    # 14. GRAPH 2
    # ==============================

    st.markdown(
        '<div class="section-title">🏘️ Répartition des biens par quartier</div>',
        unsafe_allow_html=True,
    )

    district_counts = df["district"].value_counts()

    fig2, ax2 = plt.subplots(figsize=(10, 4.5))

    ax2.bar(
        district_counts.index,
        district_counts.values,
    )

    ax2.set_xlabel("Quartier")
    ax2.set_ylabel("Nombre de biens")
    ax2.grid(axis="y", alpha=0.15)

    st.pyplot(fig2, use_container_width=True)

    plt.close(fig2)

    st.divider()

    # ==============================
    # 15. AI ANALYSIS
    # ==============================

    st.markdown(
        '<div class="section-title">🧠 Analyse IA avancée</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "L’analyse IA est facultative et fournit une lecture complémentaire "
        "des caractéristiques du bien."
    )

    if st.button(
        "Générer l'analyse IA",
        use_container_width=False,
    ):

        if not OPENAI_API_KEY:

            st.warning("Aucune clé OpenAI n'a été configurée.")

        else:

            try:

                client = OpenAI(api_key=OPENAI_API_KEY)

                prompt = f"""
                Analyse ce bien immobilier de manière professionnelle.

                Caractéristiques :
                - Surface : {surface} m²
                - Nombre de pièces : {rooms}
                - Étage : {floor}
                - Balcon : {"Oui" if balcony else "Non"}
                - Parking : {"Oui" if parking else "Non"}
                - Âge : {age} ans
                - Quartier : {district}

                Prix estimé :
                {prediction:,.0f} €

                Prix moyen du dataset :
                {market_average:,.0f} €

                Prix moyen du quartier :
                {district_average:,.0f} €

                Donne :
                - les principaux points positifs ;
                - les éventuels points faibles ;
                - le positionnement du bien ;
                - une conclusion courte.

                Ne présente pas l'estimation comme une expertise immobilière réelle.
                """

                with st.spinner("Analyse en cours..."):

                    response = client.chat.completions.create(
                        model="gpt-4.1-mini",
                        messages=[
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ],
                    )

                analysis = response.choices[0].message.content

                st.success("Analyse générée")

                st.write(analysis)

            except Exception as e:

                st.warning("L'analyse IA est actuellement indisponible.")

                with st.expander("Détails techniques"):
                    st.code(str(e))


# ==============================
# 16. INITIAL STATE
# ==============================

else:

    st.info(
        "👈 Renseignez les caractéristiques du bien puis cliquez sur « Estimer le bien »."
    )


# ==============================
# 17. FOOTER
# ==============================

st.divider()

st.markdown(
    """
    <div class="disclaimer">
        Projet de démonstration Machine Learning — données synthétiques.
        Les estimations fournies ne constituent pas une expertise immobilière professionnelle.
    </div>
    """,
    unsafe_allow_html=True,
)
