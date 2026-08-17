# 🏠 Real Estate ML Platform

An end-to-end Machine Learning application for real estate price estimation, combining a scikit-learn pipeline, FastAPI, Streamlit and optional AI-powered analysis.

## 🚀 Live Demo

👉 **Streamlit App**
https://real-estate-ml-platform.streamlit.app/

---

## 📌 Project Overview

The goal of this project is to demonstrate the complete lifecycle of a Machine Learning application:

* dataset preparation
* exploratory data analysis
* preprocessing
* model training and comparison
* end-to-end ML pipeline
* REST API with FastAPI
* interactive web application with Streamlit
* optional AI-generated property analysis
* deployment to Streamlit Community Cloud

The application estimates the price of a property based on several characteristics and provides additional contextual information about the estimated property.

---

## ✨ Main Features

### Machine Learning

The model uses the following property characteristics:

* surface
* number of rooms
* floor
* balcony
* parking
* property age
* district

Categorical preprocessing is handled automatically through a scikit-learn pipeline using `OneHotEncoder`.

This allows the application to work directly with raw property information without manually encoding district categories.

### Interactive Estimation

The Streamlit interface allows users to configure a property and generate an estimated price.

The dashboard displays:

* estimated property price
* estimated price per square meter
* difference from the dataset average
* positive property characteristics
* potential weaknesses
* comparison with the selected district
* visual market positioning

### AI Analysis

An optional OpenAI-powered feature generates a short professional interpretation of the property based on:

* property characteristics
* predicted price
* dataset average
* district average

The AI analysis is deliberately separated from the Machine Learning prediction and only runs when requested by the user.

### FastAPI

The project also exposes the trained Machine Learning pipeline through a REST API.

Example request:

```json
{
  "surface": 100,
  "rooms": 4,
  "floor": 2,
  "balcony": 1,
  "parking": 1,
  "age": 10,
  "district": "centre"
}
```

Example response:

```json
{
  "predicted_price": 420000.0,
  "currency": "EUR"
}
```

---

## 🧠 Machine Learning Pipeline

The project uses an end-to-end scikit-learn pipeline:

```text
Raw property data
        ↓
ColumnTransformer
        ↓
OneHotEncoder
        ↓
Machine Learning model
        ↓
Price prediction
```

This architecture improves portability and ensures that preprocessing remains consistent between training and inference.

---

## 🤖 Models

Several regression algorithms are evaluated during training:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor

The selected model is saved together with its preprocessing pipeline using `joblib`.

Model file:

```text
models/real_estate_model.pkl
```

---

## 📊 Dataset

The project uses a synthetic real estate dataset containing approximately **10,000 properties**.

Main variables include:

* `surface`
* `rooms`
* `floor`
* `balcony`
* `parking`
* `age`
* `district`
* `price`

### Important limitation

The dataset is synthetic.

This project is therefore designed primarily to demonstrate:

* Machine Learning architecture
* preprocessing
* model serving
* API integration
* dashboard development
* deployment

It should **not** be interpreted as a production-grade real estate valuation system or a representation of the actual French property market.

---

## 🗂️ Project Structure

```text
real_estate_ml_platform/
│
├── api/
│   └── api_real_estate.py
│
├── dashboard/
│   └── dashboard_real_estate.py
│
├── data/
│   └── real_estate_dataset.csv
│
├── models/
│   └── real_estate_model.pkl
│
├── scripts/
│   ├── eda_real_estate.py
│   ├── preprocessing_real_estate.py
│   ├── predict_real_estate.py
│   └── train_models.py
│
├── screenshots/
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

### Machine Learning

* Python
* pandas
* NumPy
* scikit-learn
* joblib

### Backend

* FastAPI
* Uvicorn
* Pydantic

### Frontend

* Streamlit
* Matplotlib

### AI

* OpenAI API

### Deployment

* GitHub
* Streamlit Community Cloud

---

## ⚙️ Local Installation

Clone the repository:

```bash
git clone https://github.com/aitaliamelvin/real_estate_ml_platform.git
```

Move into the project directory:

```bash
cd real_estate_ml_platform
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🧪 Train the Model

From the project root:

```bash
python scripts/train_models.py
```

The trained pipeline will be saved to:

```text
models/real_estate_model.pkl
```

---

## 🔮 Test a Local Prediction

```bash
python scripts/predict_real_estate.py
```

---

## ⚡ Run the FastAPI API

```bash
uvicorn api.api_real_estate:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

FastAPI automatically provides an interactive Swagger interface where the `/predict` endpoint can be tested.

---

## 🖥️ Run the Streamlit Application

```bash
streamlit run dashboard/dashboard_real_estate.py
```

---

## 🔐 Environment Variables

The OpenAI feature requires an API key.

Create a local `.env` file:

```text
OPENAI_API_KEY=your_api_key
```

The `.env` file must never be committed to GitHub.

For Streamlit Community Cloud, the key should be configured through the application's **Secrets** settings.

---

## 📈 Possible Improvements

Future versions could include:

* real-world property datasets
* geographic coordinates
* interactive maps
* richer feature engineering
* model monitoring
* automated model retraining
* Docker deployment
* cloud-hosted FastAPI backend
* MLflow experiment tracking
* SHAP model explanations
* authentication and user accounts

---

## 🎯 What This Project Demonstrates

This project demonstrates practical skills in:

* Machine Learning
* feature preprocessing
* scikit-learn pipelines
* model comparison
* API development
* application development
* deployment
* AI API integration
* software project organization

---

## 👤 Author

**Melvin Ait-Alia**

Founder of **Optymia**
Digital solutions, automation, data & AI

GitHub:
https://github.com/aitaliamelvin

Live application:
https://real-estate-ml-platform.streamlit.app/

---

## ⚠️ Disclaimer

This project is a technical demonstration.

The dataset is synthetic and the generated property prices do not constitute professional real estate valuations or investment advice.
