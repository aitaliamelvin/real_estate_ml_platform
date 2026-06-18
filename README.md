# 🏠 AI-Powered Real Estate ML Platform

End-to-end Machine Learning and AI application for real estate price prediction.

This project combines:

- Machine Learning
- FastAPI
- Streamlit Dashboard
- OpenAI GPT Analysis
- Data Visualization
- AI-powered explanations

---

# 🚀 Project Overview

This application predicts real estate prices using Machine Learning models trained on a realistic synthetic dataset.

The platform also includes:

✅ AI-generated property analysis  
✅ Interactive dashboard  
✅ API deployment  
✅ Real-time predictions  
✅ Data visualizations  

---

# 📊 Dataset

A realistic synthetic dataset was generated with more than 10,000 properties.

Features include:

- surface
- rooms
- floor
- balcony
- parking
- property age
- district

Realistic noise was intentionally added to simulate real-world market conditions.

---

# 🤖 Machine Learning Models

The following regression models were tested:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

## ✅ Best Model

Linear Regression achieved the best performance.

### Metrics

| Metric | Score |
|---|---|
| MAE | ~23,000€ |
| R² | ~0.985 |

---

# 🧠 AI Features

The application includes two AI layers:

## 1️⃣ Rule-Based AI Explanation

Explains the main factors influencing the price.

Example:

> “The estimated price is influenced by the large surface area, luxury district and parking availability.”

---

## 2️⃣ GPT-Powered Analysis

OpenAI GPT generates a professional real estate analysis directly inside the dashboard.

Example:

> “This property appears highly attractive for premium family housing due to its location and modern characteristics.”

---

# ⚙️ Technologies Used

| Category | Technologies |
|---|---|
| Data | Pandas, NumPy |
| ML | Scikit-learn |
| API | FastAPI |
| Dashboard | Streamlit |
| Visualization | Matplotlib |
| AI | OpenAI API |
| Deployment | Uvicorn |
| Serialization | Joblib |

---

# 📁 Project Structure

```text
real_estate_ml/
│
├── data/
├── models/
├── dashboard/
├── api/
├── scripts/
├── screenshots/
├── README.md
├── requirements.txt
```

---

# 🌐 API

FastAPI is used to expose the trained ML model.

## Run API

```bash
uvicorn api.api_real_estate:app --reload
```

## Swagger Documentation

```text
http://127.0.0.1:8000/docs
```

---

# 📊 Dashboard

Interactive Streamlit dashboard for real-time property analysis.

## Run Dashboard

```bash
streamlit run dashboard/dashboard_real_estate.py
```

---

# 📈 Features

✅ Machine Learning prediction  
✅ Multiple ML models comparison  
✅ Interactive dashboard  
✅ FastAPI backend  
✅ GPT-generated analysis  
✅ Data visualization  
✅ Real-time predictions  
✅ Professional project architecture  

---

# 🚀 Future Improvements

- Docker deployment
- Cloud deployment
- User authentication
- Database integration
- Batch predictions
- Advanced AI assistant
- Historical prediction tracking

---

# 💼 Key Learnings

This project helped develop skills in:

- Machine Learning pipelines
- Data preprocessing
- Model evaluation
- API development
- AI integration
- Dashboard development
- AI product design
- ML Engineer workflow

---

# 👨‍💻 Author

AI & Machine Learning portfolio project.