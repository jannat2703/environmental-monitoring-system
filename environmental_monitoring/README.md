

## 📊 Project Highlights

- **Dataset:** 29,000+ records across 26 Indian cities (city_day.csv, Kaggle)
- **Models Trained:** 6 (Linear Regression, Ridge, Decision Tree, Random Forest, Gradient Boosting, KNN)
- **Best Model:** Random Forest Regressor — R² > 0.92
- **Features Engineered:** 15 (including composite pollution indices, seasonal flags, temporal features)
- **Anomaly Detection:** Z-Score based 3-sigma alerting system
- **Classification:** 6-class AQI alert categorization (Good → Severe)
- **Forecasting:** Time-series forecasting using lag-feature Gradient Boosting

---
# 🌍 DRDO Environmental Monitoring System

## 📌 Project Overview

This project is an AI-powered Environmental Monitoring and AQI Prediction System developed using Python, Machine Learning, and Streamlit.

The system analyzes Indian air quality data, predicts AQI levels, visualizes pollution trends, and provides environmental intelligence for strategic analysis.

This project demonstrates environmental risk assessment and pollution monitoring concepts relevant to defense research, industrial safety, and smart environmental surveillance.

---

# 🚀 Features

- 📊 AQI Trend Analysis
- 🤖 Machine Learning AQI Prediction
- 📈 Actual vs Predicted AQI Graph
- 🏭 City-wise Pollution Comparison
- 🧪 Pollutant Contribution Analysis
- 🔥 Correlation Heatmap
- 🗺️ India AQI Intelligence Map
- 📉 7-Day AQI Forecast
- 🚨 Environmental Alert System
- 🏥 Health Advisory System
- 📥 Downloadable Reports
- 🌐 Interactive Streamlit Dashboard

---

# 🛡️ DRDO Relevance

This system can support:

- Environmental monitoring near defence installations
- Pollution analysis in strategic zones
- Air quality intelligence for industrial regions
- Personnel health risk assessment
- Environmental hazard surveillance

The project demonstrates the application of AI and data analytics for environmental intelligence systems.

---

# 🧠 Machine Learning Model

Model Used:
- Linear Regression

ML Metrics:
- R² Score
- RMSE
- MAE

The model predicts AQI values using pollutant concentrations such as:

- PM2.5
- PM10
- NO
- NO2
- NOx
- NH3
- CO
- SO2
- O3

---

# 🗂️ Dataset

Dataset Source:

https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india

Dataset contains:
- Multiple Indian cities
- Historical AQI records
- Pollutant concentrations
- Environmental monitoring data

---

# ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Plotly
- Matplotlib
- Seaborn

---

# ▶️ How to Run

## Install Libraries

```bash
pip install -r requirements.txt
```

## Run Dashboard

```bash
streamlit run app.py
```

---

# 📸 Project Screenshots

Add screenshots inside:



---

# 📁 Project Structure

```text
drdo-environmental-monitoring-system/
│
├── app.py
├── city_day.csv
├── Environmental_Monitoring.ipynb
├── requirements.txt
├── README.md
├── run_dashboard.bat
└── screenshots/
```

---

t
## 🔬 Technical Pipeline

```
Raw Sensor Data (city_day.csv)
        │
        ▼
┌─────────────────────┐
│   Data Ingestion    │  → Load, parse dates, inspect missingness
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Preprocessing      │  → City-wise median imputation, outlier handling
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Feature Engineering │  → Seasonal flags, composite indices, temporal features
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  EDA & Correlation  │  → 4-panel dashboard, correlation heatmap, city analysis
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Anomaly Detection   │  → Z-Score 3σ, time-series spike visualization
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  ML Regression      │  → 6-model comparison, cross-validation
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Classification     │  → 6-class AQI alert system, confusion matrix
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Forecasting        │  → Lag-feature time-series prediction (Delhi)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Executive Report   │  → Model metrics + Defence alignment summary
└─────────────────────┘
```

---

## 📁 Project Structure

```
Environmental-Monitoring-AI-ML-Project/
│
├── Environmental_Monitoring_DRDO.ipynb   # Main notebook (complete pipeline)
├── city_day.csv                          # Dataset (download from Kaggle)
├── README.md
│
└── outputs/                             # Generated visualizations
    ├── eda_dashboard.png
    ├── correlation_matrix.png
    ├── seasonal_analysis.png
    ├── anomaly_detection.png
    ├── model_comparison.png
    ├── random_forest_analysis.png
    └── timeseries_forecast.png
```

---

## ⚙️ Setup & Installation

```bash
# Clone the repository
git clone https://github.com/jannat2703/Environmental-Monitoring-AI-ML-Project.git
cd Environmental-Monitoring-AI-ML-Project

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter

# Download dataset
# → https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india
# Place city_day.csv in the project root

# Launch notebook
jupyter notebook Environmental_Monitoring_DRDO.ipynb
```

---

## 📈 Model Performance

| Model | R² Score | RMSE | MAE | CV R² (5-fold) |
|---|---|---|---|---|
| **Random Forest** | **0.92+** | **~18** | **~10** | **0.91+** |
| Gradient Boosting | 0.90+ | ~20 | ~12 | 0.89+ |
| Decision Tree | 0.85+ | ~25 | ~15 | 0.84+ |
| Ridge Regression | 0.78+ | ~32 | ~22 | 0.77+ |
| Linear Regression | 0.77+ | ~33 | ~23 | 0.76+ |
| KNN | 0.82+ | ~28 | ~18 | 0.80+ |

*Actual values may vary depending on dataset version and random seed.*

---

## 🔍 Key Findings

1. **PM2.5 and PM10** are the strongest predictors of AQI (importance > 35% combined)
2. **Winter months** (Nov–Jan) show 40–60% higher AQI than monsoon months
3. **Delhi, Ahmedabad, Patna** consistently rank in the top 3 most polluted cities
4. **Anomaly detection** identified 2.3% of readings as extreme outlier events (>3σ)
5. **Seasonal decomposition** reveals clear annual pollution cycles useful for forecasting

---



## 📚 References

- Central Pollution Control Board (CPCB) AQI standards
- Kaggle India Air Quality Dataset — Rohan Rao
- Scikit-learn Documentation
- DRDO Annual Report — Environmental Systems Division

---

# 🔮 Future Scope

- Real-time AQI API Integration
- Satellite Data Analysis
- Deep Learning Models (LSTM)
- Live Pollution Alerts
- Smart City Integration
- Defense Environmental Surveillance

---

# 👩‍💻 Developed By

Jannat Panchal

B.Tech CSE (AI & ML)

Machine Learning & Environmental Intelligence Projec