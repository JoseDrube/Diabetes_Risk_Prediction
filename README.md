# 🩺 Diabetes Risk Prediction System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployment-red.svg)

## 📌 Project Overview
This project is an end-to-end Machine Learning solution designed to classify individuals into three risk categories: **Low Risk, Prediabetes, and High Risk**. 

Using a dataset of clinical and lifestyle markers, I developed a production-ready pipeline that handles severe class imbalance and provides real-time inference through a web interface.

## 🚀 Key Features
* **Multi-Class Classification:** Goes beyond binary "Yes/No" to provide more clinical nuance (3 classes).
* **Imbalance Handling:** Utilized **SMOTE** (Synthetic Minority Over-sampling Technique) to ensure the model accurately identifies minority classes (Prediabetes).
* **Production Pipeline:** Includes serialized models and scalers for consistent inference.
* **Interactive Web App:** A **Streamlit** dashboard for real-time risk assessment.

## 📊 Model Performance
After evaluating Logistic Regression, Random Forests, and XGBoost (with GridSearchCV), the **Logistic Regression** model was selected as the champion due to its superior linear interpretability and performance.

| Metric | Score |
| :--- | :--- |
| **Recall (Macro)** | **94.25%** |
| **Precision** | 94.69% |
| **F1-Score** | 94.37% |

*Note: In a medical context, we prioritized **Recall** to minimize the risk of missing potentially at-risk patients.*

## 📁 Repository Structure
```text
├── notebooks/
│   ├── 01_Preprocessing_FE.ipynb  # Data cleaning, SMOTE, and Scaling
│   ├── 02_Modeling.ipynb          # Model selection and Hyperparameter tuning
│   └── 03_Inference_Testing.ipynb # Post-serialization validation
├── models/
│   ├── diabetes_risk_model_v1.pkl # Trained Champion Model
│   └── feature_scaler_v1.pkl      # Standardizer for input data
├── app.py                         # Streamlit Web Application
└── requirements.txt               # Project dependencies


🛠️ Installation & Usage
To set up this project locally, follow these steps:

1. Clone the repository:
    git clone https://github.com/JoseDrube/Diabetes_Risk_Prediction.git
cd Diabetes_Risk_Prediction

2. Install dependencies: 
    pip install -r requirements.txt 

3. Run the Web Application:
    streamlit run app.py

💡 Key Insights
Top Predictors: Clinical markers such as BMI and Fasting Glucose were identified as the strongest drivers of risk within the model coefficients.

Operational Efficiency: The Logistic Regression model achieved near-instantaneous inference (<1ms), making it highly suitable for real-time edge deployment or mobile integration.

Author: Jose Drube

Status: Completed - March 2026
