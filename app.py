import numpy as np
import pandas as pd
import joblib
import streamlit as st


MODEL_PATH = "models/diabetes_risk_model_v1.pkl"
SCALER_PATH = "models/feature_scaler_v1.pkl"

FEATURE_ORDER = [
    "age",
    "blood_pressure",
    "fasting_glucose_level",
    "insulin_level",
    "cholesterol_level",
    "triglycerides_level",
    "physical_activity_level",
    "daily_calorie_intake",
    "sugar_intake_grams_per_day",
    "sleep_hours",
    "stress_level",
    "family_history_diabetes",
    "waist_circumference_cm",
    "gender_Male",
    "health_habits_score",
    "lipid_ratio",
    "high_lipid_risk",
]

FEATURES_TO_SCALE = [
    "age",
    "blood_pressure",
    "fasting_glucose_level",
    "insulin_level",
    "cholesterol_level",
    "triglycerides_level",
    "daily_calorie_intake",
    "sugar_intake_grams_per_day",
    "sleep_hours",
    "stress_level",
    "waist_circumference_cm",
    "health_habits_score",
    "lipid_ratio",
]

ACTIVITY_MAP = {"Low": 0, "Moderate": 1, "High": 2}
FAMILY_HISTORY_MAP = {"No": 0, "Yes": 1}
LABEL_MAP = {0: "Low Risk", 1: "Prediabetes", 2: "High Risk"}


@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def build_feature_row(raw: dict) -> dict:
    physical_activity = raw.get("physical_activity_level", "Moderate")
    if isinstance(physical_activity, str):
        physical_activity = ACTIVITY_MAP.get(physical_activity, 1)

    family_history = raw.get("family_history_diabetes", "No")
    if isinstance(family_history, str):
        family_history = FAMILY_HISTORY_MAP.get(family_history, 0)

    gender = raw.get("gender", "Male")
    gender_male = 1 if (gender == "Male" or gender is True) else 0

    trig = float(raw["triglycerides_level"])
    chol = float(raw["cholesterol_level"])
    high_lipid_risk = 1 if (trig > 150 and chol > 200) else 0
    lipid_ratio = raw.get("lipid_ratio")
    if lipid_ratio is None:
        lipid_ratio = trig / chol if chol else 0.0

    # Recreate health_habits_score as in 02_preprocessing_fe:
    # activity_norm = physical_activity_level / 2
    # sleep_norm = sleep_hours / 8
    # sugar_norm = sugar_intake_grams_per_day / 255
    activity_norm = physical_activity / 2.0
    sleep_norm = float(raw["sleep_hours"]) / 8.0
    sugar_norm = float(raw["sugar_intake_grams_per_day"]) / 255.0
    health_habits_score = activity_norm + sleep_norm - sugar_norm

    return {
        "age": float(raw["age"]),
        "blood_pressure": float(raw["blood_pressure"]),
        "fasting_glucose_level": float(raw["fasting_glucose_level"]),
        "insulin_level": float(raw["insulin_level"]),
        "cholesterol_level": chol,
        "triglycerides_level": trig,
        "physical_activity_level": physical_activity,
        "daily_calorie_intake": float(raw["daily_calorie_intake"]),
        "sugar_intake_grams_per_day": float(raw["sugar_intake_grams_per_day"]),
        "sleep_hours": float(raw["sleep_hours"]),
        "stress_level": float(raw["stress_level"]),
        "family_history_diabetes": family_history,
        "waist_circumference_cm": float(raw["waist_circumference_cm"]),
        "gender_Male": gender_male,
        "health_habits_score": float(health_habits_score),
        "lipid_ratio": float(lipid_ratio),
        "high_lipid_risk": int(high_lipid_risk),
    }


def predict_risk(raw_input: dict):
    model, scaler = load_assets()
    row = build_feature_row(raw_input)
    df = pd.DataFrame([row])[FEATURE_ORDER]
    df[FEATURES_TO_SCALE] = scaler.transform(df[FEATURES_TO_SCALE])
    probs = model.predict_proba(df)[0]
    pred = int(model.predict(df)[0])
    return LABEL_MAP.get(pred, str(pred)), probs


def main():
    st.set_page_config(page_title="Diabetes Risk Prediction", layout="centered")
    st.title("Diabetes Risk Prediction")
    st.markdown(
        "Provide patient information below to estimate **diabetes risk** "
        "using the trained Logistic Regression model."
    )

    with st.form("risk_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age (years)", min_value=18, max_value=90, value=50)
            gender = st.selectbox("Gender", ["Male", "Female"])
            blood_pressure = st.number_input(
                "Systolic blood pressure (mmHg)", min_value=80, max_value=220, value=130
            )
            fasting_glucose = st.number_input(
                "Fasting glucose level (mg/dL)", min_value=60, max_value=300, value=110
            )
            insulin_level = st.number_input(
                "Insulin level (µU/mL)", min_value=0.0, max_value=300.0, value=15.0
            )
            cholesterol = st.number_input(
                "Total cholesterol (mg/dL)", min_value=100, max_value=400, value=220
            )
            triglycerides = st.number_input(
                "Triglycerides (mg/dL)", min_value=50, max_value=400, value=180
            )

        with col2:
            physical_activity = st.selectbox(
                "Physical activity level", ["Low", "Moderate", "High"], index=1
            )
            daily_calories = st.number_input(
                "Daily calorie intake (kcal)", min_value=1000, max_value=5000, value=2500
            )
            sugar_intake = st.number_input(
                "Sugar intake (g/day)", min_value=0, max_value=400, value=80
            )
            sleep_hours = st.number_input(
                "Sleep hours per night", min_value=3.0, max_value=12.0, value=7.0
            )
            stress_level = st.slider("Stress level (0–10)", min_value=0, max_value=10, value=5)
            family_history = st.selectbox(
                "Family history of diabetes", ["No", "Yes"], index=0
            )
            waist_cm = st.number_input(
                "Waist circumference (cm)", min_value=50.0, max_value=200.0, value=95.0
            )

        submitted = st.form_submit_button("Predict risk")

    if submitted:
        raw_input = {
            "age": age,
            "gender": gender,
            "blood_pressure": blood_pressure,
            "fasting_glucose_level": fasting_glucose,
            "insulin_level": insulin_level,
            "cholesterol_level": cholesterol,
            "triglycerides_level": triglycerides,
            "physical_activity_level": physical_activity,
            "daily_calorie_intake": daily_calories,
            "sugar_intake_grams_per_day": sugar_intake,
            "sleep_hours": sleep_hours,
            "stress_level": stress_level,
            "family_history_diabetes": family_history,
            "waist_circumference_cm": waist_cm,
            "lipid_ratio": triglycerides / cholesterol if cholesterol else 0.0,
        }

        with st.spinner("Running prediction..."):
            label, probs = predict_risk(raw_input)

        st.subheader("Predicted risk")
        confidence = float(probs.max())
        st.markdown(f"**{label}**  \nConfidence: **{confidence*100:.1f}%**")

        st.subheader("Class probabilities")
        prob_dict = {LABEL_MAP[i]: float(p) for i, p in enumerate(probs)}
        st.json({k: round(v, 4) for k, v in prob_dict.items()})


if __name__ == "__main__":
    main()

