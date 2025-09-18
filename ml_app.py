import streamlit as st
import pandas as pd
import joblib
import os
from config import (
    MODEL_FILE, FEATURE_COLUMNS, GENDER_OPTIONS, EDUCATION_OPTIONS, 
    PET_OPTIONS, ACTIVITY_OPTIONS, LOCATION_OPTIONS, SEASON_OPTIONS, 
    ENVIRONMENTAL_OPTIONS, AGE_RANGE, AGE_DEFAULT, INCOME_RANGE, 
    INCOME_DEFAULT, INCOME_STEP, TRAVEL_FREQUENCY_RANGE, 
    TRAVEL_FREQUENCY_DEFAULT, VACATION_BUDGET_RANGE, 
    VACATION_BUDGET_DEFAULT, VACATION_BUDGET_STEP, PROXIMITY_RANGE, 
    PROXIMITY_DEFAULT, PREDICTION_LABELS, CLASS_NAMES
)

def load_model(model_file):
    if not os.path.exists(model_file):
        st.error("Model file not found. Please train the model first.")
        st.stop()
    else:
        return joblib.load(model_file)

def run_ml_app():
    st.title("🏖️ Vacation Destination Predictor")
    st.markdown("Find out whether you're more likely to prefer beaches or mountains for your next vacation!")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Personal Information")
        age = st.slider("Age", AGE_RANGE[0], AGE_RANGE[1], AGE_DEFAULT)
        gender = st.radio("Gender", GENDER_OPTIONS)
        income = st.number_input("Annual Income ($)", INCOME_RANGE[0], INCOME_RANGE[1], INCOME_DEFAULT, step=INCOME_STEP)
        education_level = st.selectbox("Education Level", EDUCATION_OPTIONS)
        pets = st.radio("Do you have pets?", PET_OPTIONS)

    with col2:
        st.subheader("Preferences")
        travel_frequency = st.slider("Travel Frequency (trips per year)", 
                                    TRAVEL_FREQUENCY_RANGE[0], TRAVEL_FREQUENCY_RANGE[1], TRAVEL_FREQUENCY_DEFAULT)
        preferred_activities = st.selectbox("Preferred Activities", ACTIVITY_OPTIONS)
        vacation_budget = st.number_input("Vacation Budget ($)", 
                                         VACATION_BUDGET_RANGE[0], VACATION_BUDGET_RANGE[1], 
                                         VACATION_BUDGET_DEFAULT, step=VACATION_BUDGET_STEP)
        location = st.selectbox("Location", LOCATION_OPTIONS)
        favorite_season = st.selectbox("Favorite Season", SEASON_OPTIONS)
        environmental_concerns = st.radio("Environmental Concerns", ENVIRONMENTAL_OPTIONS)

    st.subheader("Proximity Preferences")
    col3, col4 = st.columns(2)
    with col3:
        proximity_to_mountains = st.slider("Proximity to Mountains (miles)", 
                                          PROXIMITY_RANGE[0], PROXIMITY_RANGE[1], PROXIMITY_DEFAULT)
    with col4:
        proximity_to_beaches = st.slider("Proximity to Beaches (miles)", 
                                        PROXIMITY_RANGE[0], PROXIMITY_RANGE[1], PROXIMITY_DEFAULT)

      
    # Tombol untuk memicu prediksi
    predict_button = st.button("Predict My Destination")
    
    if predict_button:
        with st.expander("Review Your Selections"):
            result = {
                'Age': age,
                'Gender': gender,
                'Income': income,
                'Education_Level': education_level,
                'Travel_Frequency': travel_frequency,
                'Preferred_Activities': preferred_activities,
                'Vacation_Budget': vacation_budget,
                'Location': location,
                'Proximity_to_Mountains': proximity_to_mountains,
                'Proximity_to_Beaches': proximity_to_beaches,
                'Favorite_Season': favorite_season,
                'Pets': pets,
                'Environmental_Concerns': environmental_concerns,
            }
            st.json(result)

        # Konversi 'y'/'n' ke 1/0 untuk kolom numerik
        pets_num = 1 if pets == 'y' else 0
        env_concerns_num = 1 if environmental_concerns == 'y' else 0

        single_df = pd.DataFrame([{
            'Age': age,
            'Gender': gender,
            'Income': income,
            'Education_Level': education_level,
            'Travel_Frequency': travel_frequency,
            'Preferred_Activities': preferred_activities,
            'Vacation_Budget': vacation_budget,
            'Location': location,
            'Proximity_to_Mountains': proximity_to_mountains,
            'Proximity_to_Beaches': proximity_to_beaches,
            'Favorite_Season': favorite_season,
            'Pets': pets_num,
            'Environmental_Concerns': env_concerns_num,
        }], columns=FEATURE_COLUMNS)

        model = load_model(MODEL_FILE)
        prediction = model.predict(single_df)
        pred_proba = model.predict_proba(single_df)

        pred_probability_score = {
            CLASS_NAMES[0]: round(pred_proba[0][0] * 100, 2),
            CLASS_NAMES[1]: round(pred_proba[0][1] * 100, 2)
        }

        st.markdown("---")
        st.subheader("Prediction Result")

        # Menampilkan hasil prediksi
        prediction_label = PREDICTION_LABELS.get(prediction[0])
        st.success(prediction_label)
        
        if prediction[0] == 0:
            st.balloons()
        else:
            st.snow()

        st.markdown("**Prediction Confidence:**")

        col5, col6 = st.columns(2)
        with col5:
            st.markdown(f"**{CLASS_NAMES[0]}:** {pred_probability_score[CLASS_NAMES[0]]}%")
            st.progress(float(pred_probability_score[CLASS_NAMES[0]] / 100))
        
        with col6:
            st.markdown(f"**{CLASS_NAMES[1]}:** {pred_probability_score[CLASS_NAMES[1]]}%")
            st.progress(float(pred_probability_score[CLASS_NAMES[1]] / 100))

if __name__ == "__main__":
    run_ml_app()
