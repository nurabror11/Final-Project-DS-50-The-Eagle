import streamlit as st
import pandas as pd
import joblib
import os

feature_columns = [
    'Age', 'Gender', 'Income', 'Education_Level', 'Travel_Frequency',
    'Preferred_Activities', 'Vacation_Budget', 'Location',
    'Proximity_to_Mountains', 'Proximity_to_Beaches',
    'Favorite_Season', 'Pets', 'Environmental_Concerns'
]

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
        age = st.slider("Age", 18, 100, 30)
        gender = st.radio("Gender", ['male', 'female', 'non'])
        income = st.number_input("Annual Income ($)", 0, 500000, 50000, step=1000)
        education_level = st.selectbox("Education Level", ['high school','bachelor', 'master', 'doctorate'])
        pets = st.radio("Do you have pets?", ['y', 'n'])

    with col2:
        st.subheader("Preferences")
        travel_frequency = st.slider("Travel Frequency (trips per year)", 0, 10, 2)
        preferred_activities = st.selectbox("Preferred Activities", ['skiing', 'swimming', 'hiking', 'sunbathing'])
        vacation_budget = st.number_input("Vacation Budget ($)", 0, 10000, 1000, step=100)
        location = st.selectbox("Location", ['urban', 'suburban', 'rural'])
        favorite_season = st.selectbox("Favorite Season", ['summer', 'fall', 'winter', 'spring'])
        environmental_concerns = st.radio("Environmental Concerns", ['y', 'n'])

    st.subheader("Proximity Preferences")
    col3, col4 = st.columns(2)
    with col3:
        proximity_to_mountains = st.slider("Proximity to Mountains (miles)", 0, 300, 50)
    with col4:
        proximity_to_beaches = st.slider("Proximity to Beaches (miles)", 0, 300, 50)

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
    }], columns=feature_columns)

    model = load_model("model_pipeline.pkl")
    prediction = model.predict(single_df)
    pred_proba = model.predict_proba(single_df)

    pred_probability_score = {
        'Beaches': round(pred_proba[0][0] * 100, 2),
        'Mountains': round(pred_proba[0][1] * 100, 2)
    }

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 0:
        st.success("🏖️ You're a Beach Person!")

        st.balloons()
    else:
        st.success("⛰️ You're a Mountain Person!")
        st.snow()

    st.markdown("**Prediction Confidence:**")

    col5, col6 = st.columns(2)
    with col5:
        st.markdown(f"**Beaches:** {pred_probability_score['Beaches']}%")
        st.progress(float(pred_probability_score['Beaches'] / 100))
    
    with col6:
        st.markdown(f"**Mountains:** {pred_probability_score['Mountains']}%")
        st.progress(float(pred_probability_score['Mountains'] / 100))

    
    

if __name__ == "__main__":
    run_ml_app()
