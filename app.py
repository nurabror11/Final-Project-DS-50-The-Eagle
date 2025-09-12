import streamlit as st
from PIL import Image

from ml_app  import run_ml_app

def main():
   
   menu = ['Home', 'Machine Learning']
   choice = st.sidebar.selectbox('Menu', menu)
   
   if choice == 'Home':
        st.subheader("Welcome to Homepage")
        st.title("Vacation Preference Predictor")
        col1, col2 = st.columns(2)
        with col1:
           st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/73/Beach_at_Fort_Lauderdale.jpg/330px-Beach_at_Fort_Lauderdale.jpg", 
                caption="Beach Vacation", use_column_width=True)
           with col2:
               st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Everest_North_Face_toward_Base_Camp_Tibet_Luca_Galuzzi_2006.jpg/330px-Everest_North_Face_toward_Base_Camp_Tibet_Luca_Galuzzi_2006.jpg", 
                caption="Mountain Vacation", use_column_width=True)
   
        st.markdown("This app helps you discover your travel preference between beaches and mountains. Simply answer a few quick questions about your interests, comfort, and lifestyle, and get a personalized result showing whether you’d enjoy a beach getaway or a mountain retreat more.")
     
   elif choice == 'Machine Learning':
       run_ml_app()
if __name__ == '__main__' :
    main()