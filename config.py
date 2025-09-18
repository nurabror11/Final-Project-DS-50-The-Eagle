# Konfigurasi untuk aplikasi Vacation Destination Predictor

# Nama file model
MODEL_FILE = "model_pipeline.pkl"

# Daftar kolom fitur
FEATURE_COLUMNS = [
    'Age', 'Gender', 'Income', 'Education_Level', 'Travel_Frequency',
    'Preferred_Activities', 'Vacation_Budget', 'Location',
    'Proximity_to_Mountains', 'Proximity_to_Beaches',
    'Favorite_Season', 'Pets', 'Environmental_Concerns'
]

# Opsi untuk input pengguna
GENDER_OPTIONS = ['male', 'female', 'non']
EDUCATION_OPTIONS = ['high school', 'bachelor', 'master', 'doctorate']
PET_OPTIONS = ['y', 'n']
ACTIVITY_OPTIONS = ['skiing', 'swimming', 'hiking', 'sunbathing']
LOCATION_OPTIONS = ['urban', 'suburban', 'rural']
SEASON_OPTIONS = ['summer', 'fall', 'winter', 'spring']
ENVIRONMENTAL_OPTIONS = ['y', 'n']

# Rentang nilai untuk slider dan input numerik
AGE_RANGE = (18, 100)
AGE_DEFAULT = 30
INCOME_RANGE = (0, 500000)
INCOME_DEFAULT = 50000
INCOME_STEP = 1000
TRAVEL_FREQUENCY_RANGE = (0, 10)
TRAVEL_FREQUENCY_DEFAULT = 2
VACATION_BUDGET_RANGE = (0, 10000)
VACATION_BUDGET_DEFAULT = 1000
VACATION_BUDGET_STEP = 100
PROXIMITY_RANGE = (0, 300)
PROXIMITY_DEFAULT = 50

# Label untuk hasil prediksi
PREDICTION_LABELS = {
    0: "🏖️ You're a Beach Person!",
    1: "⛰️ You're a Mountain Person!"
}

# Nama kelas untuk probabilitas
CLASS_NAMES = ['Beaches', 'Mountains']