import streamlit as st
import joblib
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Load dumped models & preprocessors

model = joblib.load("final_depression_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")  # dicionário com encoders por coluna
num_cols = joblib.load("num_cols.pkl")             # lista de colunas numéricas
category_mapping = joblib.load("category_mapping.pkl")  # se precisar de mapping manual

# Streamlit app

st.set_page_config(page_title="Am I Depressed?", page_icon="☹️", layout="centered")

st.title("Depression Probability Calculator")
st.write("Fill the form to predict the odds of having the big sad.")

with st.sidebar:
    st.title("About the App")
    st.write(
        "This web app leverages a machine learning model trained on an open-source Kaggle dataset "
        "to estimate the probability of depression. ")
    st.write(
        "While the model achieves an accuracy of 87%, it is **not** a substitute for professional "
        "mental health support. Please consult a qualified professional for any medical concerns."
    )

    st.title("Hire Me")
    st.markdown(
        """
        [![LinkedIn](https://img.shields.io/badge/LinkedIn-aaron--albrecht-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/aaron-albrecht-32692b259/)
        """,
        unsafe_allow_html=True
    )
col1, col2 = st.columns([1, 1])

# Input form

with col1:

    st.subheader("Personal Information")

    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    profession = st.selectbox("Profession", ['Student', 'Working Professional'])
    age = st.slider("Age", 10, 60, 20)

    st.subheader("Mental Health Factors")

    suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts ?", ["Yes", "No"])
    family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])

with col2:

    st.subheader("Academic Information")

    degree = st.selectbox("Degree", ['B.Pharm', 'BSc', 'BA', 'BCA', 'M.Tech', 'MSc', 'MD', 'Class 12', 'Other'])
    cgpa = st.slider("CGPA", 0, 10, 5)
    academic_pressure = st.slider("Academic Pressure (1–5)", 1, 5, 3)

    st.subheader("Lifestyle Factors")

    sleep_duration = st.selectbox("Sleep Duration", ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours'])
    dietary_habits = st.selectbox("Dietary Habits", ['Healthy', 'Moderate', 'Unhealthy'])




# Organize data to a DataFrame

input_dict = {
    'Gender': gender,
    'Age': float(age),
    'Profession': profession,
    'Academic Pressure': float(academic_pressure),
    'Work Pressure': 0.0,
    'CGPA': float(cgpa),
    'Study Satisfaction': 3.0,
    'Job Satisfaction': 0.0,
    'Sleep Duration': sleep_duration,
    'Dietary Habits': dietary_habits,
    'Degree': degree,
    'Have you ever had suicidal thoughts ?': suicidal_thoughts,
    'Work/Study Hours': 5.0,
    'Financial Stress': 1.0,
    'Family History of Mental Illness': family_history
}

input_df = pd.DataFrame([input_dict])

# Pre processing

# Apply LabelEncoder to categorical cols

for col, le in label_encoders.items():
    if col in input_df.columns:
        input_df[col] = le.transform(input_df[col])

# Scaler to numerical cols

input_df[num_cols] = scaler.transform(input_df[num_cols])

# Prediction

prediction = model.predict(input_df)[0]
prob = model.predict_proba(input_df)[0][1]

st.write(f"**Predicted Class:** {'Depressed' if prediction == 1 else 'Not Depressed'}")

# Convert probability to % and set bar palette

prob_percent = int(prob * 100)

cmap = cm.get_cmap("turbo")  
color = cmap(prob) 
hex_color = mcolors.rgb2hex(color)  # converts hex to string

# Custom bar

st.markdown(f"""
<div style="border: 1px solid #ccc; border-radius: 10px; width: 100%; background-color: #f5f5f5; position: relative; height: 30px;">
  <div style="background-color: {hex_color}; width: {prob_percent}%; height: 100%; border-radius: 10px; text-align: center; color: black; font-weight: bold;">
    {prob_percent}%
  </div>
</div>
""", unsafe_allow_html=True)

