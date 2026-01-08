import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Career Predictor", page_icon="🚀", layout="wide")

st.markdown("""
<style>
.header {background: linear-gradient(90deg,#4facfe,#00f2fe); padding: 15px; border-radius: 10px; text-align:center;}
</style>
<div class="header"><h1>🚀 Career Prediction Based on Skills & Interests</h1></div>
""", unsafe_allow_html=True)

# -------------------------
# Load Model
# -------------------------
try:
    model = joblib.load("career_rf_model.pkl")
    le = joblib.load("career_label_encoder.pkl")
except:
    st.error("❌ Model files not found. Please upload career_rf_model.pkl & career_label_encoder.pkl to the same folder.")
    st.stop()

# -------------------------
# Sidebar: Input Features
# -------------------------
st.sidebar.header("Enter Your Scores & Interests")

def user_input():
    scores = {
        "Math_Score": st.sidebar.slider("Math Score", 0, 100, 75),
        "Science_Score": st.sidebar.slider("Science Score", 0, 100, 75),
        "Biology_Score": st.sidebar.slider("Biology Score", 0, 100, 75),
        "Computer_Score": st.sidebar.slider("Computer Score", 0, 100, 75),
        "English_Score": st.sidebar.slider("English Score", 0, 100, 75),
        "Interest_Technology": st.sidebar.slider("Interest Technology", 0, 10, 5),
        "Interest_Medicine": st.sidebar.slider("Interest Medicine", 0, 10, 5),
        "Interest_Business": st.sidebar.slider("Interest Business", 0, 10, 5),
        "Interest_Arts": st.sidebar.slider("Interest Arts", 0, 10, 5),
        "Interest_Research": st.sidebar.slider("Interest Research", 0, 10, 5),
        "Logical_Thinking": st.sidebar.slider("Logical Thinking", 0, 10, 5),
        "Creativity": st.sidebar.slider("Creativity", 0, 10, 5),
        "Communication_Skills": st.sidebar.slider("Communication Skills", 0, 10, 5),
        "Problem_Solving": st.sidebar.slider("Problem Solving", 0, 10, 5),
        "Risk_Taking": st.sidebar.slider("Risk Taking", 0, 10, 5),
        "Leadership": st.sidebar.slider("Leadership", 0, 10, 5),
        "Family_Preference": st.sidebar.slider("Family Preference", 0, 10, 5),
        "Financial_Stability_Required": st.sidebar.slider("Financial Stability Required", 0, 10, 5)
    }
    return pd.DataFrame([scores])

input_df = user_input()

# -------------------------
# Prediction
# -------------------------
if st.button("Predict Career"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    career_name = le.inverse_transform(prediction)[0]

    st.success(f"✅ Recommended Career: **{career_name}**")

    # Probability Bar Chart
    prob_df = pd.DataFrame(prediction_proba, columns=le.classes_).T.reset_index()
    prob_df.columns = ["Career", "Probability"]
    fig = px.bar(prob_df, x="Career", y="Probability", color="Career", text="Probability")
    st.plotly_chart(fig)

# -------------------------
# Feature Radar Chart
# -------------------------
st.header("Your Skill & Interest Radar")
radar_df = input_df.T
radar_df.columns = ["Score"]
fig2 = px.line_polar(radar_df, r=radar_df["Score"], theta=radar_df.index, line_close=True)
st.plotly_chart(fig2)
