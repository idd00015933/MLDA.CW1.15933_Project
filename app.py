import streamlit as st
import numpy as np
import pickle

# ============================
# Load models
# ============================
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("kmeans.pkl", "rb") as f:
    kmeans = pickle.load(f)

# ============================
# Page Configuration
# ============================
st.set_page_config(page_title="Travel Review Cluster App", layout="centered")

# ============================
# Custom CSS (Dark Purple Theme)
# ============================
dark_purple = "#2b0a3d"
light_purple = "#b084f5"
white = "#ffffff"

st.markdown(
    f"""
    <style>

    /* Background */
    .stApp {{
        background-color: {dark_purple};
        color: {white};
        font-family: 'Arial';
    }}

    /* Input labels */
    label {{
        color: {white} !important;
        font-weight: bold;
    }}

    /* Prediction button */
    .stButton > button {{
        background-color: {light_purple};
        color: {white};
        border-radius: 10px;
        padding: 12px 22px;
        font-size: 18px;
        border: none;
        transition: 0.3s;
    }}

    .stButton > button:hover {{
        background-color: #d2b7ff;
        color: black;
        transform: scale(1.05);
    }}

    /* Card style */
    .result-card {{
        background-color: #3a0d55;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 0px 12px rgba(255, 255, 255, 0.15);
        margin-top: 20px;
    }}

    h1, h2, h3 {{
        color: {light_purple};
        text-align: center;
        font-weight: 700;
    }}

    </style>
    """,
    unsafe_allow_html=True
)

# ============================
# App Title
# ============================
st.markdown("<h1>🌍 Travel Review Cluster Predictor</h1>", unsafe_allow_html=True)
st.write("Provide your 10 category ratings and discover your cluster group!")

# ============================
# Input Form
# ============================
st.markdown("<h3>Enter Your Ratings (0 - 4)</h3>", unsafe_allow_html=True)

categories = [
    "1. Average user feedback on art galleries", "2. Average user feedback on dance clubs", "3. Average user feedback on juice bars", "4. Average user feedback on restaurants", "5. Average user feedback on museums",
    "6. Average user feedback on resorts", "7. Average user feedback on parks/picnic spots", "8. Average user feedback on beaches", "9. Average user feedback on theaters", "10. Average user feedback on religious institutions"
]

inputs = []

with st.form("user_input_form"):
    for cat in categories:
        value = st.number_input(
            f"{cat} Rating", min_value=0.0, max_value=4.0, step=0.1, format="%.1f"
        )
        inputs.append(value)

    submit = st.form_submit_button("Predict Cluster")

# ============================
# Prediction Logic
# ============================
if submit:
    user_data = np.array(inputs).reshape(1, -1)
    user_scaled = scaler.transform(user_data)

    cluster = kmeans.predict(user_scaled)[0]
    distances = kmeans.transform(user_scaled)[0]

    # Display result inside a fancy card
    st.markdown(
        f"""
        <div class="result-card">
            <h2>🔮 Prediction Result</h2>
            <h3>Assigned Cluster: <span style="color:{light_purple};">{cluster}</span></h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 📏 Distance to Each Centroid")
    st.json({f"Cluster {i}": float(distances[i]) for i in range(len(distances))})
