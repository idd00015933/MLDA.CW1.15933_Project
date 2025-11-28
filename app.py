import streamlit as st
import numpy as np
import pickle
import os

# ============================
# SAFE MODEL LOADING
# ============================

def safe_load(path):
    if not os.path.exists(path):
        st.error(f"❌ Required file missing: {path}")
        st.stop()
    with open(path, "rb") as f:
        return pickle.load(f)

scaler = safe_load("scaler.pkl")
kmeans = safe_load("kmeans_model.pkl")

# ============================
# Streamlit Page Settings
# ============================
st.set_page_config(page_title="Travel Review Cluster App", layout="centered")

dark_purple = "#2b0a3d"
light_purple = "#b084f5"
white = "#ffffff"

# ============================
# CSS
# ============================
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {dark_purple};
        color: {white};
    }}
    label {{
        color: {white} !important;
        font-weight: bold;
    }}
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
    .result-card {{
        background-color: #3a0d55;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 0px 12px rgba(255,255,255,0.15);
        margin-top: 20px;
    }}
    h1, h2, h3 {{
        color: {light_purple};
        text-align: center;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ============================
# TITLE
# ============================
st.markdown("<h1>🌍 Travel Review Cluster Predictor</h1>", unsafe_allow_html=True)

# ============================
# RATING OPTIONS
# ============================
rating_map = {
    "Terrible (0)": 0,
    "Poor (1)": 1,
    "Average (2)": 2,
    "Very Good (3)": 3,
    "Excellent (4)": 4
}

categories = [
    "Category 1", "Category 2", "Category 3", "Category 4", "Category 5",
    "Category 6", "Category 7", "Category 8", "Category 9", "Category 10"
]

# ============================
# INPUT FORM
# ============================
st.markdown("<h3>Rate each category</h3>", unsafe_allow_html=True)

inputs = []

with st.form("rating_form"):
    for c in categories:
        choice = st.selectbox(
            c,
            options=list(rating_map.keys()),
            key=c
        )
        inputs.append(rating_map[choice])

    submit = st.form_submit_button("Predict Cluster")


# ============================
# FINAL SAFE PREDICTION
# ============================
if submit:
    user_data = np.array(inputs).reshape(1, -1)

    # --- FIXED SCALING ---
    # scaler might be sklearn or a numpy array → both supported
    if hasattr(scaler, "transform"):
        user_scaled = scaler.transform(user_data)
    else:
        # assume numpy mean/std arrays
        user_scaled = (user_data - scaler[0]) / scaler[1]

    # Predict
    cluster = int(kmeans.predict(user_scaled)[0])
    distances = kmeans.transform(user_scaled)[0]

    # Display result
    st.markdown(
        f"""
        <div class="result-card">
            <h2>🔮 Prediction Result</h2>
            <h3>Assigned Cluster: 
                <span style="color:{light_purple};">{cluster}</span>
            </h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 📏 Distance to Each Cluster Center")
    st.json({f"Cluster {i}": float(distances[i]) for i in range(len(distances))})
