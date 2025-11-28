import streamlit as st
import numpy as np
import pickle
import os

# ===============================================
# Safe model loader
# ===============================================
def load_file(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

scaler = load_file("scaler.pkl")
kmeans = load_file("kmeans_model.pkl")

# ===============================================
# Page UI Setup
# ===============================================
st.set_page_config(page_title="Travel Review Clustering", layout="centered")

dark_purple = "#2b0a3d"
light_purple = "#b084f5"
white = "#ffffff"

# Custom CSS
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
        padding: 10px 20px;
        font-size: 17px;
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
        margin-top: 20px;
        box-shadow: 0px 0px 12px rgba(255,255,255,0.2);
    }}

    h1, h2, h3 {{
        color: {light_purple};
        text-align: center;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ===============================================
# App Title
# ===============================================
st.markdown("<h1>🌍 Travel Review Cluster Prediction</h1>", unsafe_allow_html=True)

# ===============================================
# Check if models exist
# ===============================================
if scaler is None or kmeans is None:
    st.error("❌ scaler.pkl or kmeans_model.pkl not found.\n\n"
             "Please upload or include the trained model files.")
    st.stop()

# ===============================================
# Rating Mapping
# ===============================================
rating_map = {
    "Excellent (4)": 4.0,
    "Very Good (3)": 3.0,
    "Average (2)": 2.0,
    "Poor (1)": 1.0,
    "Terrible (0)": 0.0
}

categories = [
    "1. Average user feedback on art galleries", "2. Average user feedback on dance clubs", "3. Average user feedback on juice bars", "4. Average user feedback on restaurants", "5. Average user feedback on museums",
    "6. Average user feedback on resorts", "7. Average user feedback on parks/picnic spots", "8. Average user feedback on beaches", "9. Average user feedback on theaters", "10. Average user feedback on religious institutions"
]

# ===============================================
# Collect User Input
# ===============================================
with st.form("rating_form"):
    st.markdown("<h3>Select Ratings</h3>", unsafe_allow_html=True)

    inputs = []
    for cat in categories:
        choice = st.selectbox(cat, list(rating_map.keys()))
        inputs.append(rating_map[choice])

    submitted = st.form_submit_button("Predict Cluster")

# ===============================================
# Prediction Logic
# ===============================================
if submitted:
    user_data = np.array(inputs).reshape(1, -1)

    # Scale input properly
    user_scaled = scaler.transform(user_data)

    # Predict cluster
    cluster = int(kmeans.predict(user_scaled)[0])

    distances = kmeans.transform(user_scaled)[0]

    # Output card
    st.markdown(
        f"""
        <div class="result-card">
            <h2>🔮 Predicted Cluster: {cluster}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("### 📏 Distance to each centroid")
    st.json({f"Cluster {i}": float(distances[i]) for i in range(len(distances))})
