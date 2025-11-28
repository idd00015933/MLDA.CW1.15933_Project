import streamlit as st
import numpy as np
import pickle

# ============================
# Safe model loading
# ============================
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("dbscan_model.pkl", "rb") as f:
        dbscan_model = pickle.load(f)
    model_loaded = True
except Exception as e:
    model_loaded = False
    error_message = str(e)

# ============================
# Page UI Configuration
# ============================
st.set_page_config(page_title="Travel Review Cluster Predictor", layout="centered")

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
        font-family: 'Arial';
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
        background-color: #d7c3ff;
        color: black;
        transform: scale(1.05);
    }}
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
# Title
# ============================
st.markdown("<h1>🌍 Travel Review Cluster Predictor</h1>", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"Model files not found: {error_message}")
    st.stop()

# ============================
# Rating options
# ============================
rating_map = {
    "Terrible (0)": 0,
    "Poor (1)": 1,
    "Average (2)": 2,
    "Very Good (3)": 3,
    "Excellent (4)": 4
}

categories = [
    "1. Average user feedback on art galleries", "2. Average user feedback on dance clubs", "3. Average user feedback on juice bars", "4. Average user feedback on restaurants", "5. Average user feedback on museums",
    "6. Average user feedback on resorts", "7. Average user feedback on parks/picnic spots", "8. Average user feedback on beaches", "9. Average user feedback on theaters", "10. Average user feedback on religious institutions"
]

# ============================
# User Input Form
# ============================
st.markdown("<h3>Provide Your Ratings</h3>", unsafe_allow_html=True)

inputs = []

with st.form("rating_form"):
    for cat in categories:
        rating = st.selectbox(
            f"{cat}",
            list(rating_map.keys())
        )
        inputs.append(rating_map[rating])
    
    submit = st.form_submit_button("Predict Cluster")

# ============================
# Cluster Meaning Mapping
# ============================
cluster_description = {
    0: "Terrible destination to go",
    1: "Poor destination to go",
    2: "Average destination to go",
    3: "Very Good destination to go",
    4: "Excellent destination to go"
}

# ============================
# Prediction Logic
# ============================
if submit:
    user_data = np.array(inputs).reshape(1, -1)

    # Scale input
    user_scaled = scaler.transform(user_data)

    # Predict cluster
    cluster = int(dbscan_model.predict(user_scaled)[0])

    description = cluster_description.get(cluster, "Unknown cluster")

    # Distances from centroids
    distances = dbscan_model.transform(user_scaled)[0]
    dist_dict = {f"Cluster {i}": float(distances[i]) for i in range(len(distances))}

    # Display result
    st.markdown(
        f"""
        <div class="result-card">
            <h2>🔮 Prediction Output</h2>
            <h3> Predicted Cluster: <span style="color:{light_purple};">{cluster}</span> </h3>
            <h3>Meaning: <span style="color:{light_purple};">{description}</span> </h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 📏 Distance to Each Centroid")
    st.json(dist_dict)
