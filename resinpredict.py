import streamlit as st
import pandas as pd
import pickle

# -------------------------------
# Load the trained model
# -------------------------------
with open("raisinmodel.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Raisin Classifier", page_icon="🍇", layout="centered")

st.title("🍇 Raisin Type Classifier")
st.write(
    "This app predicts whether a raisin belongs to the **Kecimen** or **Besni** "
    "class based on its physical measurements."
)

st.divider()
st.subheader("Enter Raisin Measurements")

# -------------------------------
# Input fields (match your training feature columns)
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area", min_value=0.0, value=50000.0, step=100.0)
    major_axis_length = st.number_input("Major Axis Length", min_value=0.0, value=300.0, step=1.0)
    minor_axis_length = st.number_input("Minor Axis Length", min_value=0.0, value=180.0, step=1.0)
    eccentricity = st.number_input("Eccentricity", min_value=0.0, max_value=1.0, value=0.75, step=0.01)

with col2:
    convex_area = st.number_input("Convex Area", min_value=0.0, value=51000.0, step=100.0)
    extent = st.number_input("Extent", min_value=0.0, max_value=1.0, value=0.65, step=0.01)
    perimeter = st.number_input("Perimeter", min_value=0.0, value=1000.0, step=1.0)

# -------------------------------
# Prepare input for prediction
# -------------------------------
input_df = pd.DataFrame(
    [[area, major_axis_length, minor_axis_length, eccentricity, convex_area, extent, perimeter]],
    columns=[
        "Area",
        "MajorAxisLength",
        "MinorAxisLength",
        "Eccentricity",
        "ConvexArea",
        "Extent",
        "Perimeter",
    ],
)

st.divider()

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Raisin Class"):
    prediction = model.predict(input_df)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0]

    label = "Kecimen" if prediction == 1 else "Besni"
    st.success(f"### Predicted Class: **{label}**")

    if proba is not None:
        st.write("Prediction probabilities:")
        st.write(
            pd.DataFrame(
                {"Class": ["Besni", "Kecimen"], "Probability": proba}
            ).set_index("Class")
        )

st.divider()
st.caption("Model: RandomForestClassifier trained on the Raisin Dataset.")
