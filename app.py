import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor

# --- DYNAMIC FILE LOCATIONS ---
# This looks in the SAME FOLDER where your app.py is saved
current_folder = os.path.dirname(os.path.abspath(__file__))
model_file = os.path.join(current_folder, "model.pkl")
csv_file = os.path.join(current_folder, "all_mobile_details.csv")

st.title("📱 Smartphone Price Predictor")

# --- TRAINING LOGIC ---
@st.cache_resource # This saves the model so it doesn't retrain every time
def get_model():
    if os.path.exists(model_file):
        with open(model_file, 'rb') as f:
            return pickle.load(f)
    elif os.path.exists(csv_file):
        st.info("Training model from CSV... please wait.")
        df = pd.read_csv(csv_file)
        # Cleaning
        df["RAM"] = df["Description"].str.extract(r'(\d+)\s*GB RAM').astype(float)
        df["ROM"] = df["Description"].str.extract(r'(\d+)\s*GB ROM').astype(float)
        df["Battery"] = df["Description"].str.extract(r'(\d+)\s*mAh').astype(float)
        df["Price"] = df["Price"].str.replace("₹", "").str.replace(",", "").astype(float)
        df.dropna(subset=["RAM", "ROM", "Battery", "Price"], inplace=True)
        # Train
        X = df[["RAM", "ROM", "Battery"]]
        y = df["Price"]
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        # Save for next time
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        return model
    else:
        return None

model = get_model()

# --- USER INTERFACE ---
if model is None:
    st.error(f"❌ ERROR: File not found!")
    st.write(f"Please put **all_mobile_details.csv** in this folder: `{current_folder}`")
else:
    ram = st.slider("RAM (GB)", 2, 16, 8)
    rom = st.slider("Storage (GB)", 32, 512, 128)
    battery = st.number_input("Battery (mAh)", 2000, 7000, 5000)

    if st.button("Predict Price"):
        prediction = model.predict([[ram, rom, battery]])
        st.balloons()
        st.success(f"### Estimated Price: ₹{prediction[0]:,.2f}")