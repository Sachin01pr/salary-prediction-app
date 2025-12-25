import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression

st.title("💼 Salary Prediction App")

# Sample data
data = {
    "gender": ["Male","Female","Male","Female","Male","Female","Male","Female","Male","Female"],
    "education": ["Graduate","Post Graduate","Graduate","Graduate","Post Graduate",
                  "Graduate","Graduate","Post Graduate","Graduate","Post Graduate"],
    "experience": [1,2,3,2,5,1,4,3,6,2],
    "salary": [20000,25000,30000,28000,50000,22000,45000,32000,60000,27000]
}

df = pd.DataFrame(data)

# Encoding
le_gender = LabelEncoder()
le_education = LabelEncoder()

df["gender_encoded"] = le_gender.fit_transform(df["gender"])
df["education_encoded"] = le_education.fit_transform(df["education"])

X = df[["gender_encoded", "education_encoded", "experience"]]
y = df["salary"]

# Scaling
scaler = MinMaxScaler()
X["experience"] = scaler.fit_transform(X[["experience"]])

# Model
model = LinearRegression()
model.fit(X, y)

# User input
gender = st.selectbox("Gender", ["Male", "Female"])
education = st.selectbox("Education", ["Graduate", "Post Graduate"])
experience = st.slider("Experience (years)", 0, 10, 1)

g = le_gender.transform([gender])[0]
e = le_education.transform([education])[0]
exp_scaled = scaler.transform([[experience]])[0][0]

if st.button("Predict Salary"):
    prediction = model.predict([[g, e, exp_scaled]])
    st.success(f"Predicted Salary: ₹{int(prediction[0])}")
