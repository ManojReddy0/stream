import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('logistic_model_protocol4.pkl')

st.title("🚢 Titanic Survival Prediction App")
st.write("Enter passenger details to predict survival")

pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 30)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 600.0, 30.0)
embarked = st.selectbox("Port of Embarkation", ["S", "Q", "C"])

sex = 1 if sex == 'female' else 0
embarked_Q = 1 if embarked == 'Q' else 0
embarked_S = 1 if embarked == 'S' else 0

if st.button("Predict"):
    data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked_Q, embarked_S]])
    prediction = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]
    if prediction == 1:
        st.success(f"🎉 Likely to Survive! (Probability: {prob:.2f})")
    else:
        st.error(f"❌ Unlikely to Survive. (Probability: {prob:.2f})")
