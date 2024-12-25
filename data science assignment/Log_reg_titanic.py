import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

# saved the model as 'titanic_model.pkl'
with open('titanic_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Titanic Survival Prediction App")

# Collecting user input for the features
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3], index=0)
sex = st.selectbox("Sex", ["Male", "Female"], index=0)
age = st.slider("Age", 0, 100, 29)
sibsp = st.slider("Number of Siblings/Spouses Aboard (SibSp)", 0, 8, 0)
parch = st.slider("Number of Parents/Children Aboard (Parch)", 0, 6, 0)
fare = st.slider("Fare Paid", 0.0, 512.0, 32.0)
embarked = st.selectbox("Port of Embarkation", ["Cherbourg", "Queenstown", "Southampton"], index=2)

# Convert the user input to a DataFrame
user_input = pd.DataFrame({'Pclass': [pclass],'Sex': [1 if sex == "Female" else 0],'Age': [age],'SibSp': [sibsp],'Parch': [parch],'Fare': [fare],
                           'Embarked_Q': [1 if embarked == "Queenstown" else 0],'Embarked_S': [1 if embarked == "Southampton" else 0]})

# Make the prediction using the model
prediction = model.predict(user_input)
prediction_proba = model.predict_proba(user_input)

# Display the prediction
if prediction[0] == 1:
    st.success(f"The passenger is likely to survive with a probability of {prediction_proba[0][1]:.2f}")
else:
    st.error(f"The passenger is unlikely to survive with a probability of {prediction_proba[0][0]:.2f}")