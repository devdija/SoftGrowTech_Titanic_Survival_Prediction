import streamlit as st
import pandas as pd
import joblib

# Load models 
models = {
   "Random Forest": joblib.load("models/rfc_model.pkl"),
    "KNN": joblib.load("models/knn_model.pkl"),
    "SVC": joblib.load("models/svc_model.pkl"),
    "Naive Bayes": joblib.load("models/gnb_model.pkl"),
    "Logistic Regression": joblib.load("models/lr_model.pkl"),
    "Decision Tree": joblib.load("models/dtc_model.pkl")
}

st.title("Titanic Survival Prediction 🚢")

# User selects model 
selected_model_name = st.selectbox("Choose Model", list(models.keys()))
selected_model = models[selected_model_name]

# User input form 
st.header("Passenger Information")

# Replace these fields with the actual features your model expects
pclass = st.selectbox("Pclass", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=30)
fare = st.number_input("Fare", min_value=0.0, value=32.0)
embarked = st.selectbox("Embarked", ["C", "Q", "S"])


# Collect input into a DataFrame
input_df = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [sex],
    "Age": [age],
    "Fare": [fare],
    "Embarked": [embarked]
})

# Predict button 
if st.button("Predict Survival"):
    prediction = selected_model.predict(input_df)
    proba = selected_model.predict_proba(input_df)
    
    if prediction[0] == 1:
        st.success(f"✅ Passenger is predicted to Survive with probability {proba[0][1]:.2f}")
    else:
        st.error(f"❌ Passenger is predicted Not to Survive with probability {proba[0][0]:.2f}")
        