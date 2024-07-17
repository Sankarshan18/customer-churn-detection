import streamlit as st
import pickle
from sklearn.preprocessing import LabelEncoder

def main():
    # Display title and image
    st.title("Churn Analysis using Python")
    st.image("static\churn1.png", use_column_width=True)

    # Input form for user data
    st.subheader("Enter Customer Information:")

    # Additional attributes
    gender = st.radio("Gender:", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen:", [0, 1])
    partner = st.radio("Partner:", ["Yes", "No"])
    dependents = st.radio("Dependents:", ["Yes", "No"])
    tenure = st.number_input("Tenure (months):", min_value=0)
    phone_service = st.radio("Phone Service:", ["Yes", "No"])
    multiple_lines = st.radio("Multiple Lines:", ["Yes", "No", "No phone service"])
    internet_service = st.radio("Internet Service:", ["DSL", "Fiber optic", "No"])
    online_security = st.radio("Online Security:", ["Yes", "No", "No internet service"])
    online_backup = st.radio("Online Backup:", ["Yes", "No", "No internet service"])
    device_protection = st.radio("Device Protection:", ["Yes", "No", "No internet service"])
    tech_support = st.radio("Tech Support:", ["Yes", "No", "No internet service"])
    streaming_tv = st.radio("Streaming TV:", ["Yes", "No", "No internet service"])
    streaming_movies = st.radio("Streaming Movies:", ["Yes", "No", "No internet service"])
    contract = st.radio("Contract:", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.radio("Paperless Billing:", ["Yes", "No"])
    payment_method = st.radio("Payment Method:", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges = st.number_input("Monthly Charges:", min_value=0.0)
    total_charges = st.number_input("Total Charges:", min_value=0.0)

    # Button to trigger prediction
    if st.button("Predict Churn"):
        # Perform prediction
        prediction = predict_churn(gender, senior_citizen, partner, dependents, tenure, phone_service, multiple_lines,
                                   internet_service, online_security, online_backup, device_protection,
                                   tech_support, streaming_tv, streaming_movies, contract, paperless_billing,
                                   payment_method, monthly_charges, total_charges)

        # Display prediction result
        if prediction == 1:
            st.error("Churn Prediction: Churn (Customer likely to leave)")
        else:
            st.success("Churn Prediction: No Churn (Customer likely to stay)")



import numpy as np
from sklearn.preprocessing import OneHotEncoder

def predict_churn(gender, senior_citizen, partner, dependents, tenure, phone_service, multiple_lines,
                  internet_service, online_security, online_backup, device_protection,
                  tech_support, streaming_tv, streaming_movies, contract, paperless_billing,
                  payment_method, monthly_charges, total_charges):
    # Load the trained model
    model = pickle.load(open("model.sav", "rb"))

    # Encode categorical features
    categorical_features = [[gender, senior_citizen, partner, dependents, phone_service, multiple_lines,
                             internet_service, online_security, online_backup, device_protection,
                             tech_support, streaming_tv, streaming_movies, contract, paperless_billing,
                             payment_method]]
    ohe = OneHotEncoder(categories="auto", drop="first")
    ohe.fit(categorical_features)
    feature_vector = ohe.transform([[gender, senior_citizen, partner, dependents, phone_service, multiple_lines,
                                      internet_service, online_security, online_backup, device_protection,
                                      tech_support, streaming_tv, streaming_movies, contract, paperless_billing,
                                      payment_method]]).toarray()

    # Add numerical features
    feature_vector = np.hstack((feature_vector, [[tenure, monthly_charges, total_charges]]))

    # Perform prediction
    prediction = model.predict(feature_vector)

    return prediction[0]  # Return the predicted churn label (1 or 0)




if __name__ == "__main__":
    main()
    
    


