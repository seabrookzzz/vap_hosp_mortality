import streamlit as st
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def web_app():
    st.set_page_config(page_title='Hospital Mortality Risk in VAP Patients')
    # rf = joblib.load('./vapweb/rf.pkl')
    rf = joblib.load('./rf.pkl')

    class Subject:
        def __init__(self, Age, BMI, DBP, Temperature, UO, Platelets, Aniongap, Bicarbonate,
                     BUN, Sodium, Congestive_heart_failure, Cerebrovascular_disease):
            self.Age = Age
            self.BMI = BMI
            self.DBP = DBP
            self.Temperature = Temperature
            self.UO = UO
            self.Platelets = Platelets
            self.Aniongap = Aniongap
            self.Bicarbonate = Bicarbonate
            self.BUN = BUN
            self.Sodium = Sodium
            self.Congestive_heart_failure = Congestive_heart_failure
            self.Cerebrovascular_disease = Cerebrovascular_disease

        def make_predict(self):
            subject_data = {
                "Age": [self.Age],
                "BMI": [self.BMI],
                "DBP": [self.DBP],
                "Temperature": [self.Temperature],
                "UO": [self.UO],
                "Platelets": [self.Platelets],
                "Aniongap": [self.Aniongap],
                "Bicarbonate": [self.Bicarbonate],
                "BUN": [self.BUN],
                "Sodium": [self.Sodium],
                "Congestive_heart_failure": [self.Congestive_heart_failure],
                "Cerebrovascular_disease": [self.Cerebrovascular_disease],
            }

            df_subject = pd.DataFrame(subject_data)
            prediction = rf.predict_proba(df_subject)[:, 1]
            cutoff = 0.2595592
            if prediction >= cutoff:
                adjusted_prediction = (prediction - cutoff) * (0.5 / (1 - cutoff)) + 0.5
                adjusted_prediction = np.clip(adjusted_prediction, 0.5, 1)
            else:
                adjusted_prediction = prediction * (0.5 / cutoff)
                adjusted_prediction = np.clip(adjusted_prediction, 0, 0.5)

            adjusted_prediction = np.round(adjusted_prediction * 100, 2)
            st.write(f"""
                <div class='all'>
                    <p style='text-align: center; font-size: 20px;'>
                        <b>Based on the information provided,<br>the model predicts a {adjusted_prediction}% risk of hospital mortality.</b>
                    </p>
                </div>
            """, unsafe_allow_html=True)

            explainer = shap.Explainer(rf)
            shap_values = explainer.shap_values(df_subject)
            shap.force_plot(explainer.expected_value[1], shap_values[1][0, :], df_subject.iloc[0, :], matplotlib=True)
            st.pyplot(plt.gcf())

    st.markdown(f"""
                <div class='all'>
                    <h1 style='text-align: center;'>Web App - Hospital Mortality Risk<br>in VAP Patients</h1>
                    <p class='intro'></p>
                </div>
                """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        Age = st.number_input("Age (years)", min_value=18, max_value=100, value=75)
        BMI = st.number_input("Body Mass Index (BMI)", min_value=10, max_value=100, value=30)
        DBP = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=30, max_value=120, value=55)
        Temperature = st.number_input("Temperature (℃)", min_value=34.0, max_value=40.0, value=37.0, format="%.1f")
        UO = st.number_input("24-hour Urine Output (L)", 0.00, 10.00, 0.80, format="%.2f")
        Congestive_heart_failure = st.radio("Congestive Heart Failure", options=['No', 'Yes'], index=0)
        Congestive_heart_failure = 1 if Congestive_heart_failure == 'Yes' else 0

    with col2:
        Platelets = st.number_input("Platelets (K/μL)", 10, 1000, 80)
        Aniongap = st.number_input("Aniongap (mmol/L)", 1, 35, 20)
        Bicarbonate = st.number_input("Bicarbonate (mmol/L)", 9, 45, 16)
        BUN = st.number_input("Blood Urea Nitrogen (mg/dL)", 1, 160, 40)
        Sodium = st.number_input("Sodium (mmol/L)", 120, 160, 145)
        Cerebrovascular_disease = st.radio("Cerebrovascular Disease", options=['No', 'Yes'], index=0)
        Cerebrovascular_disease = 1 if Cerebrovascular_disease == 'Yes' else 0

    if st.button(label="Submit"):
        user = Subject(Age, BMI, DBP, Temperature, UO, Platelets, Aniongap, Bicarbonate,
                       BUN, Sodium, Congestive_heart_failure, Cerebrovascular_disease)
        user.make_predict()


web_app()
