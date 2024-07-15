import streamlit as st
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def web_app():
    rf = joblib.load('./rf4.pkl')

    class Subject:
        def __init__(self, age, temp, cci, sapsii):
            self.age = age
            self.temp = temp
            self.cci = cci
            self.sapsii = sapsii

        def make_predict(self):
            subject_data = {
                "age": [self.age],
                "temp": [self.temp],
                "cci": [self.cci],
                "sapsii": [self.sapsii]
            }

            # Create a DataFrame
            df_subject = pd.DataFrame(subject_data)

            # Make the prediction
            prediction = rf.predict_proba(df_subject)[:, 1]
            cutoff = 0.244376
            if prediction >= cutoff:
                adjusted_prediction = (prediction - cutoff) * (0.5 / (1 - cutoff)) + 0.5
                adjusted_prediction = np.clip(adjusted_prediction, 0, 1)
            else:
                adjusted_prediction = prediction * (0.5 / cutoff)
                adjusted_prediction = np.clip(adjusted_prediction, 0, 0.5)

            adjusted_prediction = np.round(adjusted_prediction * 100, 2)
            st.write(f"""
                <div class='all'>
                    <p style='text-align: center; font-size: 20px;'>
                        <b>Based on the information provided, the model predicts the risk of 28-day death is {adjusted_prediction} %</b>
                    </p>
                </div>
            """, unsafe_allow_html=True)

            explainer = shap.Explainer(rf)
            shap_values = explainer.shap_values(df_subject)
            shap.force_plot(explainer.expected_value[1], shap_values[1][0, :], df_subject.iloc[0, :], matplotlib=True)
            st.pyplot(plt.gcf())

    st.set_page_config(page_title='VAP 28-Day Mortality')
    st.markdown(f"""
                <div class='all'>
                    <h1 style='text-align: center;'>28-day Death Risk for VAP Patients in ICU</h1>
                    <p class='intro'></p>
                </div>
                """, unsafe_allow_html=True)
    age = st.number_input("Age (years)", min_value=18, max_value=99, value=50)
    temp = st.number_input("Temperature (max, ℃)", min_value=30.0, max_value=45.0, value=37.0)
    cci = st.number_input("Charlson comorbidity index (CCI)", min_value=0, max_value=37, value=0)
    sapsii = st.number_input("Simplified Acute Physiology Score (SAPS II)", min_value=0, max_value=163, value=0)

    if st.button(label="Submit"):
        user = Subject(age, temp, cci, sapsii)
        user.make_predict()


web_app()
