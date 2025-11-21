# app.py (FINAL COMPLETE CODE)

import streamlit as st
import joblib
import pandas as pd
import time
from src.mappings import GERMAN_CREDIT_MAPPINGS, NUMERICAL_RANGES, PREDICTION_MAPPING, MODEL_FEATURE_ORDER 
import os
from pathlib import Path

# --- Configuration & Path ---
MODEL_PATH = "models/xgb_pipeline.joblib" 

st.set_page_config(
    page_title="German Credit Risk Predictor",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- 1. Load Model and Mappings ---
@st.cache_resource
def load_model(path):
    """Loads the joblib model file."""
    if not os.path.exists(path):
        st.error(f"Model file not found at: {path}")
        st.error(f"Please check your path. The app expects the model at: {os.path.abspath(path)}")
        st.stop()
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model(MODEL_PATH)
st.sidebar.success("✅ Model and Mappings Loaded Successfully.")

# Helper function to get human-readable options
def get_options(feature):
    return list(GERMAN_CREDIT_MAPPINGS.get(feature, {}).keys())


# ----------------------------------------------------------------------
# VVVV CRITICAL SECTION: Define ALL 21 Inputs Sequentially at the Top VVVV
# ----------------------------------------------------------------------

st.sidebar.header("Credit Application Details (21 Features)")

# --- NUMERICAL INPUTS (7) ---
st.sidebar.subheader("Numerical Features")

duration = st.sidebar.slider("1. Duration for Loan  (Months)", min_value=NUMERICAL_RANGES["Duration_months"]["min"], max_value=NUMERICAL_RANGES["Duration_months"]["max"], value=NUMERICAL_RANGES["Duration_months"]["default"])
credit_amount = st.sidebar.slider("2. Credit Amount Requested for Loan (DM)", min_value=NUMERICAL_RANGES["Credit_amount"]["min"], max_value=NUMERICAL_RANGES["Credit_amount"]["max"], value=NUMERICAL_RANGES["Credit_amount"]["default"])
age = st.sidebar.slider("3. Age (Years)", min_value=NUMERICAL_RANGES["Age"]["min"], max_value=NUMERICAL_RANGES["Age"]["max"], value=NUMERICAL_RANGES["Age"]["default"])
loan_wage_ratio = st.sidebar.slider("4. Installment Rate (% of Income need to pay debt)", min_value=NUMERICAL_RANGES["loan_wage_ratio"]["min"], max_value=NUMERICAL_RANGES["loan_wage_ratio"]["max"], value=NUMERICAL_RANGES["loan_wage_ratio"]["default"])
tenure = st.sidebar.slider("5. Present Residence Tenure (Years)", min_value=NUMERICAL_RANGES["Tenure"]["min"], max_value=NUMERICAL_RANGES["Tenure"]["max"], value=NUMERICAL_RANGES["Tenure"]["default"])
existing_credit = st.sidebar.slider("6. Number of Existing Credits", min_value=NUMERICAL_RANGES["Existing Credit"]["min"], max_value=NUMERICAL_RANGES["Existing Credit"]["max"], value=NUMERICAL_RANGES["Existing Credit"]["default"])
dependents = st.sidebar.slider("7. Number of Dependents", min_value=NUMERICAL_RANGES["Dependents"]["min"], max_value=NUMERICAL_RANGES["Dependents"]["max"], value=NUMERICAL_RANGES["Dependents"]["default"])

# --- CATEGORICAL INPUTS (14) ---
st.sidebar.subheader("Categorical Features")

account_status_hr = st.sidebar.selectbox("8. Checking Account Status", options=get_options("Account_status"))
credit_history_hr = st.sidebar.selectbox("9. Credit History", options=get_options("Credit_history"))
purpose_hr = st.sidebar.selectbox("10. Purpose of Credit", options=get_options("loan_Purpose"))
savings_status_hr = st.sidebar.selectbox("11. Savings Account Status", options=get_options("Savings/Bonds_AC"))
employment_hr = st.sidebar.selectbox("12. Employment Since", options=get_options("Present_Employment_since"))
sex_hr = st.sidebar.selectbox("13. Sex", options=get_options("Sex"))
marital_status_hr = st.sidebar.selectbox("14. Marital Status", options=get_options("Marital_status"))
co_debtors_hr = st.sidebar.selectbox("15. Co-debtors / Guarantors", options=get_options("co-debtors"))
assets_hr = st.sidebar.selectbox("16. Assets / Property (Owned)", options=get_options("Assets/Physical_property"))
other_loans_hr = st.sidebar.selectbox("17. Other Installment Plans", options=get_options("Other_loans"))
housing_hr = st.sidebar.selectbox("18. Housing Situation", options=get_options("Housing"))
job_status_hr = st.sidebar.selectbox("19. Job Status", options=get_options("Job_status"))
telephone_hr = st.sidebar.selectbox("20. Has a Telephone", options=get_options("Telephone"))
foreign_worker_hr = st.sidebar.selectbox("21. Is a Foreign Worker", options=get_options("Foreign_worker"))

# ----------------------------------------------------------------------
# ^^^^ END CRITICAL SECTION: All inputs are now defined ^^^^
# ----------------------------------------------------------------------


# --- Main Application Logic with UX Enhancements ---
st.title("🏦 German Credit Risk Assessment App")
st.markdown("---")

# 1. UX: Show Input Summary
st.subheader("Customer Financial Snapshot")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Loan Amount", f"{credit_amount:,.0f} DM", "Target: 4,000 DM")
col2.metric("Duration", f"{duration} Months", "Max 72")
col3.metric("Applicant Age", f"{age} Years", "Min 19")
col4.metric("Checking Account", account_status_hr)

st.markdown("---")

# 2. UX: Prediction Container
prediction_container = st.container(border=True)
with prediction_container:
    st.markdown("### 🚀 Ready for Prediction?")
    
    if st.button("PREDICT CREDIT RISK", type="primary", use_container_width=True):
        
        with st.spinner('Analyzing 21 features and running XGBoost Pipeline...'):
            time.sleep(1.5)

            # --- 2. THE CRUCIAL MAPPING STEP (14 CATEGORICAL) ---
            account_status_code = GERMAN_CREDIT_MAPPINGS["Account_status"][account_status_hr]
            credit_history_code = GERMAN_CREDIT_MAPPINGS["Credit_history"][credit_history_hr]
            purpose_code = GERMAN_CREDIT_MAPPINGS["loan_Purpose"][purpose_hr]
            savings_status_code = GERMAN_CREDIT_MAPPINGS["Savings/Bonds_AC"][savings_status_hr]
            employment_code = GERMAN_CREDIT_MAPPINGS["Present_Employment_since"][employment_hr]
            sex_code = GERMAN_CREDIT_MAPPINGS["Sex"][sex_hr]
            marital_status_code = GERMAN_CREDIT_MAPPINGS["Marital_status"][marital_status_hr]
            co_debtors_code = GERMAN_CREDIT_MAPPINGS["co-debtors"][co_debtors_hr]
            assets_code = GERMAN_CREDIT_MAPPINGS["Assets/Physical_property"][assets_hr]
            other_loans_code = GERMAN_CREDIT_MAPPINGS["Other_loans"][other_loans_hr]
            housing_code = GERMAN_CREDIT_MAPPINGS["Housing"][housing_hr]
            job_status_code = GERMAN_CREDIT_MAPPINGS["Job_status"][job_status_hr]
            telephone_code = GERMAN_CREDIT_MAPPINGS["Telephone"][telephone_hr]
            foreign_worker_code = GERMAN_CREDIT_MAPPINGS["Foreign_worker"][foreign_worker_hr]
            
            # Combine all 21 features into a dictionary
            input_data = {
                'Account_status': account_status_code,
                'Duration_months': duration,
                'Credit_history': credit_history_code,
                'loan_Purpose': purpose_code,
                'Credit_amount': credit_amount,
                'Savings/Bonds_AC': savings_status_code,
                'Present_Employment_since': employment_code,
                'loan_wage_ratio': loan_wage_ratio,
                'Sex': sex_code,
                'Marital_status': marital_status_code,
                'co-debtors': co_debtors_code,
                'Tenure': tenure,
                'Assets/Physical_property': assets_code,
                'Age': age,
                'Other_loans': other_loans_code,
                'Housing': housing_code,
                'Existing Credit': existing_credit,
                'Job_status': job_status_code,
                'Dependents': dependents,
                'Telephone': telephone_code,
                'Foreign_worker': foreign_worker_code
            }
            
            # Convert dictionary to DataFrame and enforce the correct column order
            input_df = pd.DataFrame([input_data])
            input_df = input_df[MODEL_FEATURE_ORDER] 

            # --- 3. Prediction ---
            try:
                prediction = model.predict(input_df)[0]
                result_label = PREDICTION_MAPPING[prediction]

                # --- 4. UX: Display Results with Style ---
                st.subheader("Prediction Outcome")
                if prediction == 0:
                    st.success(f"🎉 APPROVED: {result_label}", icon="✅")
                    st.balloons()
                else:
                    st.error(f"🚨 REJECTED: {result_label}", icon="❌")
                    
                # 5. UX: Use an Expander for Model Details
                with st.expander("Model Details: Input Data"):
                    st.dataframe(input_df.T, use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred during prediction.")
                st.error(f"Error Details: {e}")
                st.warning("Prediction failed. Check your model's categories/feature names and retraining status.")

# 6. UX: Footer/Documentation
st.sidebar.markdown("---")
st.sidebar.info("Application built for MLOps deployment. All features collected.")