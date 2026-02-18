import streamlit as st
import pandas as pd
import joblib

st.set_page_config("Deployment Project")
st.title("Loan Status Prediction")
st.subheader("By Sindhura Kuntamukkula")

pre = joblib.load('pre.joblib')
model = joblib.load('model_random.joblib')

age = st.number_input("Age",min_value=18,step=1)
income = st.number_input("Income",min_value=4000)
home_ownership = st.selectbox("Home ownership",options=['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
emp_length = st.number_input("Employee length")
intent = st.selectbox("Loan intent",options=['EDUCATION','MEDICAL','PERSONAL','VENTURE', 'DEBTCONSOLIDATION','HOMEIMPROVEMENT'])
grade = st.selectbox("Loan grade",options=['B', 'C', 'A', 'D', 'E', 'F', 'G'])
loan_amnt=st.number_input("Loan Amount")
int_rate = st.number_input("Interest Rate")
percent_income = st.number_input("Percent Income")
cb_file = st.selectbox("Credit Person default on file",options=['N', 'Y'])
cb_history = st.number_input("Credit History Length")

submit = st.button("Predict Loan Status")

if submit:
    data = {
        'person_age':[age],
        'person_income':[income],
        'person_home_ownership':[home_ownership],
        'person_emp_length':[emp_length],
        'loan_intent':[intent],
        'loan_grade':[grade],
        'loan_amnt':[loan_amnt],
        'loan_int_rate':[int_rate],
        'loan_percent_income':[percent_income],
        'cb_person_default_on_file':[cb_file],
        'cb_person_cred_hist_length':[cb_history]
    }
    xnew = pd.DataFrame(data)
    xnew_pre = pre.transform(xnew)
    preds = model.predict(xnew_pre)
    if preds[0]==1:
        st.subheader("Loan Status is Approved")
    else:
        st.subheader("Loan Status is not Approved")

