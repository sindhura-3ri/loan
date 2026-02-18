from flask import Flask,render_template,request
import pandas as pd
import joblib

# Create flask app
flask_app = Flask(__name__)

# Load the pickle model and processor
with open('model_random.joblib','rb') as file:
    model=joblib.load(file)
with open('pre.joblib','rb') as file:
    pre=joblib.load(file)

@flask_app.route("/")
def Home():
  return render_template("index2.html")

@flask_app.route("/predict", methods=["POST"])
def predict():

    # Numerical features (convert to float)
    Age = float(request.form["Age"])
    Income = float(request.form["Income"])
    Employee_Length = float(request.form["Employee_Length"])
    Loan_Amount = float(request.form["Loan_Amount"])
    Interest_Rate = float(request.form["Interest_Rate"])
    Percent_Income = float(request.form["Percent_Income"])
    Credit_History_Length = float(request.form["Credit_History_Length"])

    # Categorical features (keep as string)
    Home_Ownership = request.form["Home_Ownership"]
    Loan_Intent = request.form["Loan_Intent"]
    Loan_Grade = request.form["Loan_Grade"]
    Credit_Default = request.form["Credit_Default"]


    data = {
        "person_age": Age,
        "person_income": Income,
        "person_home_ownership": Home_Ownership,
        "person_emp_length": Employee_Length,
        "loan_intent": Loan_Intent,
        "loan_grade": Loan_Grade,
        "loan_amnt": Loan_Amount,
        "loan_int_rate": Interest_Rate,
        "loan_percent_income": Percent_Income,
        "cb_person_default_on_file": Credit_Default,
        "cb_person_cred_hist_length": Credit_History_Length
    }

    df = pd.DataFrame([data])
    xpre = pre.transform(df)
    preds = model.predict(xpre)

    if preds[0] == 1:
        op = "Loan Status is Approved"
    else:
        op = "Loan Status is Not Approved"

    return render_template("index2.html", prediction_text=op)

if __name__=="__main__":
  flask_app.run(debug=True)
