from flask import Flask,render_template,request
import joblib

# Create flask app
flask_app = Flask(__name__)

# Load the pickle model and processor
with open('model_rnd.joblib','rb') as file:
    model=joblib.load(file)
with open('pre.joblib','rb') as file:
    pre=joblib.load(file)

@flask_app.route("/")
def Home():
  return render_template("index2.html")

@flask_app.route("/predict",methods=["POST"])
def predict():
  x = [i for i in request.form.values()]
  xpre = pre.transform([x])
  preds = model.predict(xpre)
  if preds[0]==1:
    op = "Loan Status is Approved"
  else:
    op = "Loan status is Not Approved"
  return render_template("index2.html",prediction_text=op)

if __name__=="__main__":
  flask_app.run(debug=True)
