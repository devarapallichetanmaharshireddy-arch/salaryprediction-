
from flask import Flask, render_template, request
import numpy as np
import joblib

app=Flask(__name__)
model=joblib.load("salary_model.pkl")
le_gender, le_edu, le_job, df = joblib.load("encoders.pkl")

@app.route("/")
def home():
    education=list(le_edu.classes_)
    gender=list(le_gender.classes_)
    jobs=list(le_job.classes_)
    return render_template("index.html",education=education,gender=gender,jobs=jobs)

@app.route("/predict",methods=["POST"])
def predict():
    age=float(request.form["age"])
    exp=float(request.form["experience"])
    current=float(request.form["current_salary"])
    gender=request.form["gender"]
    education=request.form["education"]
    job=request.form["job"]

    if exp>=age or age-exp<=18:
        return render_template("result.html",error="Invalid age/experience combination.")

    data=np.array([[age,
                    le_gender.transform([gender])[0],
                    le_edu.transform([education])[0],
                    le_job.transform([job])[0],
                    exp]])
    pred=model.predict(data)[0]
    msg="Higher than your current salary" if pred>current else "Lower than your current salary"
    return render_template("result.html",prediction=f"₹{pred:,.2f}",message=msg)

if __name__=="__main__":
    app.run(debug=True)
