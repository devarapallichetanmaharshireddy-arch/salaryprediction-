
from flask import Flask, render_template, request
import numpy as np
import joblib
import math

app = Flask(__name__)

# Load trained model and encoders
model = joblib.load("salary_model.pkl")
le_gender, le_edu, le_job, df = joblib.load("encoders.pkl")


@app.route("/")
def home():
    education = list(le_edu.classes_)
    gender = list(le_gender.classes_)
    jobs = list(le_job.classes_)

    return render_template(
        "index.html",
        education=education,
        gender=gender,
        jobs=jobs
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read input values
        age_input = request.form.get("age", "").strip()
        exp_input = request.form.get("experience", "").strip()
        salary_input = request.form.get("current_salary", "").strip()

        if not age_input or not exp_input or not salary_input:
            raise ValueError("All numeric fields are required.")

        age_value = float(age_input)
        exp = float(exp_input)
        current = float(salary_input)

        # Reject NaN and infinity
        if not all(map(math.isfinite, [age_value, exp, current])):
            raise ValueError("Enter valid numeric values.")

        # Age must be a whole number
        if not age_value.is_integer():
            raise ValueError("Age must be a whole number.")

        age = int(age_value)

    except ValueError as e:
        return render_template("result.html", error=str(e))

    # Read categorical inputs
    gender = request.form.get("gender", "")
    education = request.form.get("education", "")
    job = request.form.get("job", "")

    # Validate age
    if age < 18 or age > 100:
        return render_template(
            "result.html",
            error="Age must be between 18 and 100."
        )

    # Validate experience
    if exp < 0:
        return render_template(
            "result.html",
            error="Experience cannot be negative."
        )

    if exp > age - 18:
        return render_template(
            "result.html",
            error="Experience cannot exceed the years since age 18."
        )

    # Validate current salary
    if current < 0:
        return render_template(
            "result.html",
            error="Current salary cannot be negative."
        )

    # Validate dropdown values against the encoders
    if gender not in le_gender.classes_:
        return render_template(
            "result.html",
            error="Please select a valid gender."
        )

    if education not in le_edu.classes_:
        return render_template(
            "result.html",
            error="Please select a valid education level."
        )

    if job not in le_job.classes_:
        return render_template(
            "result.html",
            error="Please select a valid job role."
        )

    # Prepare input in the same order used during model training
    data = np.array([[
        age,
        le_gender.transform([gender])[0],
        le_edu.transform([education])[0],
        le_job.transform([job])[0],
        exp
    ]])

    # Predict salary
    try:
        pred = float(model.predict(data)[0])

        if not math.isfinite(pred):
            raise ValueError("The model returned an invalid prediction.")

    except Exception:
        app.logger.exception("Salary prediction failed")
        return render_template(
            "result.html",
            error="Prediction failed. Please check your model files."
        )

    # Compare predicted salary with current salary
    if pred > current:
        msg = "Higher than your current salary"
    elif pred < current:
        msg = "Lower than your current salary"
    else:
        msg = "Equal to your current salary"

    return render_template(
        "result.html",
        prediction=f"₹{pred:,.2f}",
        message=msg
    )


if __name__ == "__main__":
    app.run(debug=True)
