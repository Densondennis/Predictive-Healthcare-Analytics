from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("models/xgboost_model.pkl")

# Function to provide recommendations based on prediction
def get_recommendation(prediction):
    if prediction == 1:  # At High Risk
        return {
            "status": "At High Risk",
            "recommendations": [
                "Consult a doctor immediately for further testing.",
                "Follow a heart-healthy diet (low salt, more vegetables, whole grains).",
                "Engage in light physical activity with medical guidance.",
                "Quit smoking and avoid alcohol.",
                "Monitor blood pressure, cholesterol, and glucose levels regularly."
            ],
            "avoid": [
                "Avoid high-fat, high-sodium foods.",
                "Avoid strenuous exercise without a doctor's approval.",
                "Do not ignore symptoms like chest pain or breathlessness."
            ]
        }
    else:  # Not At Risk
        return {
            "status": "Not At Risk",
            "recommendations": [
                "Maintain a balanced diet with fiber and omega-3.",
                "Exercise regularly (30 min/day).",
                "Manage stress through relaxation techniques.",
                "Monitor health regularly (blood pressure, cholesterol)."
            ],
            "avoid": [
                "Avoid smoking and excessive alcohol.",
                "Do not ignore regular health screenings."
            ]
        }

# Home route to render homepage
@app.route("/")
def home():
    return render_template("home.html")

# Route for the prediction page
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            # Get data from HTML form
            age = float(request.form["Age"])
            sex = int(request.form["Sex"])
            cp = int(request.form["ChestPainType"])
            trestbps = float(request.form["RestingBP"])
            chol = float(request.form["Cholesterol"])
            fbs = int(request.form["FBS"])
            thalach = float(request.form["MaxHR"])
            exang = int(request.form["ExAng"])
            oldpeak = float(request.form["OldPeak"])
            slope = int(request.form["Slope"])
            ca = int(request.form["CA"])
            thal = int(request.form["Thal"])

            # Create DataFrame for model input
            input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, thalach, exang, oldpeak, slope, ca, thal]],
                                      columns=["age", "sex", "cp", "trestbps", "chol", "fbs", "thalach", "exang", "oldpeak", "slope", "ca", "thal"])

            # Get prediction probability
            proba = model.predict_proba(input_data)[:, 1]  # Probability of being high risk
            threshold = 0.6  # Define threshold
            prediction = 1 if proba[0] >= threshold else 0
            result = "At High Risk" if prediction == 1 else "Not At Risk"

            # Get recommendations based on prediction
            recommendations = get_recommendation(prediction)

            return render_template("predict.html", prediction=result, proba=proba[0], recommendations=recommendations)
        
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    
    return render_template("predict.html", prediction=None, recommendations=None)

if __name__ == "__main__":
    app.run(debug=True)
