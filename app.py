from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("student_predictor.pkl")

@app.route("/")
def home():
    return "Student Predictor API is running ✅. Use POST /predict"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    features = np.array([[
        data["attendance"],
        data["assignment_avg"],
        data["quiz_avg"],
        data["mid_mark"],
        data["lab_avg"],
        data["past_gpa"]
    ]])

    prediction = model.predict(features)[0]

    return jsonify({
        "predicted_final_mark": round(float(prediction), 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
