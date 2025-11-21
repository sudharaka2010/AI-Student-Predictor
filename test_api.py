import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "attendance": 80,
    "assignment_avg": 70,
    "quiz_avg": 65,
    "mid_mark": 68,
    "lab_avg": 72,
    "past_gpa": 3.1
}

res = requests.post(url, json=data)
print(res.json())
