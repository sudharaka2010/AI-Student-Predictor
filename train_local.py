import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

df = pd.read_csv("students.csv")

X = df.drop("final_mark", axis=1)
y = df["final_mark"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, pred))

joblib.dump(model, "student_predictor.pkl")
print("✅ Saved NEW student_predictor.pkl for your PC")
