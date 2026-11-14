"""DPS970 Activity 3, Q5-Q8: ice cream sales regression in Python (scikit-learn)."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use("Agg")  # save charts to a file instead of opening a window
import matplotlib.pyplot as plt

# Load the data and remove Date (DayOfWeek and Month already cover it)
data = pd.read_csv("ice-cream.csv").drop(columns=["Date"])
# X = inputs; text columns become 0/1 columns so the model can use them
X = pd.get_dummies(data.drop(columns=["IceCreamsSold"]), drop_first=True)
# y = target (what we predict)
y = data["IceCreamsSold"]
# Hold back 20% of rows as validation data the model doesn't train on
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model, then predict the validation rows
model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_val)

# Scores: R2 = how much of the variation is explained, MAE/RMSE = average error
print(f"Train rows: {len(X_train)}, validation rows: {len(X_val)}")
print(f"R2:   {r2_score(y_val, pred):.4f}")
print(f"MAE:  {mean_absolute_error(y_val, pred):.2f}")
print(f"RMSE: {mean_squared_error(y_val, pred) ** 0.5:.2f}")

# Q5 chart: predicted vs actual (dots near the red line = good predictions)
plt.figure(figsize=(6, 6))
plt.scatter(y_val, pred)
plt.plot([0, 270], [0, 270], "r--", label="Perfect prediction")
plt.xlabel("Actual IceCreamsSold"); plt.ylabel("Predicted IceCreamsSold")
plt.title("Ice cream sales: predicted vs actual (validation set)")
plt.legend(); plt.tight_layout(); plt.savefig("q5_predicted_vs_actual.png", dpi=120)

# Q8: predict three new days
new_days = pd.DataFrame([
    {"DayOfWeek": "Saturday", "Month": "July", "Temperature": 90.0, "Rainfall": 0.0},
    {"DayOfWeek": "Wednesday", "Month": "May", "Temperature": 70.0, "Rainfall": 0.2},
    {"DayOfWeek": "Tuesday", "Month": "October", "Temperature": 55.0, "Rainfall": 0.8},
])
# Give the new days the same columns as the training data
new_X = pd.get_dummies(new_days).reindex(columns=X.columns, fill_value=0)
for (_, row), p in zip(new_days.iterrows(), model.predict(new_X)):
    print(f"Prediction: {row.DayOfWeek}, {row.Month}, {row.Temperature} degrees, {row.Rainfall} rain -> {p:.0f} ice creams")
