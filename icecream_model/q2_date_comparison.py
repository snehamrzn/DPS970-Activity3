# Q2: Linear regression with and without Date (ice-cream.csv)
# Run: pip install pandas scikit-learn   then   python q2_date_comparison.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv("ice-cream.csv")
y = df["IceCreamsSold"]

# Date turned into a day number so the model can use it
df["DateNum"] = pd.to_datetime(df["Date"]).map(pd.Timestamp.toordinal)

base = pd.get_dummies(df[["DayOfWeek", "Month", "Temperature", "Rainfall"]], drop_first=True)
models = {
    "Model A - WITH Date": pd.concat([df[["DateNum"]], base], axis=1),
    "Model B - WITHOUT Date": base,
}

print("Linear Regression: with vs without Date (80/20 split, random_state=42)\n")
for name, X in models.items():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"{name}")
    print(f"  R2  = {r2_score(y_test, pred):.3f}")
    print(f"  MAE = {mean_absolute_error(y_test, pred):.2f}\n")
