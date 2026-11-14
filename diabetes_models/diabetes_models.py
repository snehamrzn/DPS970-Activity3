# DPS970 Activity 3 - Q16: predict diabetes with 3 scikit-learn models.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. Create a data frame from the uploaded data
data = pd.read_csv("diabetes.csv")
print("Rows loaded:", len(data))

# 2. Fix the issues from Q15
print("Missing values:", data.isnull().sum().sum())                   # check: none
print("Repeated patients:", data["PatientID"].duplicated().sum())    # 105
data = data.drop_duplicates(subset="PatientID")                       # keep one row per patient
data = data.drop(columns="PatientID")                                 # ID says nothing about health
print("Rows after cleaning:", len(data))

X = data.drop(columns="Diabetic")    # the 8 health measurements
y = data["Diabetic"]                 # the answer: 1 = diabetic, 0 = not

# Stratified 80/20 split keeps the same 2:1 mix in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print("Train rows:", len(X_train), "| Test rows:", len(X_test))

# 3. Create 3 different types of models
#    class_weight="balanced" gives the smaller diabetic group more attention
models = {
    "Logistic Regression": make_pipeline(StandardScaler(),   # scale the columns
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=42),
}

# 4. Train each model and test it on data it has never seen
results = {}           # accuracy of each model (for the tests)
all_scores = {}        # all 4 metrics for each model (for the bar chart)
all_predictions = {}   # each model's guesses (for the confusion matrices)
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    results[name] = accuracy
    print(f"\n{name}")
    print(f"  Accuracy : {accuracy:.3f}")
    print(f"  Precision: {precision_score(y_test, predictions):.3f}")
    print(f"  Recall   : {recall_score(y_test, predictions):.3f}")
    print(f"  F1 score : {f1_score(y_test, predictions):.3f}")
    all_scores[name] = [accuracy, precision_score(y_test, predictions),
                        recall_score(y_test, predictions), f1_score(y_test, predictions)]
    all_predictions[name] = predictions
    # Confusion matrix: rows = real answer, columns = model's guess (0 = not diabetic, 1 = diabetic)
    print("  Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

# 5. Tests: each check stops the program with an error if it fails
baseline = y_test.value_counts().max() / len(y_test)    # always guessing "not diabetic"
print(f"\nBaseline (always guess 'not diabetic'): {baseline:.3f}")

assert "PatientID" not in X.columns, "PatientID should be removed"
assert len(data) == 14895, "each patient should appear only once"
for name, accuracy in results.items():
    assert accuracy > baseline, f"{name} is no better than guessing"
    assert accuracy > 0.70, f"{name} accuracy is too low"
    print(f"TEST PASSED: {name} accuracy {accuracy:.3f} beats the baseline {baseline:.3f}")

print("\nAll tests passed.")

# 6. Charts for Q17
table = pd.DataFrame(all_scores, index=["Accuracy", "Precision", "Recall", "F1"])
chart = table.plot(kind="bar", rot=0, ylim=(0, 1), figsize=(9, 5))
chart.set_title("Model comparison on the test set")
chart.set_ylabel("Score (0 to 1)")
for bars in chart.containers:
    chart.bar_label(bars, fmt="%.3f", fontsize=8)    # print the number on each bar
plt.tight_layout()
plt.savefig("chart_model_comparison.png")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (name, predictions) in zip(axes, all_predictions.items()):
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, ax=ax, colorbar=False,
        display_labels=["Not diabetic", "Diabetic"])
    ax.set_title(name)
plt.tight_layout()
plt.savefig("chart_confusion_matrices.png")
print("Charts saved: chart_model_comparison.png and chart_confusion_matrices.png")