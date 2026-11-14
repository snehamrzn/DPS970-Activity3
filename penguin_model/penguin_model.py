# DPS970 Activity 3, Q9 and Q10: guess a penguin's species from its body measurements.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. Load the data and drop the 2 rows that have missing measurements.
data = pd.read_csv("penguins.csv")
print("Rows in file:", len(data))
data = data.dropna()
print("Rows after dropping missing values:", len(data))

# 2. X = the 4 measurements (inputs). y = Species (the answer: 0, 1 or 2).
X = data.drop(columns=["Species"])
y = data["Species"]

# 3. Keep 80% for training and hide 20% for testing (same split as the ice cream model).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Train rows:", len(X_train), " Test rows:", len(X_test))

# 4. Train the model. StandardScaler puts all 4 measurements on the same scale first,
#    so body mass (thousands of grams) doesn't drown out bill depth (about 15-20 mm).
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)

# 5. Test it on the hidden 20% and print the scores.
pred = model.predict(X_test)
print("Accuracy: ", round(accuracy_score(y_test, pred), 3))
print("Precision:", round(precision_score(y_test, pred, average="macro"), 3))
print("Recall:   ", round(recall_score(y_test, pred, average="macro"), 3))
print("F1 score: ", round(f1_score(y_test, pred, average="macro"), 3))

# 6. Confusion matrix: rows = real species, columns = guessed species.
print("Confusion matrix:")
print(confusion_matrix(y_test, pred))
