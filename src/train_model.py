import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os

df = pd.read_csv("data/features.csv")
X = df.drop("label", axis=1)
y = df["label"]

# Split data (80% train / 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\nModel Performance")
print("------------------")
print("Accuracy:", round(acc * 100, 2), "%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=3))

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/stress_model.pkl")
print("\nSaved model → models/stress_model.pkl ✔")
