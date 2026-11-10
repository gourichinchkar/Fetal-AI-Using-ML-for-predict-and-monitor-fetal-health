# Step 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from flask import Flask, request, jsonify

# Step 2: Load dataset (download from Kaggle and place in ./data folder)
data = pd.read_csv("fetal_health.csv")

# Step 3: Features and target
X = data.drop("fetal_health", axis=1)
y = data["fetal_health"]

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 5: Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Model training
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Step 7: Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 8: Save model + scaler
joblib.dump(model, "fetal_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Step 9: Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    
    labels = {1: "Normal", 2: "Suspect", 3: "Pathological"}
    result = labels.get(prediction, "Unknown")
    
    return jsonify({"prediction": int(prediction), "status": result})

if __name__ == '__main__':
    app.run(debug=True)