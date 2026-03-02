from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import joblib

# 1. Load Data
df = pd.read_csv('ship_fire_data.csv')
X = df.drop('risk_level', axis=1)
y = df['risk_level']

# 2. Split (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize and Train Model
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate
y_pred = model.predict(X_test)
print("\n--- Model Performance Report ---")
print(classification_report(y_test, y_pred, target_names=['Low Risk', 'Medium Risk', 'High Risk']))

# 5. Feature Importance (Which sensor matters most?)
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop Predictors of Fire Risk:")
print(importances)
joblib.dump(model, 'fire_model.pkl')