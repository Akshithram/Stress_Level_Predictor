import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import VotingClassifier
import lightgbm as lgb
import catboost as cb
import joblib

# Load the dataset
df = pd.read_csv('StressLevelDataset.csv')

# Selected 11 effective features
selected_features = [
    'anxiety_level', 'depression', 'sleep_quality', 'blood_pressure', 'headache',
    'social_support', 'study_load', 'teacher_student_relationship',
    'future_career_concerns', 'peer_pressure', 'extracurricular_activities'
]

X = df[selected_features]
y = df['stress_level']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# LightGBM model
lgb_model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=3,
    learning_rate=0.03,
    n_estimators=1000,
    max_depth=8,
    num_leaves=60,
    subsample=0.9,
    colsample_bytree=0.8,
    class_weight='balanced',
    random_state=42
)

# CatBoost model
catboost_model = cb.CatBoostClassifier(
    iterations=1000,
    depth=8,
    learning_rate=0.03,
    loss_function='MultiClass',
    random_state=42,
    verbose=200
)

# Ensemble voting classifier (LightGBM + CatBoost)
voting_clf = VotingClassifier(estimators=[('lgb', lgb_model), ('catboost', catboost_model)], voting='soft')

# Train the model
voting_clf.fit(X_train_scaled, y_train)

# Predict & evaluate
y_pred = voting_clf.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\nEnsemble Boosted Accuracy: {acc * 100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=['Low', 'Moderate', 'High']))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the trained ensemble model
joblib.dump(voting_clf, 'stress_ensemble_model.pkl')

# Save the scaler
joblib.dump(scaler, 'stress_scaler.pkl')
