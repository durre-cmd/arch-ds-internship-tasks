"""
Titanic Survival Classification
Author: durre-cmd
Date: 2026-01-12

This script loads the Titanic dataset, cleans and preprocesses it,
trains a Random Forest classifier, and evaluates its performance.
"""

# Step 1: Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load the dataset
data = sns.load_dataset('titanic')  # seaborn built-in dataset
# If you have CSV: data = pd.read_csv('titanic.csv')

# Step 3: Explore dataset
print("First 5 rows:\n", data.head())
print("\nDataset info:\n", data.info())

# Step 4: Handle missing values
data['age'].fillna(data['age'].median(), inplace=True)
data['embarked'].fillna(data['embarked'].mode()[0], inplace=True)

# Step 5: Encode categorical variables
le_sex = LabelEncoder()
data['sex'] = le_sex.fit_transform(data['sex'])

le_embarked = LabelEncoder()
data['embarked'] = le_embarked.fit_transform(data['embarked'])

# Step 6: Select features and target
features = ['pclass', 'sex', 'age', 'fare', 'sibsp', 'parch', 'embarked']
X = data[features]
y = data['survived']

# Step 7: Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 9: Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 10: Make predictions
y_pred = model.predict(X_test)

# Step 11: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

# Step 12: Optional - Feature Importance
import matplotlib.pyplot as plt
importances = model.feature_importances_
plt.figure(figsize=(8,5))
plt.bar(features, importances)
plt.title("Feature Importance in Random Forest")
plt.show()
