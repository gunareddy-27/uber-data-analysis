import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
dataset_path = r'C:\Users\anush\OneDrive\Desktop\uber\datasets\UberDataset.csv'
df = pd.read_csv(dataset_path)

# Handle missing values
df.dropna(subset=["START_DATE", "END_DATE", "PURPOSE"], inplace=True)

# Convert to datetime
df["START_DATE"] = pd.to_datetime(df["START_DATE"])
df["END_DATE"] = pd.to_datetime(df["END_DATE"])

# Feature Engineering
df["Trip_Duration"] = (df["END_DATE"] - df["START_DATE"]).dt.total_seconds() / 60  # Convert to minutes
df["Start_Hour"] = df["START_DATE"].dt.hour
df["Day_of_Week"] = df["START_DATE"].dt.dayofweek  # Monday=0, Sunday=6

# Encode categorical target variable (Trip Purpose)
label_encoder = LabelEncoder()
df["Purpose_Label"] = label_encoder.fit_transform(df["PURPOSE"])

# Select Features and Target
X = df[["Trip_Duration", "Start_Hour", "Day_of_Week"]]
y = df["Purpose_Label"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.2f}")

print("\n📌 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
