import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Load the dataset
data_path = "" #Dataset Path
df = pd.read_csv(data_path)

# 2. Quick data check
print("Dataset shape:", df.shape)
print("Missing values:\n", df.isnull().sum())
print("Sample labels:\n", df['label'].value_counts())

# 3. Drop non-feature columns (e.g., filename)
X = df.drop(columns=['filename', 'label'])
y = df['label']

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Initialize and train the model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 6. Predict & evaluate
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Save the trained model
joblib.dump(model, "") #.pkl file path
print("Model saved successfully!")
