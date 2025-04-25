import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# === Configuration ===
base_dirs = {
    "Normal": 0,
    "Outer Ring": 1,
    "10g": 2,
    "37g" : 3    
}
target_columns = [
    "Total Energy", "Max Amplitude", "Frequency of Max Amplitude",
    "Spectral Centroid", "Spectral Bandwidth", "Spectral Entropy",
    "Mean", "Std Dev", "Variance", "Skewness", "Kurtosis",
    "Entropy", "Peak-to-Peak", "Zero Crossing Rate"
]
root_path = "Second Batch"  # CHANGE TO YOUR FOLDER

# === Data Collection ===
data = []
labels = []

for class_name, class_label in base_dirs.items():
    class_dir = os.path.join(root_path, class_name)
    for dirpath, _, filenames in os.walk(class_dir):
        for filename in filenames:
            if filename.endswith(".csv"):
                file_path = os.path.join(dirpath, filename)
                try:
                    df = pd.read_csv(file_path, usecols=target_columns, nrows=1)
                    data.append(df.iloc[0].values)
                    labels.append(class_label)
                except Exception as e:
                    print(f"⚠️ Skipped {file_path}: {e}")

print(f"🔢 Total samples collected: {len(data)} (Normal: {labels.count(0)}, Outer Ring: {labels.count(1)}, 10g: {labels.count(2)}, 37g: {labels.count(3)})")

# === Check if data was found ===
if not data:
    raise ValueError("❌ No valid data found. Check your folder structure or column names.")

# === Convert to DataFrame ===
df_data = pd.DataFrame(data, columns=target_columns)
X_train, X_test, y_train, y_test = train_test_split(df_data, labels, test_size=0.2, random_state=42)

# === Model Training ===

# === Random Forest Model ===
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# === Predictions ===
y_pred = model.predict(X_test)

# === Evaluation ===
print("✅ Model Parameters:")
print(model.get_params())

print("\n✅ Classification Report:")
report = classification_report(y_test, y_pred, target_names=["Normal", "Outer Ring", "10g", "37g"])
print(report)

# === Confusion Matrix ===
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=["Normal", "Outer Ring", "10g", "37g"], yticklabels=["Normal", "Outer Ring", "10g", "37g"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# === Save Model ===
joblib.dump(model, "random_forest_model.pkl")
print("💾 Model saved as 'random_forest_model.pkl'")
