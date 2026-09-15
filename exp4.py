import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

# Keep the CSV file in the same folder as this Python file
data = pd.read_csv("diabetes_prediction_dataset.csv")

print("Dataset Shape:", data.shape)

print("\nFirst 5 Rows:")
print(data.head())

print("\nColumn Names:")
print(data.columns.tolist())

print("\nDataset Information:")
print(data.info())

print("\nMissing Values:")
print(data.isnull().sum())


# ============================================================
# 2. ENCODE CATEGORICAL COLUMNS
# ============================================================

# Convert gender into numerical values
gender_encoder = LabelEncoder()
data["gender"] = gender_encoder.fit_transform(data["gender"])

# Convert smoking_history into numerical values
smoking_encoder = LabelEncoder()
data["smoking_history"] = smoking_encoder.fit_transform(
    data["smoking_history"]
)


# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

X = data.drop("diabetes", axis=1)
y = data["diabetes"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget Distribution:")
print(y.value_counts())


# ============================================================
# 4. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)


# ============================================================
# 5. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 6. NAIVE BAYES
# ============================================================

nb_model = GaussianNB()

nb_model.fit(
    X_train_scaled,
    y_train
)

y_pred_nb = nb_model.predict(
    X_test_scaled
)


# ============================================================
# 7. SUPPORT VECTOR MACHINE
# ============================================================

svm_model = SVC(
    kernel="linear",
    random_state=42
)

svm_model.fit(
    X_train_scaled,
    y_train
)

y_pred_svm = svm_model.predict(
    X_test_scaled
)


# ============================================================
# 8. NAIVE BAYES RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("NAIVE BAYES")
print("=" * 60)

cm_nb = confusion_matrix(
    y_test,
    y_pred_nb
)

print("\nConfusion Matrix:")
print(cm_nb)

print("\nAccuracy:",
      round(accuracy_score(y_test, y_pred_nb), 4))

print("Precision:",
      round(precision_score(y_test, y_pred_nb), 4))

print("Recall:",
      round(recall_score(y_test, y_pred_nb), 4))

print("F1 Score:",
      round(f1_score(y_test, y_pred_nb), 4))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred_nb,
    target_names=["No Diabetes", "Diabetes"]
))


# ============================================================
# 9. SVM RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("SUPPORT VECTOR MACHINE (SVM)")
print("=" * 60)

cm_svm = confusion_matrix(
    y_test,
    y_pred_svm
)

print("\nConfusion Matrix:")
print(cm_svm)

print("\nAccuracy:",
      round(accuracy_score(y_test, y_pred_svm), 4))

print("Precision:",
      round(precision_score(y_test, y_pred_svm), 4))

print("Recall:",
      round(recall_score(y_test, y_pred_svm), 4))

print("F1 Score:",
      round(f1_score(y_test, y_pred_svm), 4))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred_svm,
    target_names=["No Diabetes", "Diabetes"]
))


# ============================================================
# 10. COMPARISON TABLE
# ============================================================

results = pd.DataFrame({

    "Model": [
        "Naive Bayes",
        "SVM"
    ],

    "Accuracy": [
        accuracy_score(y_test, y_pred_nb),
        accuracy_score(y_test, y_pred_svm)
    ],

    "Precision": [
        precision_score(y_test, y_pred_nb),
        precision_score(y_test, y_pred_svm)
    ],

    "Recall": [
        recall_score(y_test, y_pred_nb),
        recall_score(y_test, y_pred_svm)
    ],

    "F1-Score": [
        f1_score(y_test, y_pred_nb),
        f1_score(y_test, y_pred_svm)
    ]
})


print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results.to_string(index=False))


# ============================================================
# 11. CONFUSION MATRIX - NAIVE BAYES
# ============================================================

disp_nb = ConfusionMatrixDisplay(
    confusion_matrix=cm_nb,
    display_labels=["No Diabetes", "Diabetes"]
)

disp_nb.plot(values_format="d")

plt.title("Naive Bayes - Confusion Matrix")
plt.tight_layout()
plt.show()


# ============================================================
# 12. CONFUSION MATRIX - SVM
# ============================================================

disp_svm = ConfusionMatrixDisplay(
    confusion_matrix=cm_svm,
    display_labels=["No Diabetes", "Diabetes"]
)

disp_svm.plot(values_format="d")

plt.title("SVM - Confusion Matrix")
plt.tight_layout()
plt.show()


# ============================================================
# 13. PERFORMANCE COMPARISON GRAPH
# ============================================================

results.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1-Score"]
].plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Naive Bayes vs SVM Performance")
plt.xlabel("Model")
plt.ylabel("Score")
plt.ylim(0, 1.1)
plt.xticks(rotation=0)
plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.5
)

plt.tight_layout()
plt.show()


# ============================================================
# 14. BEST MODEL
# ============================================================

best_model = results.loc[
    results["F1-Score"].idxmax()
]

print("\n")
print("=" * 60)
print("BEST MODEL")
print("=" * 60)

print(
    "Best Model:",
    best_model["Model"]
)

print(
    "Accuracy:",
    round(best_model["Accuracy"], 4)
)

print(
    "Precision:",
    round(best_model["Precision"], 4)
)

print(
    "Recall:",
    round(best_model["Recall"], 4)
)

print(
    "F1-Score:",
    round(best_model["F1-Score"], 4)
)