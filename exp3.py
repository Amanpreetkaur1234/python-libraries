import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (confusion_matrix,ConfusionMatrixDisplay,precision_score,recall_score,f1_score,accuracy_score,classification_report)

data = load_breast_cancer()
X =pd.DataFrame(
    data.data,
    columns=data.feature_names
)
y= data.target
print("="*60)
print("BREAST CANCER CLASSIFICATION")
print("="*60)
print("\n Dataset Shape:")
print(X.shape)
print("\n Class Names:")
print(data.target_names)
print("\n First 5 rows:")
print(X.head())
X_train,X_test,y_train,y_test =train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
print("\n Training Samples:",X_train.shape[0])
print("\n Testing Samples:",X_test.shape[0])
scaler =StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled =scaler.transform(X_test)
logistic_model =LogisticRegression(
    max_iter=5000,
    random_state=42
)
logistic_model.fit(
    X_train_scaled,
    y_train
)
y_pred_logistic =logistic_model.predict(
    X_test_scaled
)
tree_model =DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)
tree_model.fit(
    X_train,
    y_train
)
y_pred_tree =tree_model.predict(
    X_test
)
knn_model =KNeighborsClassifier(
    n_neighbors=5
)
knn_model.fit(
    X_train_scaled,
    y_train
)
y_pred_knn =knn_model.predict(
    X_train_scaled
)
models ={
    "Logistic Regression": y_pred_logistic,
    "Decision Tree ": y_pred_tree,
    "k-NN": y_pred_knn
}
results =[]
for model_name ,y_pred in models.items():
    cm = confusion_matrix(
        y_test,
        y_pred
    )
    accuracy = accuracy_score(
        y_test,
        y_pred
    )
    precision = precision_score(
        y_test,
        y_pred
    )
    recall =recall_score(
        y_test,
        y_pred
    )
    f1 =f1_score(
        y_test,
        y_pred
    )
    results.append({
        "Model": model_name,
        "Accuracy":accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score":f1
    })
    print("\n")
    print("="*60)
    print(model_name)
    print("="*60)
    print("\nConfusion Matrix:")
    print(cm)
    print("\n accuracy:",round(accuracy,4))
    print("\n Precision:",round(precision,4))
    print("\n Recall:",round(recall,4))
    print("\n F1-Score:",round(f1,4))
    print('\n Classification Report;')
    print(
        classification_report(
            y_test,y_pred,target_names=data.target_names
        )
    )
  results_df =pd.DataFrame(results)  
print("\n")
print("="*60)
print("Model Performance Comparison")
print("="*60)
print(results_df.to_string(index=False))
fig,axes =plt.subplots(1,2)
# ============================================================
# 11. PLOT CONFUSION MATRICES
# ============================================================
fig, axes = plt.subplots(
    1,
    3,
    figsize=(16, 5)
)
for ax, (model_name, y_pred) in zip(
    axes,
    models.items()
):

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=data.target_names
    )

    disp.plot(
        ax=ax,
        values_format="d"
    )

    ax.set_title(
        model_name
    )

plt.tight_layout()
plt.show()


# ============================================================
# 12. PLOT PERFORMANCE COMPARISON
# ============================================================

results_plot = results_df.set_index(
    "Model"
)

results_plot[
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score"
    ]
].plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title(
    "Performance Comparison of Classification Algorithms"
)

plt.xlabel(
    "Classification Algorithm"
)

plt.ylabel(
    "Score"
)

plt.ylim(
    0,
    1.1
)

plt.xticks(
    rotation=0
)

plt.legend(
    loc="lower right"
)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.5
)

plt.tight_layout()
plt.show()


# ============================================================
# 13. FIND THE BEST MODEL
# ============================================================

best_model = results_df.loc[
    results_df["F1-Score"].idxmax()
]

print("\n")
print("=" * 60)
print("BEST CLASSIFICATION MODEL")
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


# ============================================================
# 14. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print("""
Three classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree
3. k-Nearest Neighbors (k-NN)

The models were evaluated using:

- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1-Score

The model having the highest F1-Score is considered the
best-performing model for this classification task.
""")