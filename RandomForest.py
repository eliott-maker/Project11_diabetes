# 1. Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, auc,
    precision_recall_curve
)
from sklearn.preprocessing import label_binarize, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

# 2. Output-Ordner
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# 3. Daten laden
df = pd.read_csv("dataset.csv")

# 4. Age-Mapping (optional)
age_map = {1:21,2:27,3:32,4:37,5:42,6:47,7:52,8:57,9:62,10:67,11:72,12:77,13:82}
df["Age_years"] = df["Age"].map(age_map)
df.drop(columns=["Age"], inplace=True)

# 5. Features & Ziel
X = df.drop(columns=["Diabetes_012"])
y = df["Diabetes_012"]

# 6. Feature-Typen definieren
ordinal_feats     = ["Income"]
numeric_feats     = [c for c in X.select_dtypes(include="number") if c not in ordinal_feats]
categorical_feats = [c for c in X.select_dtypes(include="object") if c not in ordinal_feats]

# 7. Preprocessing definieren
num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
ord_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ordinal", OrdinalEncoder())
])
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(drop='first', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, numeric_feats),
    ("ord", ord_pipeline, ordinal_feats),
    ("cat", cat_pipeline, categorical_feats)
])

# 8. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 9. Transformation + SMOTE
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc  = preprocessor.transform(X_test)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_proc, y_train)

# 10. Random Forest Training
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",  # Klassen­gewichte ausbalancieren
    random_state=42
)
model.fit(X_train_resampled, y_train_resampled)

# 11. Evaluation
y_pred  = model.predict(X_test_proc)
y_score = model.predict_proba(X_test_proc)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 12. Barplot: Precision/Recall/F1
report     = classification_report(y_test, y_pred, output_dict=True)
metrics_df = pd.DataFrame(report).transpose().iloc[:3][["precision", "recall", "f1-score"]]
metrics_df.plot(kind="bar", figsize=(10, 6))
plt.title("Precision, Recall, F1-Score per Class")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "precision_recall_f1_barplot.png"))
plt.close()

# 13. Confusion Matrix
cm  = confusion_matrix(y_test, y_pred)
cmd = ConfusionMatrixDisplay(cm, display_labels=["No Diabetes", "Prediabetes", "Diabetes"])
fig, ax = plt.subplots()
cmd.plot(cmap="Blues", ax=ax)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close()

# 14. Feature Importance
importances = model.feature_importances_
num_names   = numeric_feats
ord_names   = ordinal_feats

ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
cat_names = []
if hasattr(ohe, "categories_"):
    cat_names = ohe.get_feature_names_out().tolist()

feature_names = num_names + ord_names + cat_names
indices       = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=np.array(feature_names)[indices])
plt.title("Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "feature_importances.png"))
plt.close()

# 15. ROC & Precision-Recall Kurven (Multiclass)
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
fpr, tpr, roc_auc, precision, recall = {}, {}, {}, {}, {}
auc_values = []

# ROC Curve
plt.figure()
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i]        = auc(fpr[i], tpr[i])
    auc_values.append(roc_auc[i])
    plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve Random Forest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid(True)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "roc_curve.png"))
plt.close()

# Precision-Recall Curve
plt.figure()
for i in range(3):
    precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
    plt.plot(recall[i], precision[i], label=f"Class {i}")

plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"))
plt.close()

# 16. Durchschnittliche AUC ausgeben und NaNs ignorieren
average_auc = np.nanmean(auc_values)
print("Average AUC across all classes:", average_auc)
