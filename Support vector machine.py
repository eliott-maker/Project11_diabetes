import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc
)
from sklearn.feature_selection import mutual_info_classif
from imblearn.under_sampling import RandomUnderSampler

# 1) Load dataset
df = pd.read_csv("dataset.csv")
df['Diabetes_status'] = df['Diabetes_012'].map({0: "No Diabetes", 1: "Pre-Diabetes", 2: "Diabetes"})

#age map
df['Age_years'] = df['Age'].map({1:21,2:27,3:32,4:37,5:42,6:47,7:52,8:57,9:62,10:67,11:72,12:77,13:82})
y = df["Diabetes_012"]
X = df.drop(columns=["Diabetes_012", "Age", "Diabetes_status"])

# 2) Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 3) Undersampling nur auf Training 
train = pd.concat([X_train, y_train.rename('Diabetes_status')], axis=1)
n_min = train['Diabetes_status'].value_counts().min()
train_under = train.groupby('Diabetes_status', group_keys=False).apply(
    lambda g: g.sample(n=n_min, random_state=42)
)
X_train_under = train_under.drop(columns='Diabetes_status')
y_train_under = train_under['Diabetes_status']

# 4) Preprocessing pipeline
numeric_features = ["BMI", "PhysHlth", "MentHlth", "Age_years"]
ordinal_features = ["Income"]
categorical_features = [col for col in X.columns if col not in numeric_features + ordinal_features]

numeric_transformers = []
for feat in numeric_features:
    steps = [('imputer', SimpleImputer(strategy='median'))]
    if abs(df[feat].skew()) > 1:
        steps += [('log1p', FunctionTransformer(np.log1p, validate=False)),
                  ('scale', RobustScaler())]
    else:
        steps += [('scale', StandardScaler())]
    numeric_transformers.append((feat, Pipeline(steps), [feat]))

preprocessor = ColumnTransformer(
    transformers=numeric_transformers + [
        ('ord', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('ordinal', OrdinalEncoder())
        ]), ordinal_features),
        ('cat', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False))
        ]), categorical_features)
    ]
)

# Apply preprocessing
X_train_prep = preprocessor.fit_transform(X_train_under)
X_test_prep = preprocessor.transform(X_test)
feature_names = (
    [name for name, _, _ in numeric_transformers] +
    ordinal_features +
    preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features).tolist()
)
X_train_prep = pd.DataFrame(X_train_prep, columns=feature_names)
X_test_prep = pd.DataFrame(X_test_prep, columns=feature_names)

# 5) Feature Selection (Top 10 Mutual Information Features)
mi_scores = mutual_info_classif(X_train_prep, y_train_under, discrete_features='auto', random_state=42)
mi_df = pd.DataFrame({'Feature': X_train_prep.columns, 'MI_Score': mi_scores}).sort_values(by='MI_Score', ascending=False)
top_features = mi_df['Feature'].head(10).tolist()
X_train_final = X_train_prep[top_features]
X_test_final = X_test_prep[top_features]
print(top_features)

# --- Train SVM ---
c_value = 20
svm_model = SVC(kernel='rbf', C=c_value, gamma='scale', class_weight='balanced', probability=True, random_state=42)
svm_model.fit(X_train_final, y_train_under)

# --- Evaluate SVM ---
y_pred = svm_model.predict(X_test_final)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# Confusion matrix heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.title(f"Confusion Matrix SVM (undersampled) C-Value={c_value}")
plt.tight_layout()
plt.savefig(f"svm_US_pp_confusion_matrix_{c_value}.png")
plt.close()

# Recalls
for i in range(3):
    recall = cm[i, i] / cm[i, :].sum()
    print(f"Recall class {i}: {recall:.3f}")

# Classification Report
report = classification_report(y_test, y_pred, digits=3)
print("\nClassification Report:\n", report)

report_dict = classification_report(y_test, y_pred, digits=3, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose().round(3)
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')
table = ax.table(cellText=report_df.values, colLabels=report_df.columns, rowLabels=report_df.index, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
plt.title(f'Classification Report SVM (undersampled) C-Value={c_value}')
plt.tight_layout()
plt.savefig(f'svm_US_pp_classif_report_{c_value}.png')
plt.close()

# ROC Curve
y_test_bin = label_binarize(y_test, classes=[0,1,2])
y_score = svm_model.predict_proba(X_test_final)
fpr, tpr, roc_auc = {}, {}, {}
colors = ['blue', 'orange', 'green']
labels = ['Class 0', 'Class 1', 'Class 2']

plt.figure(figsize=(8,6))
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], label=f'{labels[i]} (AUC={roc_auc[i]:.2f})', color=colors[i])
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC Curve (SVM)")
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'svm_roc_C{c_value}.png')
plt.close()
