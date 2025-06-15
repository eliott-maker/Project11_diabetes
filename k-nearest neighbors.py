import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer, label_binarize
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier

from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline

# 1) Load dataset and map age categories to numerical midpoints
age_map = {1:21,2:27,3:32,4:37,5:42,6:47,7:52,8:57,9:62,10:67,11:72,12:77,13:82}
df = pd.read_csv('dataset.csv')
df['Age_years'] = df['Age'].map(age_map)
df.drop(columns=['Age'], inplace=True)

y = df['Diabetes_012'].astype(int)
X = df.drop(columns=['Diabetes_012'])

# 2) Define feature types
ordinal_feats = ['Income']
numeric_feats = ['BMI', 'PhysHlth', 'MentHlth', 'Age_years']
categorical_feats = [c for c in X.columns if c not in numeric_feats + ordinal_feats]

# 3) Build preprocessing pipelines with skew-based transformations
numeric_transformers = []
for feat in numeric_feats:
    steps = [('imputer', SimpleImputer(strategy='median'))]
    if abs(df[feat].skew()) > 1:
        steps += [
            ('log1p', FunctionTransformer(np.log1p, validate=False)),
            ('scale', RobustScaler())
        ]
    else:
        steps += [('scale', StandardScaler())]
    numeric_transformers.append((feat, SklearnPipeline(steps), [feat]))

ord_pipeline = SklearnPipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder())
])
cat_pipeline = SklearnPipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers = numeric_transformers + [
        ('ord', ord_pipeline, ordinal_feats),
        ('cat', cat_pipeline, categorical_feats)
    ],
    remainder = 'drop'
)

# 4) Apply preprocessing and create DataFrame with feature names
X_encoded = preprocessor.fit_transform(X)
feature_names = (
    numeric_feats + ordinal_feats +
    preprocessor.named_transformers_['cat']
        .named_steps['onehot']
        .get_feature_names_out(categorical_feats).tolist()
)
X = pd.DataFrame(X_encoded, columns=feature_names)

# 5) Stratified Split (20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6) Undersampling & Scaling for Mutual Information
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X_train, y_train)
scaler = StandardScaler().fit(X_res)
X_res_scaled = scaler.transform(X_res)

# 7) Mutual Information and Top-5 Feature Selection
mi = mutual_info_classif(X_res_scaled, y_res, random_state=42)
feat_scores = pd.Series(mi, index=X.columns)
top5 = feat_scores.nlargest(5).index.tolist()
print("Top-5 features:", top5)

# 8) Reduce datasets to Top-5 Features
X_train_sel = X_train[top5]
X_test_sel  = X_test[top5]

# 9) Pipeline for final model (undersampling, scaling, KNN)
pipeline = Pipeline([
    ('under', RandomUnderSampler(random_state=42)),
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

param_dist = {
    'knn__n_neighbors': [3, 5, 7, 9, 11, 15],
    'knn__weights': ['uniform', 'distance']
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=12,
    scoring='balanced_accuracy',
    cv=cv,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

search.fit(X_train_sel, y_train)
print("Best Params:", search.best_params_)
print("Best Score:", search.best_score_)

# 10) Evaluation
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

y_proba = search.predict_proba(X_test_sel)
y_pred  = search.predict(X_test_sel)

print("Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Diabetes','Prediabetes','Diabetes'],
            yticklabels=['No Diabetes','Prediabetes','Diabetes'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# Precision-Recall and ROC Curves

y_test_bin = label_binarize(y_test, classes=[0,1,2])
class_names = ['No Diabetes','Prediabetes','Diabetes']

fig, ax = plt.subplots(figsize=(6,5))
for i, cls in enumerate(class_names):
    PrecisionRecallDisplay.from_predictions(
        y_test_bin[:, i], y_proba[:, i], name=cls, ax=ax
    )
plt.legend(title='Class')
plt.title('Precision-Recall Curve')
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(6,5))
for i, cls in enumerate(class_names):
    RocCurveDisplay.from_predictions(
        y_test_bin[:, i],
        y_proba[:, i], name=cls, ax=ax
    )
plt.ylabel("True Positive Rate")
plt.legend()
plt.title('ROC Curve K-NN')
plt.tight_layout()
plt.show()
