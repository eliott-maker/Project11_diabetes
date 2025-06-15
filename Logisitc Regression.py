import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, f_oneway, kruskal

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    RobustScaler,
    OrdinalEncoder,
    FunctionTransformer,
    label_binarize
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    PrecisionRecallDisplay,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)


def load_and_map(filepath):
    df = pd.read_csv(filepath)
    df['Diabetes_status'] = df['Diabetes_012'].map({
        0: "No Diabetes",
        1: "Pre-Diabetes",
        2: "Diabetes"
    })
    return df

"""
def perform_eda(df):
    # Categorical chi-square
    cats = [c for c in df.columns if c != 'Diabetes_012' and
            (df[c].dtype == 'object' or df[c].nunique() < 10)]
    print("== Chi-square tests (categorical) ==")
    for col in cats:
        ct = pd.crosstab(df[col], df['Diabetes_012'])
        chi2, p, dof, _ = chi2_contingency(ct)
        print(f"{col:25s} χ²={chi2:8.2f}, p={p:.2e}, dof={dof}")
    # Numeric ANOVA/Kruskal
    nums = [c for c in df.columns if c not in cats + ['Diabetes_012']]
    print("\n== ANOVA vs. Kruskal–Wallis (numeric) ==")
    for col in nums:
        skew = df[col].skew()
        groups = [g[col].values for _, g in df.groupby('Diabetes_012')]
        if abs(skew) <= 1.0:
            stat, pval = f_oneway(*groups)
            test, reason = "ANOVA", f"|skew|={skew:.2f}≤1"
        else:
            stat, pval = kruskal(*groups)
            test, reason = "Kruskal–Wallis", f"|skew|={skew:.2f}>1"
        print(f"{col:25s} {test:15s} stat={stat:8.2f}, p={pval:.2e} ({reason})")
"""

def map_features(df):
    age_map = {1:21,2:27,3:32,4:37,5:42,6:47,7:52,8:57,9:62,10:67,11:72,12:77,13:82}
    df['Age_years'] = df['Age'].map(age_map)
    return df


def split_and_balance(df):
    X = df.drop(columns=['Diabetes_012','Diabetes_status','Age'])
    y = df['Diabetes_status']
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    train = pd.concat([X_tr, y_tr.rename('Diabetes_status')], axis=1)
    n_min = train['Diabetes_status'].value_counts().min()
    train_under = train.groupby('Diabetes_status', group_keys=False).apply(
        lambda g: g.sample(n=n_min, random_state=42)
    )
    X_tr_bal = train_under.drop(columns='Diabetes_status')
    y_tr_bal = train_under['Diabetes_status']
    return X_tr_bal, X_te, y_tr_bal, y_te


def build_preprocessor(X_sample, numeric_feats, ordinal_feats, categorical_feats):
    numeric_transformers = []
    for feat in numeric_feats:
        skew = X_sample[feat].skew()
        steps = [('imputer', SimpleImputer(strategy='median'))]
        if abs(skew) > 1.0:
            steps += [('log1p', FunctionTransformer(np.log1p, validate=False)),
                      ('scale', RobustScaler())]
        else:
            steps += [('scale', StandardScaler())]
        numeric_transformers.append((feat, Pipeline(steps), [feat]))

    ord_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers = numeric_transformers + [
            ('ord_income', ord_pipe, ordinal_feats),
            ('cat',       cat_pipe, categorical_feats)
        ], remainder='drop'
    )
    return preprocessor


def tune_logistic(X, y):
    param_grid = {'C':[0.01, 0.1, 1], 'penalty':['l1','l2']}
    base = LogisticRegression(solver='saga', max_iter=2000, random_state=42)
    grid = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='f1_macro',
        n_jobs=-1
    )
    grid.fit(X, y)
    print("Best hyperparameters:", grid.best_params_)
    return grid.best_estimator_


def main():
    df = load_and_map("dataset.csv")
#   perform_eda(df)
    df = map_features(df)

    X_tr, X_test, y_tr, y_test = split_and_balance(df)

    numeric_feats = ['BMI','Age_years','PhysHlth','MentHlth']
    ordinal_feats = ['Income']
    categorical_feats = [c for c in X_tr.columns if c not in numeric_feats + ordinal_feats]

    preprocessor = build_preprocessor(X_tr, numeric_feats, ordinal_feats, categorical_feats)

    # Preprocess
    X_tr_p = preprocessor.fit_transform(X_tr)
    X_te_p = preprocessor.transform(X_test)

    # List all feature names after encoding
    feature_names = (
        numeric_feats + ordinal_feats +
        preprocessor.named_transformers_['cat']
            .named_steps['onehot']
            .get_feature_names_out(categorical_feats).tolist()
    )
    print(f"All features after encoding ({len(feature_names)}):")
    for name in feature_names:
        print(name)

    # Hyperparameter tuning on training data
    best_est = tune_logistic(X_tr_p, y_tr)

    # RFECV feature selection using best_est
    rfecv = RFECV(
        estimator=best_est, step=1,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='f1_macro', min_features_to_select=5, n_jobs=-1
    )
    rfecv.fit(X_tr_p, y_tr)
    feature_names = (
        numeric_feats + ordinal_feats +
        preprocessor.named_transformers_['cat']
            .named_steps['onehot']
            .get_feature_names_out(categorical_feats).tolist()
    )
    selected = np.array(feature_names)[rfecv.support_]
    print("Selected features nach RFECV:\n", selected)

    X_tr_sel = rfecv.transform(X_tr_p)
    X_te_sel = rfecv.transform(X_te_p)

    # Calibrated classifier
    clf = CalibratedClassifierCV(estimator=best_est, cv=5, method='sigmoid')
    clf.fit(X_tr_sel, y_tr)

    # Evaluation
    y_pred = clf.predict(X_te_sel)
    y_prob = clf.predict_proba(X_te_sel)
    y_bin = label_binarize(y_test, classes=clf.classes_)
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"Macro ROC AUC: {roc_auc_score(y_bin, y_prob, average='macro', multi_class='ovr'):.4f}")

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=clf.classes_, cmap=plt.cm.Blues
    )
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

    for i, cls in enumerate(clf.classes_):
        prec, rec, _ = precision_recall_curve(y_bin[:,i], y_prob[:,i])
        PrecisionRecallDisplay(precision=prec, recall=rec, estimator_name=cls).plot()
    plt.title('PR Curves')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.legend(loc='best'); plt.tight_layout(); plt.show()

    # Multiclass ROC
    class_order = ['No Diabetes','Pre-Diabetes','Diabetes']
    idx = [list(clf.classes_).index(c) for c in class_order]
    y_prob_sorted = y_prob[:, idx]
    y_bin2 = label_binarize(y_test, classes=class_order)
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(y_bin2.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_bin2[:,i], y_prob_sorted[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure(figsize=(10,6))
    colors = ['blue','orange','green']
    for i, label in enumerate(class_order):
        plt.plot(fpr[i], tpr[i], label=f"{label} (AUC={roc_auc[i]:.2f})")
    plt.plot([0,1],[0,1],'k--'); plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC Curve'); plt.legend(loc='lower right'); plt.grid(True); plt.tight_layout(); plt.show()

if __name__ == '__main__':
    main()
