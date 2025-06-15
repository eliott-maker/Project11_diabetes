import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, f_oneway

# 1) Load the dataset
df = pd.read_csv('dataset.csv')
target = 'Diabetes_012'

# 2) Basic overview
print(f"Dataset shape: {df.shape}\n")
df.info()
print("\nDescriptive statistics:")
print(df.describe())
print("\nMissing values per column:")
print(df.isnull().sum())

# 3) Target distribution
class_counts = df[target].value_counts().sort_index()
classes = ['No Diabetes', 'Pre-Diabetes', 'Diabetes']
counts = class_counts.values
colors = ['green', 'orange', 'red']

plt.figure(figsize=(8, 6))
positions = range(len(classes))
plt.bar(positions, counts, color=colors, edgecolor='black')
plt.xticks(positions, classes)
plt.xlabel('Diabetes Class', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Target Class Distribution', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=1200)
plt.close()

# 4) Automatic feature splitting
categorical_features = [col for col in df.columns if col != target and (df[col].dtype == 'object' or df[col].nunique() < 10)]
numerical_features = [col for col in df.columns if col not in categorical_features + [target]]
print("\nCategorical features:", categorical_features)
print("Numerical features:", numerical_features)

# 5) Chi-square tests
print("\n== Chi-Square Tests ==")
for col in categorical_features:
    contingency = pd.crosstab(df[col], df[target])
    chi2, p, dof, _ = chi2_contingency(contingency)
    print(f"{col:25s} χ² = {chi2:8.2f}, p-value = {p:.4e}, dof = {dof}")

# 6) ANOVA tests
print("\n== ANOVA Tests ==")
for col in numerical_features:
    groups = [grp[col].values for _, grp in df.groupby(target)]
    f_stat, p_val = f_oneway(*groups)
    print(f"{col:25s} F = {f_stat:8.2f}, p-value = {p_val:.4e}")

# 7) Target class percentages
percentages = df[target].value_counts(normalize=True).sort_index() * 100
for idx, perc in enumerate(percentages):
    print(f"{classes[idx]} ({idx}): {perc:.2f}%")

# 8) Map Age categories to midpoint years
age_map = {1:21,2:27,3:32,4:37,5:42,6:47,7:52,8:57,9:62,10:67,11:72,12:77,13:82}
if 'Age' in df.columns:
    df['Age_years'] = df['Age'].map(age_map)
    numerical_features = ['BMI', 'Age_years', 'PhysHlth', 'MentHlth']

# ============================
# Feature Visualizations
# ============================

# 9) Numerical features: raw and log-transformed histograms
# 9a) Combined BMI raw vs. log-transformed comparison
if 'BMI' in numerical_features:
    bmi_data = df['BMI'].dropna()
    bmi_log = np.log1p(bmi_data)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    # Raw BMI
    axes[0].hist(bmi_data, bins=30, edgecolor='black')
    axes[0].set_title('BMI - Raw Data Histogram', fontweight='bold')
    axes[0].set_xlabel('BMI')
    axes[0].set_ylabel('Count')
    # Log-transformed BMI (natural log)
    axes[1].hist(bmi_log, bins=30, edgecolor='black')
    axes[1].set_title('BMI - ln(1 + BMI) Transformed Histogram', fontweight='bold')
    axes[1].set_xlabel('ln(1 + BMI)')
    plt.tight_layout()
    plt.savefig('BMI_comparison.png', dpi=1200)
    plt.close()

# 9b) Other numerical features: individual histograms and log if skewed
for feature in numerical_features:
    if feature in ['BMI', 'Age_years']:
        continue
    data = df[feature].dropna()
    # Raw histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(data, kde=True, bins=30, color='steelblue')
    plt.title(f"{feature} - Raw Data Histogram", fontsize=14, fontweight='bold')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f"{feature}_raw_histogram.png", dpi=1200)
    plt.close()
    # Log-transform if skewed
    if abs(data.skew()) > 1:
        trans = np.log1p(data)
        plt.figure(figsize=(8, 5))
        sns.histplot(trans, kde=True, bins=30, color='darkorange')
        plt.title(f"{feature} - ln(1 + {feature}) Transformed Histogram", fontsize=14, fontweight='bold')
        plt.xlabel(f"ln(1 + {feature})")
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(f"{feature}_log_transformed_histogram.png", dpi=1200)
        plt.close()

# 10) Age as ordinal category: bar plot
if 'Age' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Age', data=df, order=sorted(df['Age'].unique()), color='slateblue')
    plt.title("Age Group Distribution (Ordinal)", fontsize=14, fontweight='bold')
    plt.xlabel("Age Group Code (1=18–24 … 13=80+)")
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('age_ordinal_barplot.png', dpi=1200)
    plt.close()

# 11) Age_years: discrete bar chart aligned at mapped midpoints
if 'Age_years' in df.columns:
    age_counts = df['Age_years'].value_counts().sort_index()
    ages = age_counts.index.to_list()
    counts_age = age_counts.values
    plt.figure(figsize=(8, 5))
    plt.bar(ages, counts_age, width=4, color='teal', edgecolor='black')
    plt.title("Age (years) Distribution", fontsize=14, fontweight='bold')
    plt.xlabel("Age (years)")
    plt.ylabel("Count")
    plt.xlim(0, max(ages) + 5)
    plt.xticks(ages)
    plt.tight_layout()
    plt.savefig('age_years_countplot.png', dpi=1200)
    plt.close()

# 12) Income as ordinal: bar plot
if 'Income' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Income', data=df, order=sorted(df['Income'].unique()), color='lightcoral')
    plt.title("Income Distribution (Ordinal)", fontsize=14, fontweight='bold')
    plt.xlabel('Income Category')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('income_distribution.png', dpi=1200)
    plt.close()

# 13) Categorical features: bar plots
for col in categorical_features:
    plt.figure(figsize=(8, 5))
    sns.countplot(x=col, data=df, color='lightcoral')
    plt.title(f"{col} Distribution", fontsize=14, fontweight='bold')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{col}_barplot.png", dpi=1200)
    plt.close()


