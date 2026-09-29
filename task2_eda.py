"""
CodeAlpha - Data Analytics Internship
TASK 2: Exploratory Data Analysis (EDA)

Dataset: Titanic (loaded through seaborn). To use your own data instead
(e.g. the CSV from Task 1), set CSV_PATH below.

Steps: ask questions -> inspect structure -> clean/detect issues ->
       find patterns & anomalies -> test hypotheses -> summarise findings.

Install:  pip install pandas numpy matplotlib seaborn scipy
Run:      python task2_eda.py     (plots are saved in ./eda_plots)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

CSV_PATH = None            # e.g. "books_dataset.csv"; None = use Titanic
OUT_DIR = "eda_plots"
os.makedirs(OUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, name), dpi=150)
    plt.close()


# ---------------------------------------------------------------- 1. QUESTIONS
QUESTIONS = """
BUSINESS / RESEARCH QUESTIONS
 1. What is the overall survival rate?
 2. Did gender affect survival?
 3. Did passenger class (wealth) affect survival?
 4. Did age matter? Were children prioritised?
 5. Does the fare paid relate to survival?
 6. Are there missing values or outliers that could bias the analysis?
"""
print(QUESTIONS)

# ---------------------------------------------------------------- 2. LOAD DATA
df = pd.read_csv(CSV_PATH) if CSV_PATH else sns.load_dataset("titanic")

# ---------------------------------------------------------------- 3. STRUCTURE
print("=" * 70, "\nDATA STRUCTURE\n" + "=" * 70)
print("Shape:", df.shape)
print("\nColumns & data types:\n", df.dtypes)
print("\nFirst rows:\n", df.head())
print("\nNumeric summary:\n", df.describe().T)
print("\nCategorical summary:\n", df.describe(include=["object", "category"]).T)

# ---------------------------------------------------------------- 4. DATA QUALITY
print("=" * 70, "\nDATA QUALITY CHECKS\n" + "=" * 70)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
print(pd.DataFrame({"missing": missing, "percent": missing_pct})[missing > 0])
print("\nDuplicate rows:", df.duplicated().sum())

plt.figure(figsize=(10, 5))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis")
plt.title("Missing Values Map (yellow = missing)")
save("01_missing_values.png")

# Simple cleaning
clean = df.copy()
if "age" in clean.columns:
    clean["age"] = clean["age"].fillna(clean["age"].median())
if "embarked" in clean.columns:
    clean["embarked"] = clean["embarked"].fillna(clean["embarked"].mode()[0])
clean = clean.drop(columns=[c for c in ["deck"] if c in clean.columns])  # ~77% missing

# Outlier detection with the IQR rule
print("\nOUTLIERS (IQR method)")
for col in clean.select_dtypes(include=np.number).columns:
    q1, q3 = clean[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    n_out = ((clean[col] < q1 - 1.5 * iqr) | (clean[col] > q3 + 1.5 * iqr)).sum()
    if n_out:
        print(f"  {col:12s}: {n_out} outliers")

# ---------------------------------------------------------------- 5. UNIVARIATE
num_cols = clean.select_dtypes(include=np.number).columns.tolist()
clean[num_cols].hist(figsize=(12, 8), bins=25, edgecolor="black")
plt.suptitle("Distribution of Numeric Variables")
save("02_numeric_distributions.png")

plt.figure(figsize=(6, 4))
sns.countplot(data=clean, x="survived", palette="Set2")
plt.title("Survival Count (0 = No, 1 = Yes)")
save("03_survival_count.png")

# ---------------------------------------------------------------- 6. PATTERNS
print("=" * 70, "\nPATTERNS\n" + "=" * 70)
print(f"Overall survival rate: {clean['survived'].mean():.2%}")
print("\nSurvival by sex:\n", clean.groupby("sex")["survived"].mean().round(3))
print("\nSurvival by class:\n", clean.groupby("pclass")["survived"].mean().round(3))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.barplot(data=clean, x="sex", y="survived", palette="pastel", ax=ax[0])
ax[0].set_title("Survival Rate by Sex")
sns.barplot(data=clean, x="pclass", y="survived", palette="pastel", ax=ax[1])
ax[1].set_title("Survival Rate by Class")
save("04_survival_sex_class.png")

plt.figure(figsize=(8, 5))
sns.histplot(data=clean, x="age", hue="survived", kde=True, bins=30, multiple="stack")
plt.title("Age Distribution by Survival")
save("05_age_survival.png")

plt.figure(figsize=(8, 5))
sns.boxplot(data=clean, x="pclass", y="fare")
plt.yscale("log")
plt.title("Fare by Class (log scale) - note the outliers")
save("06_fare_boxplot.png")

plt.figure(figsize=(8, 6))
sns.heatmap(clean[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
save("07_correlation.png")

# ---------------------------------------------------------------- 7. HYPOTHESIS TESTS
print("=" * 70, "\nHYPOTHESIS TESTS (alpha = 0.05)\n" + "=" * 70)

# H1: Sex and survival are independent (Chi-square)
ct = pd.crosstab(clean["sex"], clean["survived"])
chi2, p, _, _ = stats.chi2_contingency(ct)
print(f"H1 Sex vs Survival      : chi2={chi2:.2f}, p={p:.2e} -> "
      f"{'REJECT' if p < 0.05 else 'FAIL TO REJECT'} independence")

# H2: Class and survival are independent (Chi-square)
ct = pd.crosstab(clean["pclass"], clean["survived"])
chi2, p, _, _ = stats.chi2_contingency(ct)
print(f"H2 Class vs Survival    : chi2={chi2:.2f}, p={p:.2e} -> "
      f"{'REJECT' if p < 0.05 else 'FAIL TO REJECT'} independence")

# H3: Mean age of survivors == mean age of non-survivors (Welch t-test)
a = clean.loc[clean.survived == 1, "age"]
b = clean.loc[clean.survived == 0, "age"]
t, p = stats.ttest_ind(a, b, equal_var=False)
print(f"H3 Age (surv vs died)   : t={t:.2f}, p={p:.4f} -> "
      f"{'REJECT' if p < 0.05 else 'FAIL TO REJECT'} equal means")

# H4: Survivors paid higher fares (Mann-Whitney, since fare is skewed)
u, p = stats.mannwhitneyu(clean.loc[clean.survived == 1, "fare"],
                          clean.loc[clean.survived == 0, "fare"], alternative="greater")
print(f"H4 Fare (surv > died)   : U={u:.0f}, p={p:.2e} -> "
      f"{'REJECT' if p < 0.05 else 'FAIL TO REJECT'} no difference")

# ---------------------------------------------------------------- 8. SUMMARY
print("\n" + "=" * 70 + "\nKEY FINDINGS\n" + "=" * 70)
print("""
 * About 38% of passengers survived.
 * Women survived at a much higher rate than men (significant).
 * Survival dropped sharply from 1st to 3rd class (significant).
 * Survivors were slightly younger; children had better odds.
 * Fare is heavily right-skewed with extreme outliers (consider log transform).
 * 'age' had ~20% missing (imputed with median); 'deck' ~77% missing (dropped).
 * NEXT STEPS: feature engineering (family size, title) and a predictive model.
""")
print(f"Plots saved in: ./{OUT_DIR}/")
