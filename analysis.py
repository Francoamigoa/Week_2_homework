import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    recall_score,
)
import matplotlib.pyplot as plt


def load_heart():
    return pd.read_csv("Data/cleanned.csv")


heart = load_heart()

# %% [markdown]
# # Inspect the Data

# %% [markdown]
# First rows

# %%
print("Heart Data:")
print(heart.head())

# %% [markdown]
# Data types and summary statistics

# %%
heart.info()

# %%
# Numeric Variables
heart.describe()


# %%
cat_columns = ["sex", "cp", "restecg", "fbs", "exang", "num"]


def categorical_frequencies(df: pd.DataFrame, cat_cols: list[str]) -> None:
    for col in cat_cols:
        if col in df.columns:
            summary = pd.DataFrame(
                {
                    "Count": df[col].value_counts(dropna=False),
                    "Percentage": (
                        df[col].value_counts(normalize=True, dropna=False) * 100
                    ).round(2),
                }
            )
            print(f"\nFrequencies for {col}:")
            print(summary)


freqs = categorical_frequencies(heart, cat_columns)


# %% [markdown]
# Missing values and duplicates

# %%
# Check for missing values
print("Missing values:\n", heart.isnull().sum())

# %%
# Show duplicates
heart[heart.duplicated()]

# %% [markdown]
# There were no duplicates
#
# # Basic Filtering and Grouping
# Potential subsets
#

# %%
# Subsets that could be interesting to analize
old = heart[heart["age"] > 65]
males = heart[heart["sex"] == "male"]
females = heart[heart["sex"] == "female"]
asymptomatic = heart[heart["cp"] == "asymptomatic"]


# %% [markdown]
# Statistics by group

# %%
# by sex
heart.groupby("sex")[heart.select_dtypes(include="number").columns].agg(
    ["mean", "median", "count"]
)


# %%
# by Chest pain type
heart.groupby("cp")[heart.select_dtypes(include="number").columns].agg(
    ["mean", "median", "count"]
)

# %%
# by Target variable (Heart disease diagnosis)
heart.groupby("num")[heart.select_dtypes(include="number").columns].agg(
    ["mean", "median", "count"]
)

# %% [markdown]
# # Explore a Machine Learning Algorithm
# %%
# Model 1: Only Sex and age


def run_logistic_model(
    df, subset_variables, target="disease", test_size=0.2, random_state=2000
):
    X = df[subset_variables]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()

    sensitivity = recall_score(y_test, y_pred, zero_division=0)  # TP / (TP+FN)
    specificity = TN / (TN + FP) if (TN + FP) > 0 else float("nan")

    print("Variables included:", subset_variables)
    print("Accuracy:", round(acc, 3))
    print("Sensitivity:", round(sensitivity, 3))
    print("Specificity:", round(specificity, 3))
    return {
        "Variables": subset_variables,
        "Accuracy": round(acc, 3),
        "Sensitivity": round(sensitivity, 3),
        "Specificity": round(specificity, 3),
        "Confusion_matrix": cm,
        "Model": model,
        "Feature_names": X_train.columns.tolist(),
    }


heart["disease"] = (heart["num"] > 0).astype(int)


def dummies_model(heart):
    heart_d = pd.get_dummies(
        heart, columns=["sex", "cp", "restecg", "fbs", "exang"], drop_first=True
    )

    return heart_d


heart_d = dummies_model(heart)

subset_variables = ["age", "sex_Male"]
res = run_logistic_model(heart_d, subset_variables)

# %%

subset_variables = [
    "age",
    "trestbps",
    "chol",
    "thalch",
    "oldpeak",
    "sex_Male",
    "cp_non-anginal",
    "cp_typical angina",
    "restecg_normal",
    "restecg_st-t abnormality",
    "fbs_True",
    "exang_True",
]
res = run_logistic_model(heart_d, subset_variables)


# %%

# Boxplot de edad por sexo
plt.figure(figsize=(6, 6))
heart.boxplot(column="age", by="sex")
plt.title("Boxplot of Age by Sex")
plt.suptitle("")
plt.xlabel("Sex")
plt.ylabel("Age")
plt.show()

# %%

# Boxplot sex-severity
plt.figure(figsize=(6, 6))
heart.boxplot(column="age", by="num")
plt.title("Boxplot of Age by Disease severity")
plt.suptitle("")
plt.xlabel("Disease severity (num)")
plt.ylabel("Age")
plt.show()

name_map = {
    "age": "Age",
    "sex_Male": "Male (vs Female)",
    "cp_typical angina": "Chest pain: Typical angina",
    "cp_non-anginal": "Chest pain: Non-anginal",
    "cp_asymptomatic": "Chest pain: Asymptomatic",
    "restecg_normal": "Resting ECG: Normal",
    "restecg_st-t abnormality": "Resting ECG: ST-T abnormality",
    "fbs_True": "Fasting blood sugar >120",
    "exang_True": "Exercise-induced angina",
    "trestbps": "Resting blood pressure",
    "chol": "Cholesterol",
    "thalach": "Max heart rate (thalach)",
    "oldpeak": "ST depression (oldpeak)",
}


def feature_importance(result, name_map=None):
    model = result["Model"]
    features = result["Feature_names"]
    coefs = model.coef_.flatten()
    imp = pd.DataFrame(
        {
            "Feature": features,
            "Coefficient": coefs,
            "Abs_Coefficient": np.abs(coefs),
            "OR": np.exp(coefs),
        }
    ).sort_values("Abs_Coefficient", ascending=False)
    if name_map:
        imp["Feature"] = imp["Feature"].map(lambda x: name_map.get(x, x))
    return imp


imp_df = feature_importance(res, name_map)

top = imp_df.head(10).sort_values("OR", ascending=True)
plt.figure(figsize=(6, 4))
plt.barh(top["Feature"], top["OR"])
plt.axvline(1, color="k", linestyle="--")
plt.title("Top factors – Odds Ratios")
plt.xlabel("Odds Ratio")
plt.tight_layout()
plt.show()
