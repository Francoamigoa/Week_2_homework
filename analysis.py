# %% [markdown]
# # Import the Dataset

# %%
import pandas as pd 
# Load Heart Disease UCI Dataset
# Source https://www.kaggle.com/datasets/navjotkaushal/heart-disease-uci-dataset
heart = pd.read_csv("Data/cleanned.csv") 

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
#Numeric Variables
heart.describe()


# %%
cat_cols = ["sex", "cp", "restecg", "fbs", "exang", "num"]

for col in cat_cols:
    if col in heart.columns:
        summary = pd.DataFrame({
            "Count": heart[col].value_counts(dropna=False),
            "Percentage": (heart[col].value_counts(normalize=True, dropna=False) * 100).round(2)
        })
        print(f"\nFrequencies for {col}:")
        print(summary)


# %% [markdown]
# Missing values and duplicates

# %%
# Check for missing values
print("Missing values:\n", heart.isnull().sum())

# %%
#Show duplicates
heart[heart.duplicated()]

# %% [markdown]
# There were no duplicates
# 
# # Basic Filtering and Grouping
# Potential subsets
# 

# %%
#Subsets that could be interesting to analize
old = heart[heart["age"] > 65]
males = heart[heart["sex"] == "male"]
females = heart[heart["sex"] == "female"]
asymptomatic = heart[heart["cp"] == "asymptomatic"]


# %% [markdown]
# Statistics by group

# %%
#by sex
heart.groupby("sex")[heart.select_dtypes(include="number").columns].agg(["mean", "median", "count"])



# %%
#by Chest pain type
heart.groupby("cp")[heart.select_dtypes(include="number").columns].agg(["mean", "median", "count"])

# %%
#by Target variable (Heart disease diagnosis)
heart.groupby("num")[heart.select_dtypes(include="number").columns].agg(["mean", "median", "count"])

# %% [markdown]
# # Explore a Machine Learning Algorithm
# For simplicity, the variable num will be recategorized into 0 and 1, with 0 representing no disease and 1 indicating the presence of heart disease. 

# %%
#Model 1: Only Sex and age
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
heart["disease"] = (heart["num"] > 0).astype(int)
heart_d = pd.get_dummies(
    heart,
    columns=["sex", "cp", "restecg", "fbs", "exang"],
    drop_first=True
)
heart_d
subset_variables= ["age", "sex_Male"]
X = heart_d[subset_variables]
y = heart_d["disease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2000, stratify=y
)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
sensitivity = recall_score(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()
specificity = TN / (TN + FP)
print("Variables included:", list(X.columns))
print("Accuracy:", round(acc, 3))
print("Sensitivity: ", round(sensitivity, 3))
print("Specificity: ", round(specificity, 3))

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
heart["disease"] = (heart["num"] > 0).astype(int)
heart_d = pd.get_dummies(
    heart,
    columns=["sex", "cp", "restecg", "fbs", "exang"],
    drop_first=True
)

X = heart_d.drop(columns=["num", "disease"])
y = heart_d["disease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2000, stratify=y
)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
sensitivity = recall_score(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()
specificity = TN / (TN + FP)
print("Variables included:", list(X.columns))
print("Accuracy:", round(acc, 3))
print("Sensitivity: ", round(sensitivity, 3))
print("Specificity: ", round(specificity, 3))

# %% [markdown]
# The second model, which incorporates more variables, shows improved accuracy, sensitivity, and specificity. Further refinement could involve applying more advanced methods for variable selection or exploring alternative ML algorithms.
# 
# # Visualization
# 
# Below are two examples of visualization. The first shows the distribution of age by sex. The second shows age by severity.

# %%
import matplotlib.pyplot as plt

# Boxplot de edad por sexo
plt.figure(figsize=(6,6))
heart.boxplot(column="age", by="sex")
plt.title("Boxplot of Age by Sex")
plt.suptitle("") 
plt.xlabel("Sex")
plt.ylabel("Age")
plt.show()

# %%
import matplotlib.pyplot as plt

# Boxplot sex-severity
plt.figure(figsize=(6,6))
heart.boxplot(column="age", by="num")
plt.title("Boxplot of Age by Disease severity")
plt.suptitle("") 
plt.xlabel("Disease severity (num)")
plt.ylabel("Age")
plt.show()


