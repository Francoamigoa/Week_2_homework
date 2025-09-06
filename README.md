# Week_2_homework
I'm Franco Amigo, 2nd year PhD Student in Population Health Sciences.

This repository contains my Week 2 Mini-Assignment using  Heart Disease UCI Dataset (Kaggle)

Source https://www.kaggle.com/datasets/navjotkaushal/heart-disease-uci-dataset

# How is the code structured?

1. **Import the dataset**  
   Loaded the CSV file.

2. **Inspect the data**  
   - Used `.head()`, `.info()`, `.describe()`  
   - Checked missing values and duplicates  
   - Looked at categorical frequencies with `value_counts()`

3. **Filtering & grouping**  
   - Subsets: >65 years old, male, female, asymptomatic.
   - Grouped by sex, chest pain type and  disease status to see descriptive statistics 
   
4. **Predictive modeling**  
   - Logistic Regression (scikit-learn)  
   - Goal: Predict disease (yes/no)
   - 2 models: different variables
   - Evaluated with accuracy, sensitivity, and specificity.

5. **Visualization**  
   - Boxplot of age distribution by sex  
   - Boxplot of age by disease severity  
   - Scatter/strip plot of age vs disease severity  


## Findings
- The dataset is clean, well-structured, with no missing values or duplicates.
- Men and women show different distributions disease prevalence and severity.  
- Asymptomatic chest pain appears to be associated with the presence of disease. 
- Logistic regression including all variables achieved ~82% accuracy, with higher sensitivity and specificity than the simplified model.
- This analysis was exploratory and preliminary, so further work is needed to draw more robust conclusions.

## Repository structure
- Data/cleanned.csv -> dataset
- analysis.ipynb -> Jupyter Notebook
- analysis.py -> exported python script
- README.md -> documentation
