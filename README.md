# Week_2_homework (Part 3)
This repository contains my Mini-Assignment using  Heart Disease UCI Dataset (Kaggle)

Source https://www.kaggle.com/datasets/navjotkaushal/heart-disease-uci-dataset

This project uses **GitHub Actions** to automatically check code quality and run tests on every push and pull request.  

It builds the Docker image, runs the test suite (`pytest`), and checks formatting/linting. The badge below shows the latest status of the CI pipeline on the `main` branch:  

[![CI](https://github.com/Francoamigoa/Week_2_homework/actions/workflows/ci.yml/badge.svg)](https://github.com/Francoamigoa/Week_2_homework/actions/workflows/ci.yml)

__CI workflow success__

<img width="672" height="371" alt="image" src="https://github.com/user-attachments/assets/67549716-c13e-47e7-9a24-517e93c1e8c7" />

# Personal objective

I am a PhD student in Population Health Sciences, and my research topic is not related to heart disease. The purpose of this program is to conduct exploratory analyses of the dataset, understand the available variables, identify some crude associations between variables, and generate preliminary hypotheses that could later be confirmed with more robust study designs.

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
    <img width="578" height="445" alt="image" src="https://github.com/user-attachments/assets/afb6edd3-9240-4a63-b0e7-ea2bc2a1d601" />
 
   - Boxplot of age by disease severity
  <img width="578" height="445" alt="image" src="https://github.com/user-attachments/assets/bba20ffc-2f3f-43f7-a27e-79bae4a7e818" />



## Findings
- The dataset is clean, well-structured, with no missing values or duplicates.
- Men and women show different distributions disease prevalence and severity.  
- Asymptomatic chest pain appears to be associated with the presence of disease. 
- Logistic regression including all variables (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`, `sex`, `chest pain`, `resting ECG`, `fasting blood sugar`, `exercise-induced angina`)   achieved about 82% accuracy, with both sensitivity and specificity substantially higher than the simplified age- and sex-only model.
- __Top Factor associated with higher odds of heart disease__
  
<img width="590" height="390" alt="image" src="https://github.com/user-attachments/assets/832959fd-3272-4507-93c0-4a42db484d14" />

__The strongest predictors of heart disease in this dataset are male sex, exercise-induced angina, and greater ST depression, all linked to about __3× higher odds__.
Typical/non-anginal chest pain and higher maximum heart rate are associated with lower odds.
Other factors (fasting blood sugar >120, resting ECG categories) show more modest effects.__

- This analysis was exploratory and preliminary, so further work is needed to draw more robust conclusions.

## Repository structure
- Data/cleanned.csv -> dataset
- analysis.ipynb -> Jupyter Notebook
- analysis.py -> exported python script
- README.md -> documentation
- requirements.txt -> required Python libraries
- Dockerfile
- .dockerignore 
- test_analysis.py -> 3 test (2 unit test and 1 systems test)
- .flake8 -> flake8 configuration
- .github/workflows/ci.yml -> GitHub Actions workflow (Continuous Integration)

# How to Set Up and Run?
> Local

pip install -r requirements.txt

pytest -q                # expected: all tests pass

>Docker

docker build -t week2 .

docker run --rm week2              # runs: pytest -q (as defined in Dockerfile CMD)

# Tests 
test_analysis.py includes:

- Unit: test_load_heart_ok → validates CSV load and expected columns.
- Unit: test_run_logistic_model_basic → checks model metrics.
- System (end-to-end): test_end_to_end_pipeline → real pipeline (load → dummies → model → metrics) with sanity checks on Accuracy/Sensitivity/Specificity ∈ [0,1].

