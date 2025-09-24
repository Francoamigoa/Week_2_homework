# Week_2_homework (Part 3)
This repository contains my Mini-Assignment using  Heart Disease UCI Dataset (Kaggle)

Source https://www.kaggle.com/datasets/navjotkaushal/heart-disease-uci-dataset

This project uses **GitHub Actions** to automatically check code quality and run tests on every push and pull request.  

It builds the Docker image, runs the test suite (`pytest`), and checks formatting/linting. The badge below shows the latest status of the CI pipeline on the `main` branch:  

[![CI](https://github.com/Francoamigoa/Week_2_homework/actions/workflows/ci.yml/badge.svg)](https://github.com/Francoamigoa/Week_2_homework/actions/workflows/ci.yml)

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
- requirements.txt -> required Python libraries
- Dockerfile
- .dockerignore 
- test_analysis.py -> 3 test (2 unit test and 1 systems test)

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

