# Capstone: Injury Risk Prediction (Berkeley ML/AI)

## Project Overview
This capstone project predicts short-term musculoskeletal injury risk using athlete training load, movement, and recovery signals. The goal is to provide an early warning system that can help guide training adjustments and reduce time-loss injuries.

## Research Question
How accurately can short-term musculoskeletal injury risk be predicted for athletes by analyzing training load, movement, and recovery signals?

## Dataset
The dataset contains **5,430 samples** and **31 features** representing multimodal athlete monitoring signals, including:
- **Physiological / recovery signals:** heart rate, SpO2, blood pressure, respiratory rate, skin temperature
- **Movement / biomechanics:** ground reaction force, impact force, gait symmetry, cadence, range of motion
- **Training context:** training duration, workload intensity, rest period, repetition count
- **Outcome label:** `injury_risk` (0 = no risk, 1 = risk)

## Data Access
The dataset file is **not included** in this repository.  
To run the notebook, download the dataset and upload it into your Google Colab session so it is available at:

`/content/sports_multimodal_data.csv`

## Exploratory Data Analysis (EDA)
Key EDA observations:
- Injury risk is rare (**~5% of the dataset**), making this a class-imbalanced classification problem.
- Feature distributions and boxplots show noticeable differences between injury-risk vs non-risk groups, especially in workload- and force-related signals.

## Baseline Model (Module 20.1)
A **Logistic Regression** baseline model was trained using `class_weight="balanced"` to account for class imbalance.

### Evaluation Metrics
Because injury cases are rare, accuracy alone can be misleading. The model was evaluated using:
- **ROC-AUC**: overall ranking/separation ability
- **PR-AUC**: performance on the positive injury-risk class under imbalance
- **Recall (injury-risk class)**: ability to detect injury-risk cases (important for early-warning use cases)

### Results
- **ROC-AUC:** 0.941  
- **PR-AUC:** 0.533  
- **Injury-risk recall:** 0.89 (48 out of 54 injury-risk cases detected)

The baseline model detects most injury-risk cases, which is desirable for early-warning scenarios. The main tradeoff is a higher number of false positives, which can be improved in later modeling through threshold tuning and additional algorithms.

## Top Predictors (Baseline Interpretability)
The strongest predictors in the Logistic Regression baseline included:
- EMG amplitude
- Ground reaction force
- Heart rate
- Previous injury history

## Notebook
- **Capstone 20.1 Initial Report & EDA:** `20_1_initial_report_eda.ipynb`
