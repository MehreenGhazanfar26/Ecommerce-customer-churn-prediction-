# E-Commerce Customer Churn Prediction

An end-to-end machine learning project that predicts customer churn and generates churn probabilities for new customer records.

## Project Overview

Customer churn is an important business problem for e-commerce companies. The goal of this project is to identify customers who are likely to churn so that businesses can prioritize retention strategies.

This project goes beyond model training by building a reusable machine learning pipeline that can process new customer data and generate a client-ready prediction report.

## Business Objective

Build a machine learning solution that can:

* Predict whether a customer is likely to churn
* Estimate the probability of churn
* Identify the most predictive customer features
* Apply the same preprocessing and model logic to new customer data
* Generate a structured prediction report

## Machine Learning Workflow

```text
Raw Customer Data
        ↓
Data Understanding & Exploration
        ↓
Train / Test Split
        ↓
ColumnTransformer
        ↓
Missing Value Imputation
        ↓
One-Hot Encoding
        ↓
Random Forest Classifier
        ↓
Model Evaluation
        ↓
Feature Importance Analysis
        ↓
Reusable ML Pipeline
        ↓
New Customer Data
        ↓
Churn Prediction + Probability
        ↓
Client Prediction Report
```

## Dataset

The project uses an e-commerce customer dataset containing customer behavioral and service-related features.
The original dataset is hosted on Kaggle and is downloaded programmatically using kagglehub.

Kaggle dataset:
ankitverma2010/ecommerce-customer-churn-analysis-and-prediction

The target variable is:

```text
Churn
```

`CustomerID` is treated as an identifier and is not used as a predictive feature.

## Preprocessing

The final solution uses a Scikit-learn `Pipeline` and `ColumnTransformer`.

### Numerical features

Missing numerical values are handled using median imputation.

### Categorical features

Categorical missing values are handled using most-frequent imputation and categorical variables are converted using One-Hot Encoding.

The preprocessing and model are combined into one reusable Pipeline so that the same transformations are automatically applied during prediction.

## Model

**Random Forest Classifier**

The final model is integrated directly into the preprocessing Pipeline.

## Model Performance

Test-set performance:

| Metric          | Result |
| --------------- | -----: |
| Accuracy        | 97.34% |
| Churn Precision |   1.00 |
| Churn Recall    |   0.84 |
| Churn F1-Score  |   0.91 |

The model provides strong predictive performance on the held-out test set.

## Feature Importance

The most predictive features identified by the Random Forest include:

1. Tenure
2. CashbackAmount
3. Complain
4. WarehouseToHome
5. DaySinceLastOrder
6. NumberOfAddress
7. OrderAmountHikeFromlastYear
8. SatisfactionScore
9. NumberOfDeviceRegistered
10. OrderCount

`Tenure` was the strongest predictive feature in the final model.

> Feature importance indicates predictive importance in the model. It does not establish that a feature causes churn.

## Prediction on New Customer Data

The trained Pipeline can be applied directly to new customer records.

Example workflow:

```text
new_customers.csv
        ↓
Saved churn_pipeline.pkl
        ↓
Predicted_Churn
        ↓
Churn_Probability_%
        ↓
client_churn_predictions_report.csv
```

Example output:

| CustomerID | Predicted Churn | Churn Probability |
| ---------: | --------------: | ----------------: |
|      55481 |               0 |                3% |
|      55487 |               1 |               98% |
|      55493 |               1 |               95% |
|      55494 |               1 |               83% |

## Project Structure

```text
ecommerce-customer-churn-prediction/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── customer_churn_prediction.ipynb
│
├── data/
│   └── new_customers.csv
│
├── models/
│   └── churn_pipeline.pkl
│
├── outputs/
│   └── client_churn_predictions_report.csv
│
└── images/
    ├── feature_importance.png
    ├── model_performance.png
    └── prediction_report.png
```

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Joblib
* Jupyter / Google Colab

## How to Use

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Open the notebook

Open:

```text
notebooks/customer_churn_prediction.ipynb
```

### 3. Train the model

Run the training and evaluation sections.

### 4. Save the trained Pipeline

The trained Pipeline is saved as:

```text
models/churn_pipeline.pkl
```

### 5. Provide new customer data

Place the new customer CSV inside:

```text
data/
```

### 6. Generate predictions

Run the prediction section to create:

```text
outputs/client_churn_predictions_report.csv
```

## Portfolio Purpose

This project demonstrates an end-to-end machine learning workflow with emphasis on reusable preprocessing, model deployment preparation, and prediction on new customer data.

The architecture is designed to be adaptable to future client datasets and machine learning projects.

## Future Improvements

Potential future improvements include:

* Cross-validation
* Hyperparameter tuning
* Threshold optimization
* Additional business-focused evaluation metrics
* Model monitoring
* Deployment through a web application or API

## Author

**Mehreen — AI Innovator**
