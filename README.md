Churn Prediction Pipeline — Overview
This project implements an end-to-end customer churn prediction pipeline for a telecom dataset. The goal is to estimate the probability that a customer will churn within a given time period and explain the key drivers behind it.

✅ Key Components
1️⃣ Data Preprocessing

Handles missing values in relevant columns (e.g., Offer, Internet Type, various services).

Drops irrelevant columns like customer IDs and location details.

Encodes categorical variables and converts binary responses (Yes/No) into 1/0 format.

Creates dummy variables for multi-category features like Internet Type and Payment Method.

2️⃣ Target and Features

The target is customer churn, treated as a time-to-event survival problem — tracking whether the customer churned and for how long they stayed.

The pipeline creates a survival object combining churn status and tenure.

3️⃣ Handling Imbalance

Uses a KNN-based survival oversampling strategy to handle the natural imbalance between churned and non-churned customers.

This generates synthetic examples for underrepresented churn events by interpolating between nearest neighbors.

4️⃣ Model Training

Fits a Random Survival Forest (RSF) on the balanced data.

The RSF can predict the survival probability of each customer over time.

5️⃣ Predictions for New Customers

Isolates new customers who just joined.

Predicts churn probability for each at a defined time horizon (e.g., 3 months).

6️⃣ Explainability

Trains a surrogate Random Forest Regressor on the predicted churn probabilities for new customers.

Fits a SHAP explainer on the surrogate to interpret feature importance and understand the drivers behind the predicted churn risks.

7️⃣ Outputs

The final trained RSF model is saved as a pickled file (.pkl).

The SHAP explainer for the surrogate model is also saved.

These files can be loaded later for serving predictions and explanations in applications like a Streamlit dashboard.

🎯 Purpose and Impact
This pipeline enables:

Reliable survival-based churn predictions.

Insights into which features most influence churn risk.

The ability to predict churn for new customers right after they join.

Transparent results with interpretable SHAP values, which help businesses target retention strategies more effectively.

