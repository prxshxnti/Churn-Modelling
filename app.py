import streamlit as st
import pandas as pd
import pickle
from churn_model import ChurnPrediction
from streamlit_shap import st_shap
import shap

st.title("📊 Telecom Customer Churn Prediction")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("telecom_customer_churn.csv")

df = load_data()

# Fill missing values
df['Offer'] = df['Offer'].fillna('None')
df['Avg Monthly Long Distance Charges'] = df['Avg Monthly Long Distance Charges'].fillna(0)
df['Multiple Lines'] = df['Multiple Lines'].fillna('No')
df['Internet Type'] = df['Internet Type'].fillna('None')
df['Avg Monthly GB Download'] = df['Avg Monthly GB Download'].fillna(0.0)

for col in [
    'Online Security', 'Online Backup', 'Device Protection Plan',
    'Premium Tech Support', 'Streaming TV', 'Streaming Movies',
    'Streaming Music', 'Unlimited Data'
]:
    df[col] = df[col].fillna('No')

# NEW customers 
new_customers_raw = df[df['Customer Status'] == 'Joined']
st.header("🔍 New Customers — Raw Data")
st.dataframe(new_customers_raw.head(10).reset_index(drop=True))

# Encode data
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

df = pd.get_dummies(df, columns=['Internet Type', 'Contract',
                                 'Payment Method', 'Offer'],
                    drop_first=True)

binary_cols = [
    'Married', 'Phone Service', 'Multiple Lines', 'Internet Service',
    'Online Security', 'Online Backup', 'Device Protection Plan',
    'Premium Tech Support', 'Streaming TV', 'Streaming Movies',
    'Streaming Music', 'Unlimited Data', 'Paperless Billing'
]

for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

new_customers_encoded = df[df['Customer Status'] == 'Joined'].drop(['Customer Status'], axis=1)

# Load model
@st.cache_resource
def load_saved_model():
    with open("rsf_model.pkl", "rb") as f:
        return pickle.load(f)

model_data = load_saved_model()
feature_columns = model_data["X_columns"]

first_customer = new_customers_encoded.iloc[[0]]
first_customer = first_customer.reindex(columns=feature_columns, fill_value=0)

st.header("👤 First New Customer — Encoded Features")
st.dataframe(first_customer.reset_index(drop=True))

# Predict churn prob
model_wrapper = ChurnPrediction("telecom_customer_churn.csv")
model_wrapper.model = model_data["model"]

churn_prob = model_wrapper.predict_single(first_customer, time_horizon=3)

st.header("📉 3-Month Churn Probability")
st.success(f"Predicted Probability: **{churn_prob * 100:.2f}%**")

# Load SHAP explainer
@st.cache_resource
def load_shap_explainer():
    with open("shap_explainer.pkl", "rb") as f:
        return pickle.load(f)

explainer = load_shap_explainer()

# Get SHAP values for first customer
shap_values = explainer.shap_values(first_customer)

# SHAP force plot
st.header("🔍 SHAP Force Plot — First Customer")
force_plot = shap.force_plot(
    explainer.expected_value,
    shap_values,
    first_customer
)
st_shap(force_plot)

# SHAP bar plot
st.header("🔍 SHAP Bar Plot — First Customer")
bar_plot = shap.summary_plot(
    shap_values,
    first_customer,
    plot_type="bar",
    show=False
)
st_shap(bar_plot)

#SHAP Waterfall plot

shap_value_single = shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=first_customer.iloc[0]
)

st.header("🔍 SHAP Waterfall Plot — First Customer")
st_shap(shap.plots.waterfall(shap_value_single))