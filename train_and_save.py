from churn_model import ChurnPrediction

if __name__ == "__main__":
    churn_model = ChurnPrediction("telecom_customer_churn.csv")
    churn_model.save_model("rsf_model.pkl")
    print("Model trained and saved as rsf_model.pkl")

    churn_model.predict_all_new_customers_and_save_explainer("shap_explainer.pkl")
    print("Surrogate fitted and explainer saved as shap_explainer.pkl")
