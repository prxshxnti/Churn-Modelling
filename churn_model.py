import numpy as np
import pandas as pd
import pickle
import shap

from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor


class ChurnPrediction:
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.preprocess()
        self.encode()
        self.prepare_data()
        self.balance_data()
        self.split_data()
        self.train_model()
    
    def preprocess(self):
        self.data['Offer'] = self.data['Offer'].fillna('None')
        self.data['Avg Monthly Long Distance Charges'] = self.data['Avg Monthly Long Distance Charges'].fillna(0)
        self.data['Multiple Lines'] = self.data['Multiple Lines'].fillna('No')
        self.data['Internet Type'] = self.data['Internet Type'].fillna('None')
        self.data['Avg Monthly GB Download'] = self.data['Avg Monthly GB Download'].fillna(0.0)
        for col in ['Online Security', 'Online Backup', 'Device Protection Plan',
                    'Premium Tech Support', 'Streaming TV', 'Streaming Movies',
                    'Streaming Music', 'Unlimited Data']:
            self.data[col] = self.data[col].fillna('No')

        self.data.drop(columns=['Customer ID', 'City', 'Zip Code',
                                'Churn Category', 'Churn Reason',
                                'Total Revenue'], inplace=True)
    
    def encode(self):
        self.data['Gender'] = self.data['Gender'].map({'Male': 0, 'Female': 1})
        self.data = pd.get_dummies(self.data, columns=['Internet Type', 'Contract',
                                                       'Payment Method', 'Offer'],
                                   drop_first=True)
        binary_cols = ['Married', 'Phone Service', 'Multiple Lines', 'Internet Service',
                       'Online Security', 'Online Backup', 'Device Protection Plan',
                       'Premium Tech Support', 'Streaming TV', 'Streaming Movies',
                       'Streaming Music', 'Unlimited Data', 'Paperless Billing']
        for col in binary_cols:
            self.data[col] = self.data[col].map({'Yes': 1, 'No': 0})

        self.new_customers_data = self.data[self.data['Customer Status'] == 'Joined'].drop(columns='Customer Status')

        self.data = self.data[self.data['Customer Status'].isin(['Churned', 'Stayed'])]
        self.data['Churned'] = np.where(self.data['Customer Status'] == 'Churned', True, False)
        self.data.drop(columns='Customer Status', inplace=True)

    def prepare_data(self):
        self.X = self.data.drop(['Churned', 'Tenure in Months'], axis=1)
        self.y = Surv.from_arrays(event=self.data['Churned'], time=self.data['Tenure in Months'])
        self.X_columns = self.X.columns.tolist()

    def knn_survival_oversample(self, X, y, n_neighbors=5, target_event_ratio=0.5):
        np.random.seed(42)
        event_mask = y['event']
        X_event = X[event_mask]
        y_event = y[event_mask]

        curr_ratio = X_event.shape[0] / X.shape[0]
        if curr_ratio >= target_event_ratio:
            return X.copy(), y.copy()

        n_target_event = int(target_event_ratio * X.shape[0] / (1 - target_event_ratio))
        n_to_generate = n_target_event - X_event.shape[0]

        knn = NearestNeighbors(n_neighbors=n_neighbors)
        knn.fit(X_event)

        synthetic_X = []
        synthetic_time = []
        for _ in range(n_to_generate):
            idx = np.random.randint(0, X_event.shape[0])
            base_feat = X_event.iloc[[idx]]
            base_time = y_event['time'][idx]
            neighbors = knn.kneighbors(base_feat, return_distance=False)[0]
            neighbor_idx = np.random.choice(neighbors[1:])
            neighbor_feat = X_event.iloc[neighbor_idx]
            neighbor_time = y_event['time'][neighbor_idx]
            new_feat = (base_feat.values.flatten() + neighbor_feat.values.flatten()) / 2
            new_feat += np.random.normal(0, 0.01, size=new_feat.shape)
            synthetic_X.append(new_feat)
            synthetic_time.append((base_time + neighbor_time) / 2)

        X_synth = pd.DataFrame(synthetic_X, columns=X.columns)
        y_synth = np.ones(n_to_generate, dtype=bool)
        time_synth = np.array(synthetic_time)

        X_res = pd.concat([X, X_synth], ignore_index=True)
        y_res = Surv.from_arrays(
            event=np.concatenate([y['event'], y_synth]),
            time=np.concatenate([y['time'], time_synth])
        )
        return X_res, y_res

    def balance_data(self):
        self.X_res, self.y_res = self.knn_survival_oversample(self.X, self.y)

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_res, self.y_res, test_size=0.2, stratify=self.y_res['event'], random_state=42
        )

    def train_model(self):
        self.model = RandomSurvivalForest(
            n_estimators=200,
            min_samples_split=15,
            min_samples_leaf=10,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42
        )
        self.model.fit(self.X_train, self.y_train)

    def predict_single(self, input_df, time_horizon=3):
        surv_fn = self.model.predict_survival_function(input_df)[0]
        churn_prob = 1 - surv_fn(time_horizon)
        return churn_prob
    
    def save_model(self, filepath="rsf_model.pkl"):
        with open(filepath, "wb") as f:
            pickle.dump({
                "model": self.model,
                "X_columns": self.X.columns.tolist()
            }, f)

    @staticmethod
    def load_saved_model(filepath="rsf_model.pkl"):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        return data

    def predict_all_new_customers_and_save_explainer(self, explainer_path, time_horizon=3):
        surv_fns = self.model.predict_survival_function(self.new_customers_data[self.X_columns])
        churn_probs = [1 - fn(time_horizon) for fn in surv_fns]

        surrogate = RandomForestRegressor(n_estimators=100, random_state=42)
        surrogate.fit(self.new_customers_data[self.X_columns], churn_probs)

        explainer = shap.TreeExplainer(surrogate)
        with open(explainer_path, "wb") as f:
            pickle.dump(explainer, f)