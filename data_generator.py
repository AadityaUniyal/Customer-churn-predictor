import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class ChurnDataGenerator:
    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
    
    def generate_data(self, n_samples=1000, churn_rate=0.27):
        data = {}
        
        data['CustomerID'] = [f"CUST{str(i).zfill(6)}" for i in range(1, n_samples + 1)]
        data['Age'] = np.random.normal(45, 15, n_samples).clip(18, 80).astype(int)
        data['Gender'] = np.random.choice(['Male', 'Female'], n_samples, p=[0.51, 0.49])
        tenure_dist = np.concatenate([
            np.random.exponential(8, int(n_samples * 0.4)),
            np.random.uniform(12, 72, int(n_samples * 0.6))
        ])
        np.random.shuffle(tenure_dist)
        data['Tenure'] = tenure_dist[:n_samples].clip(0, 72).astype(int)
        contract_choices = []
        for tenure in data['Tenure']:
            if tenure < 6:
                contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], 
                                          p=[0.8, 0.15, 0.05])
            elif tenure < 24:
                contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], 
                                          p=[0.5, 0.35, 0.15])
            else:
                contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], 
                                          p=[0.3, 0.3, 0.4])
            contract_choices.append(contract)
        data['Contract'] = contract_choices
        data['InternetService'] = np.random.choice(
            ['DSL', 'Fiber optic', 'No'], 
            n_samples, 
            p=[0.35, 0.45, 0.20]
        )
        base_charges = []
        for internet in data['InternetService']:
            if internet == 'No':
                charge = np.random.uniform(20, 40)
            elif internet == 'DSL':
                charge = np.random.uniform(40, 70)
            else:
                charge = np.random.uniform(60, 120)
            base_charges.append(charge)
        data['MonthlyCharges'] = np.array(base_charges).round(2)
        data['TotalCharges'] = (
            data['Tenure'] * data['MonthlyCharges'] * 
            np.random.uniform(0.9, 1.1, n_samples)
        ).round(2)
        online_security = []
        tech_support = []
        for internet in data['InternetService']:
            if internet == 'No':
                online_security.append('No internet service')
                tech_support.append('No internet service')
            else:
                online_security.append(np.random.choice(['Yes', 'No'], p=[0.4, 0.6]))
                tech_support.append(np.random.choice(['Yes', 'No'], p=[0.4, 0.6]))
        
        data['OnlineSecurity'] = online_security
        data['TechSupport'] = tech_support
        data['PaymentMethod'] = np.random.choice(
            ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'],
            n_samples,
            p=[0.35, 0.20, 0.25, 0.20]
        )
        data['PaperlessBilling'] = np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4])
        data['NumProducts'] = np.random.choice([1, 2, 3, 4], n_samples, p=[0.3, 0.35, 0.25, 0.1])
        data['SupportTickets'] = np.random.poisson(2, n_samples).clip(0, 15)
        df = pd.DataFrame(data)
        df['Churn'] = self._generate_churn_labels(df, target_rate=churn_rate)
        
        return df
    
    def _generate_churn_labels(self, df, target_rate=0.27):
        churn_score = np.zeros(len(df))
        churn_score += (df['Contract'] == 'Month-to-month').astype(int) * 1.5
        churn_score += (df['Contract'] == 'One year').astype(int) * 0.5
        churn_score += (df['Tenure'] < 6).astype(int) * 1.2
        churn_score += ((df['Tenure'] >= 6) & (df['Tenure'] < 12)).astype(int) * 0.8
        churn_score += ((df['Tenure'] >= 12) & (df['Tenure'] < 24)).astype(int) * 0.3
        churn_score += (df['MonthlyCharges'] > 80).astype(int) * 0.7
        churn_score += ((df['MonthlyCharges'] > 60) & (df['MonthlyCharges'] <= 80)).astype(int) * 0.3
        churn_score += (df['SupportTickets'] > 5).astype(int) * 0.9
        churn_score += ((df['SupportTickets'] > 3) & (df['SupportTickets'] <= 5)).astype(int) * 0.4
        churn_score += (df['PaymentMethod'] == 'Electronic check').astype(int) * 0.4
        churn_score += (df['OnlineSecurity'] == 'No').astype(int) * 0.3
        churn_score += (df['TechSupport'] == 'No').astype(int) * 0.3
        churn_score += (
            (df['InternetService'] == 'Fiber optic') & 
            (df['MonthlyCharges'] > 90)
        ).astype(int) * 0.5
        churn_score += np.random.normal(0, 0.5, len(df))
        churn_prob = 1 / (1 + np.exp(-churn_score))
        threshold = np.percentile(churn_prob, (1 - target_rate) * 100)
        churn_labels = (churn_prob > threshold).astype(int)
        
        return churn_labels
    
    def generate_batch_data(self, n_batches=5, samples_per_batch=200):
        all_data = []
        
        for i in range(n_batches):
            batch = self.generate_data(n_samples=samples_per_batch)
            batch['Batch'] = i + 1
            batch['DataDate'] = (datetime.now() - timedelta(days=30 * (n_batches - i))).strftime('%Y-%m-%d')
            all_data.append(batch)
        
        return pd.concat(all_data, ignore_index=True)
    
    def add_noise(self, df, noise_level=0.05):
        df = df.copy()
        numerical_cols = ['Age', 'Tenure', 'MonthlyCharges', 'TotalCharges', 'SupportTickets']
        
        for col in numerical_cols:
            if col in df.columns:
                noise = np.random.normal(0, df[col].std() * noise_level, len(df))
                df[col] = (df[col] + noise).clip(0)
                
                if col in ['Age', 'Tenure', 'SupportTickets']:
                    df[col] = df[col].astype(int)
                else:
                    df[col] = df[col].round(2)
        
        return df
    
    def introduce_missing_values(self, df, missing_rate=0.02):
        df = df.copy()
        cols_to_affect = ['TotalCharges', 'SupportTickets']
        
        for col in cols_to_affect:
            if col in df.columns:
                mask = np.random.random(len(df)) < missing_rate
                df.loc[mask, col] = np.nan
        
        return df


def generate_sample_customers(n=5):
    generator = ChurnDataGenerator()
    return generator.generate_data(n_samples=n)
