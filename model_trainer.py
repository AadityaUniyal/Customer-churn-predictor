import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
import pickle
from datetime import datetime

from data_generator import ChurnDataGenerator
from config import *


class ChurnModelTrainer:
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        self.best_model = None
        self.best_model_name = None
        self.training_history = []
    
    def load_data(self, filepath=None, n_samples=1000):
        if filepath:
            self.data = pd.read_csv(filepath)
            print(f"Loaded {len(self.data)} records from {filepath}")
        else:
            generator = ChurnDataGenerator()
            self.data = generator.generate_data(n_samples=n_samples)
            print(f"Generated {n_samples} synthetic records")
        
        print(f"Churn rate: {self.data['Churn'].mean():.2%}")
        return self.data
    
    def preprocess_data(self, df=None):
        if df is None:
            df = self.data.copy()
        else:
            df = df.copy()
        
        if 'CustomerID' in df.columns:
            df = df.drop('CustomerID', axis=1)
        
        # Separate features and target
        X = df.drop('Churn', axis=1)
        y = df['Churn']
        
        self.feature_names = X.columns.tolist()
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=MODEL_CONFIG['random_state'],
            stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def train_all_models(self, X_train, y_train):
        print("\nTraining models...")
        
        models_config = {
            'Logistic Regression': LogisticRegression(**LOGISTIC_REGRESSION_PARAMS),
            'Random Forest': RandomForestClassifier(**RANDOM_FOREST_PARAMS),
            'Gradient Boosting': GradientBoostingClassifier(**GRADIENT_BOOSTING_PARAMS),
            'XGBoost': XGBClassifier(**XGBOOST_PARAMS)
        }
        
        for name, model in models_config.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            self.models[name] = model
            cv_scores = cross_val_score(model, X_train, y_train, 
                                       cv=MODEL_CONFIG['cv_folds'], 
                                       scoring='accuracy')
            print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        print("All models trained successfully!")
        return self.models
    
    def evaluate_models(self, X_test, y_test):
        print("\nEvaluating models...")
        
        results = []
        best_score = 0
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'ROC AUC': roc_auc
            })
            
            print(f"\n{name}:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1 Score:  {f1:.4f}")
            print(f"  ROC AUC:   {roc_auc:.4f}")
            if accuracy > best_score:
                best_score = accuracy
                self.best_model = model
                self.best_model_name = name
        
        results_df = pd.DataFrame(results)
        
        print(f"\nBest Model: {self.best_model_name} (Accuracy: {best_score:.4f})")
        
        return results_df
    
    def get_feature_importance(self, model_name=None):
        if model_name is None:
            model_name = self.best_model_name
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            return importance_df
        else:
            print(f"{model_name} does not have feature_importances_ attribute")
            return None
    
    def save_models(self, directory='models'):
        import os
        os.makedirs(directory, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        for name, model in self.models.items():
            filename = f"{directory}/{name.replace(' ', '_')}_{timestamp}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved {name} to {filename}")
        scaler_file = f"{directory}/scaler_{timestamp}.pkl"
        with open(scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Saved scaler to {scaler_file}")
        encoders_file = f"{directory}/label_encoders_{timestamp}.pkl"
        with open(encoders_file, 'wb') as f:
            pickle.dump(self.label_encoders, f)
        print(f"Saved label encoders to {encoders_file}")
        features_file = f"{directory}/feature_names_{timestamp}.pkl"
        with open(features_file, 'wb') as f:
            pickle.dump(self.feature_names, f)
        print(f"Saved feature names to {features_file}")
    
    def full_training_pipeline(self, filepath=None, n_samples=1000, test_size=0.2):
        print("="*60)
        print("CUSTOMER CHURN PREDICTION - MODEL TRAINING PIPELINE")
        print("="*60)
        self.load_data(filepath, n_samples)
        X, y = self.preprocess_data()
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_size)
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        self.train_all_models(X_train_scaled, y_train)
        results = self.evaluate_models(X_test_scaled, y_test)
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE (Top 10)")
        print("="*60)
        importance = self.get_feature_importance()
        if importance is not None:
            print(importance.head(10).to_string(index=False))

        print("\n" + "="*60)
        print("SAVING MODELS")
        print("="*60)
        self.save_models()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        
        return results


def main():
    """Main function to run training"""
    trainer = ChurnModelTrainer()
    
    results = trainer.full_training_pipeline(
        filepath=None,  
        n_samples=2000,  
        test_size=0.2
    )
    
    print("\nFinal Results:")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
