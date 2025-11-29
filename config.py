# Model Configuration
MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5
}

# Model Hyperparameters
LOGISTIC_REGRESSION_PARAMS = {
    'random_state': 42,
    'max_iter': 1000,
    'solver': 'lbfgs'
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'random_state': 42,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2
}

GRADIENT_BOOSTING_PARAMS = {
    'n_estimators': 100,
    'random_state': 42,
    'learning_rate': 0.1,
    'max_depth': 5
}

XGBOOST_PARAMS = {
    'n_estimators': 100,
    'random_state': 42,
    'learning_rate': 0.1,
    'max_depth': 5,
    'eval_metric': 'logloss'
}

DATA_CONFIG = {
    'default_samples': 1000,
    'target_churn_rate': 0.27,
    'seed': 42
}

FEATURE_NAMES = [
    'Age', 'Gender', 'Tenure', 'MonthlyCharges', 'TotalCharges',
    'Contract', 'InternetService', 'OnlineSecurity', 'TechSupport',
    'PaymentMethod', 'PaperlessBilling', 'NumProducts', 'SupportTickets'
]

CATEGORICAL_FEATURES = [
    'Gender', 'Contract', 'InternetService', 'OnlineSecurity',
    'TechSupport', 'PaymentMethod', 'PaperlessBilling'
]

NUMERICAL_FEATURES = [
    'Age', 'Tenure', 'MonthlyCharges', 'TotalCharges',
    'NumProducts', 'SupportTickets'
]

CHURN_RISK_THRESHOLDS = {
    'low': 0.3,
    'medium': 0.5,
    'high': 0.7
}

TENURE_THRESHOLDS = {
    'new': 6,
    'established': 12,
    'loyal': 24
}

MONTHLY_CHARGES_THRESHOLDS = {
    'budget': 50,
    'standard': 70,
    'premium': 90
}

UI_CONFIG = {
    'page_title': 'Customer Churn Predictor',
    'page_icon': '📊',
    'layout': 'wide',
    'theme': {
        'primary_color': '#1f77b4',
        'secondary_color': '#ff7f0e'
    }
}

VIZ_CONFIG = {
    'color_scheme': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
    'chart_height': 500,
    'gauge_colors': {
        'low': 'lightgreen',
        'medium': 'yellow',
        'high': 'lightcoral'
    }
}
