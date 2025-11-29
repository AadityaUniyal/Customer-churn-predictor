from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from functools import wraps
import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime
import os

from data_generator import ChurnDataGenerator
from model_trainer import ChurnModelTrainer
from utils import (
    calculate_churn_risk_factors,
    generate_retention_strategies,
    get_customer_segment,
    calculate_customer_lifetime_value,
    format_currency,
    format_percentage
)
from config import *

app = Flask(__name__)
app.secret_key = 'churn-predictor-secret-key-2024'

# User database (in production, use a real database)
USERS = {
    'user': {'password': 'user123', 'role': 'user', 'name': 'John Doe'},
    'admin': {'password': 'admin123', 'role': 'admin', 'name': 'Admin User'}
}

app_data = {
    'data': None,
    'models': {},
    'scaler': None,
    'label_encoders': {},
    'feature_names': None,
    'training_results': None
}

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Admin required decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        if session.get('role') != 'admin':
            flash('Admin access required!', 'danger')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username in USERS and USERS[username]['password'] == password:
            session['username'] = username
            session['role'] = USERS[username]['role']
            session['name'] = USERS[username]['name']
            flash(f'Welcome back, {USERS[username]["name"]}!', 'success')
            return redirect(url_for('intro'))
        else:
            flash('Invalid credentials. Try user/user123 or admin/admin123', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('login'))

@app.route('/intro')
@login_required
def intro():
    return render_template('intro.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/data')
@login_required
def data_page():
    return render_template('data.html')

@app.route('/train')
@login_required
def train_page():
    return render_template('train.html')

@app.route('/predict')
@login_required
def predict_page():
    return render_template('predict.html')

@app.route('/insights')
@login_required
def insights_page():
    return render_template('insights.html')

@app.route('/admin')
@admin_required
def admin_page():
    return render_template('admin.html')

@app.route('/api/generate-data', methods=['POST'])
def generate_data():
    try:
        n_samples = int(request.json.get('n_samples', 1000))
        
        generator = ChurnDataGenerator(seed=42)
        df = generator.generate_data(n_samples=n_samples)
        app_data['data'] = df
        stats = {
            'total_customers': len(df),
            'churned': int(df['Churn'].sum()),
            'retained': int(len(df) - df['Churn'].sum()),
            'churn_rate': float(df['Churn'].mean()),
            'avg_age': float(df['Age'].mean()),
            'avg_tenure': float(df['Tenure'].mean()),
            'avg_monthly_charges': float(df['MonthlyCharges'].mean())
        }
        
        churn_by_contract = df.groupby('Contract')['Churn'].agg(['mean', 'count']).to_dict()
        age_bins = [18, 30, 40, 50, 60, 80]
        age_labels = ['18-30', '31-40', '41-50', '51-60', '60+']
        df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
        age_dist = df['AgeGroup'].value_counts().to_dict()
        
        return jsonify({
            'success': True,
            'stats': stats,
            'churn_by_contract': churn_by_contract,
            'age_distribution': {str(k): int(v) for k, v in age_dist.items()},
            'sample_data': df.head(10).to_dict('records')
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/get-data-stats', methods=['GET'])
def get_data_stats():
    if app_data['data'] is None:
        return jsonify({'success': False, 'error': 'No data available'}), 400
    
    df = app_data['data']
    stats = {
        'total': len(df),
        'churned': int(df['Churn'].sum()),
        'retained': int(len(df) - df['Churn'].sum()),
        'churn_rate': float(df['Churn'].mean()),
        'churn_by_contract': df.groupby('Contract')['Churn'].mean().to_dict(),
        'churn_by_internet': df.groupby('InternetService')['Churn'].mean().to_dict(),
        'churn_by_payment': df.groupby('PaymentMethod')['Churn'].mean().to_dict(),
        'tenure_dist': df['Tenure'].describe().to_dict(),
        'charges_dist': df['MonthlyCharges'].describe().to_dict(),
        'feature_correlations': {
            'Tenure': float(df['Tenure'].corr(df['Churn'])),
            'MonthlyCharges': float(df['MonthlyCharges'].corr(df['Churn'])),
            'TotalCharges': float(df['TotalCharges'].corr(df['Churn'])),
            'SupportTickets': float(df['SupportTickets'].corr(df['Churn']))
        }
    }
    
    return jsonify({'success': True, 'stats': stats})


@app.route('/api/train-models', methods=['POST'])
def train_models():
    try:
        if app_data['data'] is None:
            return jsonify({'success': False, 'error': 'No data available. Generate data first.'}), 400
        
        test_size = float(request.json.get('test_size', 0.2))
        
        trainer = ChurnModelTrainer()
        trainer.data = app_data['data']
        X, y = trainer.preprocess_data()
        app_data['label_encoders'] = trainer.label_encoders
        app_data['feature_names'] = trainer.feature_names
        X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size)
        X_train_scaled, X_test_scaled = trainer.scale_features(X_train, X_test)
        app_data['scaler'] = trainer.scaler
        models = trainer.train_all_models(X_train_scaled, y_train)
        app_data['models'] = models
        results_df = trainer.evaluate_models(X_test_scaled, y_test)
        app_data['training_results'] = results_df
        best_model_name = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
        importance_df = trainer.get_feature_importance(best_model_name)
        
        results = {
            'success': True,
            'models': results_df.to_dict('records'),
            'best_model': best_model_name,
            'feature_importance': importance_df.to_dict('records') if importance_df is not None else None
        }
        
        return jsonify(results)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if not app_data['models']:
            return jsonify({'success': False, 'error': 'No trained models. Train models first.'}), 400
        
        data = request.json
        model_name = data.get('model', list(app_data['models'].keys())[0])
        input_data = pd.DataFrame({
            'Age': [int(data['age'])],
            'Gender': [data['gender']],
            'Tenure': [int(data['tenure'])],
            'MonthlyCharges': [float(data['monthly_charges'])],
            'TotalCharges': [float(data['total_charges'])],
            'Contract': [data['contract']],
            'InternetService': [data['internet_service']],
            'OnlineSecurity': [data['online_security']],
            'TechSupport': [data['tech_support']],
            'PaymentMethod': [data['payment_method']],
            'PaperlessBilling': [data['paperless_billing']],
            'NumProducts': [int(data['num_products'])],
            'SupportTickets': [int(data['support_tickets'])]
        })
        for col, le in app_data['label_encoders'].items():
            if col in input_data.columns:
                input_data[col] = le.transform(input_data[col].astype(str))
        
        input_scaled = app_data['scaler'].transform(input_data)
        model = app_data['models'][model_name]
        prediction = int(model.predict(input_scaled)[0])
        prediction_proba = model.predict_proba(input_scaled)[0].tolist()
        customer_dict = {
            'Age': int(data['age']),
            'Gender': data['gender'],
            'Tenure': int(data['tenure']),
            'MonthlyCharges': float(data['monthly_charges']),
            'TotalCharges': float(data['total_charges']),
            'Contract': data['contract'],
            'InternetService': data['internet_service'],
            'OnlineSecurity': data['online_security'],
            'TechSupport': data['tech_support'],
            'PaymentMethod': data['payment_method'],
            'PaperlessBilling': data['paperless_billing'],
            'NumProducts': int(data['num_products']),
            'SupportTickets': int(data['support_tickets'])
        }
        
        segment = get_customer_segment(customer_dict)
        clv = calculate_customer_lifetime_value(customer_dict)
        risk_factors = calculate_churn_risk_factors(customer_dict)
        strategies = generate_retention_strategies(risk_factors, prediction_proba[1])
        
        result = {
            'success': True,
            'prediction': prediction,
            'churn_probability': prediction_proba[1],
            'retention_probability': prediction_proba[0],
            'segment': segment,
            'lifetime_value': clv,
            'risk_factors': risk_factors,
            'retention_strategies': strategies,
            'model_used': model_name
        }
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/get-models', methods=['GET'])
def get_models():
    if not app_data['models']:
        return jsonify({'success': False, 'error': 'No trained models'}), 400
    
    models_list = list(app_data['models'].keys())
    results = app_data['training_results'].to_dict('records') if app_data['training_results'] is not None else []
    
    return jsonify({
        'success': True,
        'models': models_list,
        'results': results
    })


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    try:
        if app_data['data'] is None:
            return jsonify({'success': False, 'error': 'No data available'}), 400
        
        if not app_data['models']:
            return jsonify({'success': False, 'error': 'No trained models'}), 400
        
        model_name = request.json.get('model', list(app_data['models'].keys())[0])
        n_samples = int(request.json.get('n_samples', 100))
        df = app_data['data'].head(n_samples).copy()
        X = df.drop(['Churn', 'CustomerID'], axis=1, errors='ignore')
        y_true = df['Churn'].values if 'Churn' in df.columns else None
        for col, le in app_data['label_encoders'].items():
            if col in X.columns:
                X[col] = le.transform(X[col].astype(str))
        X_scaled = app_data['scaler'].transform(X)
        model = app_data['models'][model_name]
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]
        results = []
        for i in range(len(predictions)):
            results.append({
                'index': i,
                'prediction': int(predictions[i]),
                'probability': float(probabilities[i]),
                'actual': int(y_true[i]) if y_true is not None else None
            })
        accuracy = float((predictions == y_true).mean()) if y_true is not None else None
        
        return jsonify({
            'success': True,
            'results': results,
            'accuracy': accuracy,
            'total_predicted_churn': int(predictions.sum()),
            'avg_churn_probability': float(probabilities.mean())
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
