import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle


def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def calculate_churn_risk_factors(customer_data):
    risk_factors = {}
    if customer_data.get('Contract') == 'Month-to-month':
        risk_factors['Contract Type'] = {'severity': 'High', 'message': 'Month-to-month contracts have higher churn rates'}
    
    tenure = customer_data.get('Tenure', 0)
    if tenure < 6:
        risk_factors['Tenure'] = {'severity': 'High', 'message': 'New customers (< 6 months) are at higher risk'}
    elif tenure < 12:
        risk_factors['Tenure'] = {'severity': 'Medium', 'message': 'Customers with < 1 year tenure need attention'}
    
    monthly_charges = customer_data.get('MonthlyCharges', 0)
    if monthly_charges > 80:
        risk_factors['Pricing'] = {'severity': 'Medium', 'message': 'High monthly charges may lead to churn'}
    
    support_tickets = customer_data.get('SupportTickets', 0)
    if support_tickets > 5:
        risk_factors['Support Issues'] = {'severity': 'High', 'message': 'Multiple support tickets indicate dissatisfaction'}
    elif support_tickets > 3:
        risk_factors['Support Issues'] = {'severity': 'Medium', 'message': 'Moderate support ticket volume'}
    
    if customer_data.get('PaymentMethod') == 'Electronic check':
        risk_factors['Payment Method'] = {'severity': 'Low', 'message': 'Electronic check users show slightly higher churn'}
    
    if customer_data.get('OnlineSecurity') == 'No':
        risk_factors['Services'] = {'severity': 'Low', 'message': 'Customers without security services may churn more'}
    
    return risk_factors


def generate_retention_strategies(risk_factors, churn_probability):
    strategies = []
    
    if churn_probability < 0.3:
        strategies.append("✅ Low risk customer - maintain current engagement level")
        return strategies
    
    if 'Contract Type' in risk_factors:
        strategies.append("📝 Offer incentive to upgrade to annual contract (e.g., 10% discount)")
    
    if 'Tenure' in risk_factors:
        strategies.append("🎁 Provide welcome bonus or loyalty rewards for new customers")
        strategies.append("📞 Schedule proactive check-in call within first 3 months")
    
    if 'Pricing' in risk_factors:
        strategies.append("💰 Review pricing plan - offer customized package or discount")
        strategies.append("📊 Show value comparison with competitors")
    
    if 'Support Issues' in risk_factors:
        strategies.append("🛠️ Assign dedicated account manager for personalized support")
        strategies.append("⚡ Fast-track resolution of outstanding issues")
        strategies.append("💬 Follow-up survey to ensure satisfaction")
    
    if 'Services' in risk_factors:
        strategies.append("🔒 Offer free trial of premium services (security, tech support)")
        strategies.append("📚 Educate customer on available service benefits")
    
    if churn_probability > 0.7:
        strategies.append("🚨 HIGH PRIORITY: Immediate intervention required")
        strategies.append("📧 Send personalized retention offer within 24 hours")
    
    return strategies


def get_customer_segment(customer_data):
    tenure = customer_data.get('Tenure', 0)
    monthly_charges = customer_data.get('MonthlyCharges', 0)
    contract = customer_data.get('Contract', '')
    
    if tenure < 12 and monthly_charges < 50:
        return "New & Budget-Conscious"
    elif tenure < 12 and monthly_charges >= 50:
        return "New & Premium"
    elif tenure >= 12 and contract == 'Two year':
        return "Loyal & Committed"
    elif tenure >= 12 and monthly_charges >= 70:
        return "Long-term Premium"
    elif tenure >= 12 and monthly_charges < 70:
        return "Long-term Standard"
    else:
        return "Standard Customer"


def calculate_customer_lifetime_value(customer_data):
    monthly_charges = customer_data.get('MonthlyCharges', 0)
    tenure = customer_data.get('Tenure', 0)
    contract = customer_data.get('Contract', '')
    if contract == 'Two year':
        expected_lifetime = max(24, tenure + 12)
    elif contract == 'One year':
        expected_lifetime = max(12, tenure + 6)
    else:
        expected_lifetime = max(6, tenure + 3)
    
    clv = monthly_charges * expected_lifetime
    return round(clv, 2)


def get_feature_descriptions():
    return {
        'Age': 'Customer age in years',
        'Gender': 'Customer gender (Male/Female)',
        'Tenure': 'Number of months the customer has been with the company',
        'MonthlyCharges': 'Amount charged to the customer monthly',
        'TotalCharges': 'Total amount charged to the customer',
        'Contract': 'Type of contract (Month-to-month, One year, Two year)',
        'InternetService': 'Type of internet service (DSL, Fiber optic, No)',
        'OnlineSecurity': 'Whether customer has online security service',
        'TechSupport': 'Whether customer has tech support service',
        'PaymentMethod': 'Payment method used by customer',
        'PaperlessBilling': 'Whether customer uses paperless billing',
        'NumProducts': 'Number of products/services subscribed',
        'SupportTickets': 'Number of support tickets raised'
    }


def validate_customer_data(data):
    required_fields = [
        'Age', 'Gender', 'Tenure', 'MonthlyCharges', 'TotalCharges',
        'Contract', 'InternetService', 'OnlineSecurity', 'TechSupport',
        'PaymentMethod', 'PaperlessBilling', 'NumProducts', 'SupportTickets'
    ]
    
    for field in required_fields:
        if field not in data or data[field] is None:
            return False, f"Missing required field: {field}"
    
    # Validate ranges
    if data['Age'] < 18 or data['Age'] > 120:
        return False, "Age must be between 18 and 120"
    
    if data['Tenure'] < 0:
        return False, "Tenure cannot be negative"
    
    if data['MonthlyCharges'] < 0:
        return False, "Monthly charges cannot be negative"
    
    if data['TotalCharges'] < 0:
        return False, "Total charges cannot be negative"
    
    return True, "Valid"


def format_currency(amount):
    return f"${amount:,.2f}"

def format_percentage(value):
    return f"{value * 100:.1f}%"
