import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import plotly.graph_objects as go
import plotly.express as px
import shap
import pickle
import io
import matplotlib.pyplot as plt

# Import custom modules
from data_generator import ChurnDataGenerator
from utils import (
    calculate_churn_risk_factors, 
    generate_retention_strategies,
    get_customer_segment,
    calculate_customer_lifetime_value,
    format_currency,
    format_percentage
)

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}

def generate_sample_data(n_samples=1000):
    """Generate synthetic customer churn data using advanced generator"""
    generator = ChurnDataGenerator(seed=42)
    return generator.generate_data(n_samples=n_samples)

def preprocess_data(df):
    """Preprocess the data for model training"""
    df = df.copy()
    
    # Remove CustomerID
    if 'CustomerID' in df.columns:
        df = df.drop('CustomerID', axis=1)
    
    # Separate features and target
    if 'Churn' in df.columns:
        X = df.drop('Churn', axis=1)
        y = df['Churn']
    else:
        X = df
        y = None
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    return X, y, label_encoders

def train_models(X_train, y_train):
    """Train multiple classification models"""
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

def evaluate_models(models, X_test, y_test):
    """Evaluate all models and return metrics"""
    results = []
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'ROC AUC': roc_auc_score(y_test, y_pred_proba)
        })
    
    return pd.DataFrame(results)

# Main App
st.markdown('<p class="main-header">📊 Customer Churn Predictor</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["Home", "Data Overview", "Train Models", "Make Predictions", "Model Insights"])
    
    st.markdown("---")
    st.header("Data Options")
    
    if st.button("Generate Sample Data"):
        n_samples = st.number_input("Number of samples", min_value=100, max_value=5000, value=1000)
        st.session_state.data = generate_sample_data(n_samples)
        st.success(f"Generated {n_samples} samples!")
    
    uploaded_file = st.file_uploader("Or upload your CSV", type=['csv'])
    if uploaded_file is not None:
        st.session_state.data = pd.read_csv(uploaded_file)
        st.success("Data uploaded successfully!")

# Page: Home
if page == "Home":
    st.markdown("### Welcome to the Customer Churn Predictor! 🎯")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### What is Customer Churn?
        Customer churn refers to when customers stop doing business with a company. 
        Predicting churn helps businesses:
        - Identify at-risk customers
        - Take proactive retention actions
        - Reduce revenue loss
        - Improve customer satisfaction
        """)
    
    with col2:
        st.markdown("""
        #### Features of this App:
        - 📊 Generate or upload customer data
        - 🤖 Train multiple ML models
        - 🎯 Make real-time predictions
        - 📈 Visualize model performance
        - 🔍 Explain predictions with SHAP
        """)
    
    st.markdown("---")
    st.info("👈 Start by generating sample data or uploading your own CSV from the sidebar!")

# Page: Data Overview
elif page == "Data Overview":
    st.markdown('<p class="sub-header">📋 Data Overview</p>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("Please generate or upload data first!")
    else:
        df = st.session_state.data
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Customers", len(df))
        col2.metric("Churned", df['Churn'].sum())
        col3.metric("Retained", len(df) - df['Churn'].sum())
        col4.metric("Churn Rate", f"{df['Churn'].mean()*100:.1f}%")
        
        st.markdown("### Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)
        
        st.markdown("### Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Churn Distribution")
            fig = px.pie(df, names='Churn', title='Churn vs Retained',
                        labels={0: 'Retained', 1: 'Churned'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Churn by Contract Type")
            churn_contract = df.groupby('Contract')['Churn'].mean().reset_index()
            fig = px.bar(churn_contract, x='Contract', y='Churn',
                        title='Churn Rate by Contract Type')
            st.plotly_chart(fig, use_container_width=True)

# Page: Train Models
elif page == "Train Models":
    st.markdown('<p class="sub-header">🤖 Train Machine Learning Models</p>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("Please generate or upload data first!")
    else:
        df = st.session_state.data
        
        st.markdown("### Training Configuration")
        test_size = st.slider("Test set size (%)", 10, 40, 20) / 100
        
        if st.button("Train Models", type="primary"):
            with st.spinner("Training models... This may take a moment."):
                # Preprocess data
                X, y, label_encoders = preprocess_data(df)
                st.session_state.label_encoders = label_encoders
                st.session_state.feature_names = X.columns.tolist()
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                st.session_state.scaler = scaler
                
                # Train models
                models = train_models(X_train_scaled, y_train)
                st.session_state.models = models
                
                # Evaluate models
                results_df = evaluate_models(models, X_test_scaled, y_test)
                
                st.success("✅ Models trained successfully!")
                
                st.markdown("### Model Performance Comparison")
                st.dataframe(results_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']),
                           use_container_width=True)
                
                # Visualize performance
                fig = go.Figure()
                metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
                
                for metric in metrics:
                    fig.add_trace(go.Bar(
                        name=metric,
                        x=results_df['Model'],
                        y=results_df[metric],
                        text=results_df[metric].round(3),
                        textposition='auto',
                    ))
                
                fig.update_layout(
                    title='Model Performance Metrics',
                    xaxis_title='Model',
                    yaxis_title='Score',
                    barmode='group',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance for best model
                best_model_name = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
                best_model = models[best_model_name]
                
                if hasattr(best_model, 'feature_importances_'):
                    st.markdown(f"### Feature Importance ({best_model_name})")
                    importance_df = pd.DataFrame({
                        'Feature': st.session_state.feature_names,
                        'Importance': best_model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(importance_df.head(10), x='Importance', y='Feature',
                                orientation='h', title='Top 10 Important Features')
                    st.plotly_chart(fig, use_container_width=True)

# Page: Make Predictions
elif page == "Make Predictions":
    st.markdown('<p class="sub-header">🎯 Make Churn Predictions</p>', unsafe_allow_html=True)
    
    if not st.session_state.models:
        st.warning("Please train models first!")
    else:
        st.markdown("### Select Model")
        selected_model = st.selectbox("Choose a model", list(st.session_state.models.keys()))
        
        st.markdown("### Enter Customer Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            gender = st.selectbox("Gender", ['Male', 'Female'])
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0)
        
        with col2:
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=1500.0)
            contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
            internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
            online_security = st.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
        
        with col3:
            tech_support = st.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])
            payment_method = st.selectbox("Payment Method", 
                                         ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'])
            paperless_billing = st.selectbox("Paperless Billing", ['Yes', 'No'])
            num_products = st.number_input("Number of Products", min_value=1, max_value=10, value=2)
            support_tickets = st.number_input("Support Tickets", min_value=0, max_value=20, value=2)
        
        if st.button("Predict Churn", type="primary"):
            # Create input dataframe
            input_data = pd.DataFrame({
                'Age': [age],
                'Gender': [gender],
                'Tenure': [tenure],
                'MonthlyCharges': [monthly_charges],
                'TotalCharges': [total_charges],
                'Contract': [contract],
                'InternetService': [internet_service],
                'OnlineSecurity': [online_security],
                'TechSupport': [tech_support],
                'PaymentMethod': [payment_method],
                'PaperlessBilling': [paperless_billing],
                'NumProducts': [num_products],
                'SupportTickets': [support_tickets]
            })
            
            # Encode categorical variables
            for col, le in st.session_state.label_encoders.items():
                if col in input_data.columns:
                    input_data[col] = le.transform(input_data[col].astype(str))
            
            # Scale features
            input_scaled = st.session_state.scaler.transform(input_data)
            
            # Make prediction
            model = st.session_state.models[selected_model]
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            # Display results
            st.markdown("---")
            st.markdown("### Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 1:
                    st.error("⚠️ Customer is likely to CHURN")
                else:
                    st.success("✅ Customer is likely to STAY")
            
            with col2:
                st.metric("Churn Probability", f"{prediction_proba[1]*100:.1f}%")
            
            with col3:
                st.metric("Retention Probability", f"{prediction_proba[0]*100:.1f}%")
            
            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction_proba[1] * 100,
                title={'text': "Churn Risk Score"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred" if prediction_proba[1] > 0.5 else "green"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Customer insights
            st.markdown("### Customer Insights")
            
            customer_dict = {
                'Age': age,
                'Gender': gender,
                'Tenure': tenure,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges,
                'Contract': contract,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'TechSupport': tech_support,
                'PaymentMethod': payment_method,
                'PaperlessBilling': paperless_billing,
                'NumProducts': num_products,
                'SupportTickets': support_tickets
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                segment = get_customer_segment(customer_dict)
                st.info(f"**Customer Segment:** {segment}")
                
                clv = calculate_customer_lifetime_value(customer_dict)
                st.info(f"**Estimated Lifetime Value:** {format_currency(clv)}")
            
            with col2:
                risk_factors = calculate_churn_risk_factors(customer_dict)
                if risk_factors:
                    st.warning("**Risk Factors Identified:**")
                    for factor, details in risk_factors.items():
                        severity_emoji = "🔴" if details['severity'] == 'High' else "🟡" if details['severity'] == 'Medium' else "🟢"
                        st.write(f"{severity_emoji} **{factor}**: {details['message']}")
                else:
                    st.success("No significant risk factors identified")
            
            # Retention strategies
            if prediction == 1:
                st.markdown("### 💡 Recommended Retention Strategies")
                strategies = generate_retention_strategies(risk_factors, prediction_proba[1])
                for strategy in strategies:
                    st.write(f"- {strategy}")

# Page: Model Insights
elif page == "Model Insights":
    st.markdown('<p class="sub-header">🔍 Model Insights with SHAP</p>', unsafe_allow_html=True)
    
    if not st.session_state.models or st.session_state.data is None:
        st.warning("Please train models and have data available!")
    else:
        st.markdown("### SHAP (SHapley Additive exPlanations)")
        st.info("SHAP values explain how each feature contributes to the model's prediction.")
        
        selected_model = st.selectbox("Select Model for Analysis", list(st.session_state.models.keys()))
        
        if st.button("Generate SHAP Analysis"):
            with st.spinner("Generating SHAP explanations..."):
                # Prepare data
                df = st.session_state.data
                X, y, _ = preprocess_data(df)
                X_scaled = st.session_state.scaler.transform(X)
                
                # Get model
                model = st.session_state.models[selected_model]
                
                # Create SHAP explainer
                explainer = shap.Explainer(model, X_scaled[:100])  # Use subset for speed
                shap_values = explainer(X_scaled[:100])
                
                st.markdown("### SHAP Summary Plot")
                st.write("Shows the impact of each feature on model predictions")
                
                # SHAP summary plot
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, X.iloc[:100], feature_names=st.session_state.feature_names, show=False)
                st.pyplot(fig)
                
                st.success("✅ SHAP analysis complete!")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Customer Churn Predictor v1.0")
