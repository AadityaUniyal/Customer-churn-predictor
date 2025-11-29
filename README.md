# Customer Churn Predictor 📊

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/flask-3.0-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An interactive web application built with Python and Flask/Streamlit to predict customer churn. This project demonstrates a complete data science workflow, including data preprocessing, training multiple machine learning models like XGBoost, and providing real-time, explainable predictions using SHAP.

## 🎯 Features

- **Interactive Web Interface**: Built with Streamlit for easy interaction
- **Multiple ML Models**: Compare Logistic Regression, Random Forest, Gradient Boosting, and XGBoost
- **Synthetic Data Generation**: Generate realistic customer data with proper correlations
- **Real-time Predictions**: Predict churn probability for individual customers
- **Model Explainability**: SHAP values to understand feature contributions
- **Business Insights**: Customer segmentation, lifetime value calculation, and retention strategies
- **Comprehensive Visualizations**: Interactive charts using Plotly

## 🚀 Quick Start

### Two Frontend Options!

**Option 1: Modern Flask Web App (Recommended)** 🎨
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run Flask app
python flask_app.py
```
Opens at: **http://localhost:5000**

**Features:**
- Beautiful modern UI with Bootstrap 5
- Interactive Chart.js visualizations
- Smooth animations and transitions
- Production-ready design
- RESTful API endpoints

**Option 2: Streamlit App** ⚡
```bash
# Run Streamlit app
streamlit run app.py
```
Opens at: **http://localhost:8501**

**Features:**
- Quick prototyping
- Built-in widgets
- Automatic reactivity
- Minimal code

### Verify Installation
```bash
python test_app.py
```

### First Time Using the App?

1. **Generate Data**: Click "Generate Sample Data" in the sidebar
2. **Train Models**: Navigate to "Train Models" page and click the button
3. **Make Predictions**: Go to "Make Predictions" and enter customer info
4. **Explore**: Check out visualizations and insights!

📖 **Need more help?** See [QUICK_START.md](QUICK_START.md) for a 5-minute walkthrough

## 📁 Project Structure

```
Customer-churn-predictor/
├── app.py                  # Main Streamlit application
├── data_generator.py       # Synthetic data generation module
├── model_trainer.py        # Standalone model training script
├── utils.py               # Utility functions (risk factors, strategies)
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── run.bat               # Windows run script
├── .streamlit/
│   └── config.toml       # Streamlit configuration
└── README.md             # This file
```

## 🎓 Key Concepts Demonstrated

### 1. Data Preprocessing
- Label encoding for categorical variables
- Feature scaling with StandardScaler
- Handling missing values
- Train-test split with stratification

### 2. Machine Learning Models
- **Logistic Regression**: Baseline linear model
- **Random Forest**: Ensemble of decision trees
- **Gradient Boosting**: Sequential ensemble method
- **XGBoost**: Optimized gradient boosting

### 3. Model Evaluation
- Accuracy, Precision, Recall, F1 Score
- ROC AUC Score
- Confusion Matrix
- Cross-validation

### 4. Feature Engineering
- Realistic feature correlations
- Business logic-based churn probability
- Customer segmentation
- Lifetime value calculation

### 5. Model Interpretability
- SHAP (SHapley Additive exPlanations)
- Feature importance visualization
- Individual prediction explanations

## 📊 Using the Application

### 1. Home Page
- Overview of the application
- Introduction to customer churn prediction

### 2. Data Overview
- Generate synthetic data or upload CSV
- View dataset statistics
- Visualize churn distribution
- Analyze churn by different features

### 3. Train Models
- Configure training parameters
- Train multiple models simultaneously
- Compare model performance
- View feature importance

### 4. Make Predictions
- Enter customer information
- Get churn probability prediction
- View customer segment and lifetime value
- Identify risk factors
- Get personalized retention strategies

### 5. Model Insights
- Generate SHAP explanations
- Understand feature contributions
- Visualize model decision-making

## 🔧 Standalone Model Training

You can train models independently using the `model_trainer.py` script:

```bash
python model_trainer.py
```

This will:
- Generate synthetic data
- Train all models
- Evaluate performance
- Save trained models to `models/` directory

## 📈 Sample Data Format

If uploading your own CSV, ensure it has these columns:

```
CustomerID, Age, Gender, Tenure, MonthlyCharges, TotalCharges,
Contract, InternetService, OnlineSecurity, TechSupport,
PaymentMethod, PaperlessBilling, NumProducts, SupportTickets, Churn
```

## 🎨 Customization

### Modify Model Parameters
Edit `config.py` to adjust:
- Model hyperparameters
- Training configuration
- Business rule thresholds
- UI settings

### Add New Features
1. Update `data_generator.py` to include new features
2. Add feature descriptions in `utils.py`
3. Update preprocessing in `app.py`

### Change Visualizations
Modify Plotly charts in `app.py` to customize:
- Color schemes
- Chart types
- Layout options

## 🧪 Key Python Concepts Used

1. **Object-Oriented Programming**: Classes for data generation and model training
2. **Data Manipulation**: Pandas DataFrames, NumPy arrays
3. **Machine Learning**: Scikit-learn, XGBoost
4. **Data Visualization**: Plotly, Matplotlib
5. **Web Development**: Streamlit
6. **File I/O**: Pickle for model serialization
7. **Statistical Analysis**: Probability distributions, correlations
8. **Functional Programming**: Utility functions, decorators
9. **Configuration Management**: Centralized config file
10. **Session State Management**: Streamlit session state

## 📚 Learning Resources

- **Scikit-learn**: https://scikit-learn.org/
- **XGBoost**: https://xgboost.readthedocs.io/
- **Streamlit**: https://docs.streamlit.io/
- **SHAP**: https://shap.readthedocs.io/
- **Plotly**: https://plotly.com/python/

## 🤝 Contributing

Feel free to fork this project and add your own features:
- Additional ML models (Neural Networks, SVM)
- More sophisticated feature engineering
- A/B testing simulation
- Time-series analysis
- Customer clustering

## 📝 License

This project is for educational purposes.

## 🎯 Future Enhancements

- [ ] Add deep learning models (Neural Networks)
- [ ] Implement hyperparameter tuning (GridSearch, RandomSearch)
- [ ] Add model versioning and experiment tracking
- [ ] Create API endpoints for predictions
- [ ] Add batch prediction capability
- [ ] Implement automated retraining pipeline
- [ ] Add more business metrics (CAC, LTV/CAC ratio)
- [ ] Create customer cohort analysis

## 💡 Tips

1. Start with the generated sample data to understand the workflow
2. Train models before making predictions
3. Experiment with different model parameters in `config.py`
4. Use SHAP insights to understand model decisions
5. Try different customer profiles to see how predictions change

---

## 📂 Documentation

- `README.md` - This file, project overview
- `INSTALL.md` - Installation guide
- `FLASK_GUIDE.md` - Flask frontend documentation

## 🎯 What You Get

✅ **Complete Application**: Working web app with 5 interactive pages  
✅ **4 ML Models**: Logistic Regression, Random Forest, Gradient Boosting, XGBoost  
✅ **Realistic Data**: Synthetic customer data with proper correlations  
✅ **Business Insights**: Risk factors, retention strategies, customer segments  
✅ **Model Explanations**: SHAP values for interpretability  
✅ **Comprehensive Docs**: 7 documentation files covering everything  
✅ **Testing Suite**: Automated tests to verify installation  
✅ **Production Ready**: Error handling, validation, configuration  

## 🏆 Perfect For

- 🎓 **Learning**: Complete ML project from start to finish
- 💼 **Portfolio**: Showcase data science and Python skills
- 🏢 **Business**: Actual customer retention analysis
- 👨‍🏫 **Teaching**: Educational material for Python/ML courses
- 🚀 **Prototyping**: Quick POC for churn prediction systems

## ⚡ Performance

- Data generation: <1 second for 1000 customers
- Model training: 30-60 seconds for all 4 models
- Predictions: Instant (<1 second)
- Memory usage: <500 MB typical

## 🆘 Need Help?

1. **Installation issues**: See [INSTALL.md](INSTALL.md)
2. **Flask frontend**: Read [FLASK_GUIDE.md](FLASK_GUIDE.md)
3. **Errors**: Run `python test_app.py` for diagnostics

## 🎉 Success Checklist

- [ ] Installed dependencies
- [ ] Tests pass (`python test_app.py`)
- [ ] App runs (`streamlit run app.py`)
- [ ] Generated sample data
- [ ] Trained models (>80% accuracy)
- [ ] Made a prediction
- [ ] Understood the results

**All checked?** Congratulations! You're ready to predict churn! 🎯

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📞 Contact

- Create an issue for bug reports or feature requests
- Check [SECURITY.md](SECURITY.md) for security concerns

## 🙏 Acknowledgments

- Built with Flask, Streamlit, Scikit-learn, and XGBoost
- UI powered by Bootstrap 5 and Chart.js
- Visualizations with Plotly

---

**Version**: 1.0.0 | **Status**: ✅ Production Ready | **License**: MIT

Made with ❤️ for the data science community
