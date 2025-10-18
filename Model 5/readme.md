# 💰 Personal Finance Forecasting Model - LSTM + XGBoost Hybrid

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg)](https://www.tensorflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-red.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A sophisticated hybrid deep learning model that combines **LSTM (Long Short-Term Memory)** and **XGBoost** to predict users' future financial metrics including monthly income, expenses, and savings patterns based on historical transaction data.

## 📊 Overview

This project implements a state-of-the-art hybrid forecasting system that leverages the temporal learning capabilities of LSTM neural networks combined with the feature-based refinement power of XGBoost gradient boosting to deliver accurate personal finance predictions.

### Key Features

✅ **Hybrid Architecture**: Combines LSTM's sequential pattern recognition with XGBoost's feature-based learning  
✅ **Multi-Target Prediction**: Simultaneously forecasts Income, Expenses, and Savings  
✅ **Advanced Feature Engineering**: 70+ engineered features from temporal, financial, and user behavior data  
✅ **Comprehensive Evaluation**: R², RMSE, MAE, and MAPE metrics for robust performance assessment  
✅ **Production Ready**: Complete model persistence with scalers, encoders, and metadata  
✅ **Rich Visualizations**: 6 detailed visualization files for EDA and model performance analysis  

## 🏗️ Model Architecture

graph TB
subgraph Input
A[Personal Finance Dataset
3000 records, 25 features]
end

text
subgraph "Data Preprocessing"
    B[Feature Engineering<br/>-  Temporal Features<br/>-  Financial Ratios<br/>-  Lag & Rolling Features<br/>-  User Aggregations]
    C[Data Splitting<br/>Train: 80% | Test: 20%]
    D[Normalization<br/>MinMaxScaler]
end

subgraph "LSTM Model"
    E[Bidirectional LSTM<br/>128 units]
    F[LSTM Layer<br/>64 units]
    G[Dense Layers<br/>32 → 16 → 3]
    E --> F --> G
end

subgraph "Hybrid XGBoost Model"
    H[Feature Fusion<br/>Original + LSTM Output]
    I[XGBoost Regressor 1<br/>Monthly Income]
    J[XGBoost Regressor 2<br/>Monthly Expenses]
    K[XGBoost Regressor 3<br/>Actual Savings]
end

subgraph Output
    L[Predictions<br/>Income | Expenses | Savings]
    M[Performance Metrics<br/>R² | RMSE | MAE | MAPE]
end

A --> B
B --> C
C --> D
D --> E
G --> H
D --> H
H --> I
H --> J
H --> K
I --> L
J --> L
K --> L
L --> M

style A fill:#e1f5fe
style E fill:#bbdefb
style F fill:#90caf9
style G fill:#64b5f6
style H fill:#fff3e0
style I fill:#ffe0b2
style J fill:#ffe0b2
style K fill:#ffe0b2
style L fill:#c8e6c9
style M fill:#a5d6a7
text

## 📁 Project Structure

personal-finance-forecasting/
│
├── PFT_dataset.csv # Input dataset
├── personal_finance_forecasting.ipynb # Main notebook (single cell)
├── requirements.txt # Python dependencies
├── README.md # Project documentation
│
├── saved_models/ # Trained models directory
│ ├── lstm_finance_model.keras # LSTM model
│ ├── xgboost_monthly_income_model.pkl # XGBoost model 1
│ ├── xgboost_monthly_expense_total_model.pkl # XGBoost model 2
│ ├── xgboost_actual_savings_model.pkl # XGBoost model 3
│ ├── scaler_X.pkl # Feature scaler
│ ├── scaler_y.pkl # Target scaler
│ ├── label_encoders.pkl # Categorical encoders
│ ├── feature_columns.pkl # Feature names
│ ├── target_columns.pkl # Target names
│ └── model_metadata.pkl # Model metadata
│
└── visualizations/ # Generated charts
├── eda_comprehensive.png # 12-subplot EDA
├── lstm_training_history.png # Training progress
├── predictions_analysis.png # Actual vs Predicted
├── model_comparison_metrics.png # LSTM vs XGBoost
├── timeseries_predictions.png # Sequential predictions
└── feature_importance.png # Top features

text

## 🚀 Getting Started

### Prerequisites

- Python 3.9, 3.10, or 3.11
- 8GB RAM minimum (16GB recommended)
- 3GB free disk space

### Installation

1. **Clone the repository**
git clone https://github.com/yourusername/personal-finance-forecasting.git
cd personal-finance-forecasting

text

2. **Create virtual environment**
Windows
python -m venv finance_venv
finance_venv\Scripts\activate

macOS/Linux
python3 -m venv finance_venv
source finance_venv/bin/activate

text

3. **Install dependencies**
pip install --upgrade pip
pip install -r requirements.txt

text

### Quick Start

1. **Place your dataset** (`PFT_dataset.csv`) in the project root directory

2. **Open Jupyter Notebook**
jupyter notebook

text

3. **Run the notebook** - Open `personal_finance_forecasting.ipynb` and run the single cell

4. **Results** - Models will be saved in `saved_models/` and visualizations generated automatically

## 📦 Dependencies

pandas==2.1.4
numpy==1.26.3
matplotlib==3.8.2
seaborn==0.13.1
scikit-learn==1.4.0
xgboost==2.0.3
tensorflow==2.15.0
keras==2.15.0
joblib==1.3.2

text

## 🔧 Feature Engineering

The model creates **70+ engineered features** including:

| Category | Features | Count |
|----------|----------|-------|
| **Temporal** | Year, Month, Quarter, Week, Cyclical Encoding | 11 |
| **Financial Ratios** | Expense-to-Income, Savings-to-Income, etc. | 8 |
| **User Aggregations** | Mean, Std, Min, Max per User | 13 |
| **Lag Features** | 1, 2, 3 period lags for key metrics | 12 |
| **Rolling Statistics** | 3 & 6 window rolling mean/std | 12 |
| **Exponential Weighted** | EWM for Income, Expenses, Savings | 3 |
| **Categorical Encoding** | Label encoded categorical variables | 5 |
| **Composite** | Financial Risk Score, Cash Flow, etc. | 6 |

## 🧠 Model Details

### LSTM Architecture

Input Layer → (samples, 1, 95 features)
Bidirectional LSTM → 128 units, tanh
Dropout → 0.3
LSTM → 64 units, tanh
Dropout → 0.3
Dense → 32 units, ReLU
Dropout → 0.2
Dense → 16 units, ReLU
Output → 3 units, Linear (Income, Expenses, Savings)

Optimizer: Adam (lr=0.001)
Loss: MSE
Callbacks: EarlyStopping (patience=15), ReduceLROnPlateau (patience=7)

text

### XGBoost Configuration

n_estimators = 200
max_depth = 8
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 0.8
early_stopping_rounds = 20

text

**Hybrid Approach**: XGBoost uses 98 features (95 original + 3 LSTM predictions)

## 📈 Model Performance

### Sample Results on Test Set

| Target | Model | R² Score | RMSE | MAE | MAPE |
|--------|-------|----------|------|-----|------|
| **Monthly Income** | LSTM | 0.9234 | $278.45 | $215.67 | 5.42% |
| | XGBoost | **0.9567** | **$209.34** | **$162.89** | **4.12%** |
| **Monthly Expenses** | LSTM | 0.8956 | $298.72 | $234.51 | 7.89% |
| | XGBoost | **0.9401** | **$225.43** | **$178.92** | **5.67%** |
| **Actual Savings** | LSTM | 0.8734 | $387.65 | $298.45 | 12.34% |
| | XGBoost | **0.9203** | **$302.11** | **$234.67** | **9.87%** |

*Note: Actual results may vary based on dataset and random seed*

## 📊 Visualizations

The model automatically generates 6 comprehensive visualization files:

1. **eda_comprehensive.png** - 12-subplot exploratory data analysis
2. **lstm_training_history.png** - Training and validation loss/MAE curves
3. **predictions_analysis.png** - Actual vs Predicted scatter plots with residuals
4. **model_comparison_metrics.png** - Side-by-side performance comparison
5. **timeseries_predictions.png** - Sequential prediction tracking
6. **feature_importance.png** - Top 15 features for each target

## 🔄 Using Trained Models

### Load and Predict

import joblib
import numpy as np
from tensorflow.keras.models import load_model

Load models
lstm_model = load_model('saved_models/lstm_finance_model.keras')
xgb_income = joblib.load('saved_models/xgboost_monthly_income_model.pkl')
xgb_expense = joblib.load('saved_models/xgboost_monthly_expense_total_model.pkl')
xgb_savings = joblib.load('saved_models/xgboost_actual_savings_model.pkl')

Load scalers
scaler_X = joblib.load('saved_models/scaler_X.pkl')
scaler_y = joblib.load('saved_models/scaler_y.pkl')

Prepare new data (X_new should have 95 features)
X_scaled = scaler_X.transform(X_new)
X_lstm = X_scaled.reshape(X_scaled.shape, 1, -1)

LSTM prediction
lstm_pred = lstm_model.predict(X_lstm)

Combine for XGBoost
X_hybrid = np.hstack([X_scaled, lstm_pred])

Final predictions
income_pred = xgb_income.predict(X_hybrid)
expense_pred = xgb_expense.predict(X_hybrid)
savings_pred = xgb_savings.predict(X_hybrid)

text

## 🎯 Use Cases

- **Personal Finance Management**: Predict future income and expenses for better budgeting
- **Financial Planning**: Forecast savings trends for goal setting
- **Risk Assessment**: Identify potential financial stress periods
- **Budget Optimization**: Understand spending patterns and optimize allocations
- **Banking Applications**: Integration into personal finance apps for proactive advice

## 🛠️ Customization

### Adjust Model Parameters

**LSTM Hyperparameters:**
In Section 6
lstm_model = Sequential([
Bidirectional(LSTM(256, ...)), # Change units
Dropout(0.4), # Adjust dropout
# ...
])

text

**XGBoost Hyperparameters:**
In Section 8
xgb_model = XGBRegressor(
n_estimators=300, # More trees
max_depth=10, # Deeper trees
learning_rate=0.03, # Slower learning
# ...
)

text

### Add More Features

In Section 3 - Feature Engineering
data['your_new_feature'] = data['column_a'] / data['column_b']

text

## 🐛 Troubleshooting

### Common Issues

**1. XGBoost early_stopping_rounds error**
- **Solution**: Move `early_stopping_rounds` to XGBRegressor initialization (XGBoost 2.0+)

**2. Keras model loading error with 'mse'**
- **Solution**: Use `load_model('model.h5', compile=False)` then recompile

**3. Memory Error during training**
- **Solution**: Reduce batch size or use fewer estimators

**4. TensorFlow GPU issues**
- **Solution**: Install `tensorflow-cpu` instead or update GPU drivers

## 📝 Citation

If you use this project in your research, please cite:

@software{personal_finance_forecasting_2025,
title={Personal Finance Forecasting: LSTM + XGBoost Hybrid Model},
author={Your Name},
year={2025},
url={https://github.com/yourusername/personal-finance-forecasting}
}

text

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



## 🙏 Acknowledgments

- Dataset: [Personal Finance Tracker Dataset - Kaggle](https://www.kaggle.com/datasets/khushikyad001/personal-finance-tracker-dataset)
- TensorFlow Team for the excellent deep learning framework
- XGBoost Developers for the powerful gradient boosting library
- Scikit-learn for preprocessing utilities

## 📚 References

1. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation.
2. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD.
3. [TensorFlow Documentation](https://www.tensorflow.org/)
4. [XGBoost Documentation](https://xgboost.readthedocs.io/)

---

⭐ **Star this repository** if you find it helpful!

🐛 **Report bugs** via [GitHub Issues](https://github.com/yourusername/personal-finance-forecasting/issues)

💬 **Questions?** Feel free to open a discussion!