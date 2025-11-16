"""
SmartFin - Module 3: Credit Score Prediction
=============================================
A comprehensive credit scoring system using XGBoost and SHAP explainability
Dataset: Give Me Some Credit (Kaggle)
Author: AI/ML Developer
"""

# ============================================================================
# 1. IMPORTS AND SETUP
# ============================================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score
)
import xgboost as xgb

# Explainability
import shap

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
sns.set_style("whitegrid")

print("=" * 80)
print("SmartFin - Credit Score Prediction System")
print("=" * 80)
print("\n✓ Libraries imported successfully\n")


# ============================================================================
# 2. DATA LOADING
# ============================================================================

def load_data(train_path='cs-training.csv', test_path='cs-test.csv'):
    """
    Load training and testing datasets
    
    Parameters:
    -----------
    train_path : str
        Path to training CSV file
    test_path : str
        Path to testing CSV file
        
    Returns:
    --------
    train_df, test_df : pd.DataFrame
        Training and testing datasets
    """
    print("Loading datasets...")
    
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Drop unnecessary columns
    if 'Unnamed: 0' in train_df.columns:
        train_df = train_df.drop('Unnamed: 0', axis=1)
    if 'Unnamed: 0' in test_df.columns:
        test_df = test_df.drop('Unnamed: 0', axis=1)
    
    print(f"✓ Training data loaded: {train_df.shape}")
    print(f"✓ Test data loaded: {test_df.shape}")
    print(f"\nFeatures: {list(train_df.columns)}")
    
    return train_df, test_df


# ============================================================================
# 3. EXPLORATORY DATA ANALYSIS
# ============================================================================

def perform_eda(df, title="Dataset"):
    """
    Perform exploratory data analysis
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    title : str
        Title for the analysis
    """
    print(f"\n{'=' * 80}")
    print(f"{title} - Exploratory Data Analysis")
    print(f"{'=' * 80}\n")
    
    # Basic info
    print("Dataset Shape:", df.shape)
    print("\nColumn Data Types:")
    print(df.dtypes)
    
    # Missing values
    print("\nMissing Values:")
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    missing_table = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    print(missing_table[missing_table['Missing Count'] > 0].sort_values('Percentage', ascending=False))
    
    # Statistical summary
    print("\nStatistical Summary:")
    print(df.describe())
    
    # Target distribution (if exists)
    if 'SeriousDlqin2yrs' in df.columns:
        print("\nTarget Distribution:")
        target_dist = df['SeriousDlqin2yrs'].value_counts()
        print(target_dist)
        print(f"\nClass Balance: {100 * target_dist[1] / len(df):.2f}% defaulters")
    
    return missing_table


# ============================================================================
# 4. DATA PREPROCESSING
# ============================================================================

class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline
    """
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.feature_names = None
        
    def handle_missing_values(self, df):
        """
        Handle missing values using domain-specific strategies
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        df : pd.DataFrame
            Dataframe with imputed values
        """
        print("\nHandling missing values...")
        
        df = df.copy()
        
        # MonthlyIncome: Impute with median
        if df['MonthlyIncome'].isnull().sum() > 0:
            median_income = df['MonthlyIncome'].median()
            df['MonthlyIncome'].fillna(median_income, inplace=True)
            print(f"  - MonthlyIncome: Filled {df['MonthlyIncome'].isnull().sum()} values with median ({median_income:.2f})")
        
        # NumberOfDependents: Impute with 0 (assume no dependents if missing)
        if df['NumberOfDependents'].isnull().sum() > 0:
            df['NumberOfDependents'].fillna(0, inplace=True)
            print(f"  - NumberOfDependents: Filled with 0")
        
        print("✓ Missing values handled\n")
        return df
    
    def handle_outliers(self, df):
        """
        Handle outliers using capping strategy
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        df : pd.DataFrame
            Dataframe with capped outliers
        """
        print("Handling outliers...")
        
        df = df.copy()
        
        # Cap age at reasonable limits
        df['age'] = df['age'].clip(lower=18, upper=100)
        
        # Cap RevolvingUtilizationOfUnsecuredLines
        df['RevolvingUtilizationOfUnsecuredLines'] = df['RevolvingUtilizationOfUnsecuredLines'].clip(upper=2)
        
        # Cap DebtRatio (remove extreme values)
        df['DebtRatio'] = df['DebtRatio'].clip(upper=5)
        
        # Cap late payment counts
        late_payment_cols = [
            'NumberOfTime30-59DaysPastDueNotWorse',
            'NumberOfTimes90DaysLate',
            'NumberOfTime60-89DaysPastDueNotWorse'
        ]
        for col in late_payment_cols:
            df[col] = df[col].clip(upper=10)
        
        print("✓ Outliers handled\n")
        return df
    
    def engineer_features(self, df):
        """
        Create engineered features based on domain knowledge
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        df : pd.DataFrame
            Dataframe with new features
        """
        print("Engineering features...")
        
        df = df.copy()
        
        # 1. Debt-to-Income Ratio (normalized)
        df['DebtToIncomeRatio'] = df['DebtRatio'] / (df['MonthlyIncome'] / 1000 + 1)
        
        # 2. Income per Dependent
        df['IncomePerDependent'] = df['MonthlyIncome'] / (df['NumberOfDependents'] + 1)
        
        # 3. Total Late Payments
        df['TotalLatePayments'] = (
            df['NumberOfTime30-59DaysPastDueNotWorse'] +
            df['NumberOfTime60-89DaysPastDueNotWorse'] +
            df['NumberOfTimes90DaysLate']
        )
        
        # 4. Severe Delinquency Flag
        df['HasSevereDelinquency'] = (df['NumberOfTimes90DaysLate'] > 0).astype(int)
        
        # 5. Credit Utilization Score (binned)
        df['UtilizationCategory'] = pd.cut(
            df['RevolvingUtilizationOfUnsecuredLines'],
            bins=[-0.01, 0.3, 0.7, 1.0, np.inf],
            labels=[0, 1, 2, 3]
        ).astype(int)
        
        # 6. Age Category
        df['AgeCategory'] = pd.cut(
            df['age'],
            bins=[0, 25, 35, 45, 55, 65, 100],
            labels=[0, 1, 2, 3, 4, 5]
        ).astype(int)
        
        # 7. Credit Line Utilization
        df['CreditLineUtilization'] = (
            df['RevolvingUtilizationOfUnsecuredLines'] * 
            df['NumberOfOpenCreditLinesAndLoans']
        )
        
        # 8. Real Estate Ownership Ratio
        df['RealEstateRatio'] = (
            df['NumberRealEstateLoansOrLines'] / 
            (df['NumberOfOpenCreditLinesAndLoans'] + 1)
        )
        
        # 9. Financial Stability Score
        df['FinancialStabilityScore'] = (
            (df['MonthlyIncome'] / 10000) -
            (df['DebtRatio']) -
            (df['TotalLatePayments'] * 0.5) +
            (df['NumberOfOpenCreditLinesAndLoans'] * 0.1)
        )
        
        # 10. Debt Burden Category
        df['DebtBurdenCategory'] = pd.cut(
            df['DebtRatio'],
            bins=[-0.01, 0.2, 0.4, 0.6, np.inf],
            labels=[0, 1, 2, 3]
        ).astype(int)
        
        print(f"✓ Created {10} engineered features\n")
        return df
    
    def fit_transform(self, train_df):
        """
        Fit preprocessor on training data and transform
        
        Parameters:
        -----------
        train_df : pd.DataFrame
            Training dataframe
            
        Returns:
        --------
        X_processed, y : np.array
            Processed features and target
        """
        print("\n" + "=" * 80)
        print("PREPROCESSING PIPELINE")
        print("=" * 80 + "\n")
        
        # Separate features and target
        if 'SeriousDlqin2yrs' in train_df.columns:
            y = train_df['SeriousDlqin2yrs'].values
            X = train_df.drop('SeriousDlqin2yrs', axis=1)
        else:
            y = None
            X = train_df.copy()
        
        # Apply preprocessing steps
        X = self.handle_missing_values(X)
        X = self.handle_outliers(X)
        X = self.engineer_features(X)
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Scale features
        print("Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        X_processed = pd.DataFrame(X_scaled, columns=self.feature_names)
        print("✓ Features scaled\n")
        
        return X_processed, y
    
    def transform(self, test_df):
        """
        Transform test data using fitted preprocessor
        
        Parameters:
        -----------
        test_df : pd.DataFrame
            Test dataframe
            
        Returns:
        --------
        X_processed : pd.DataFrame
            Processed features
        """
        # Separate target if exists
        if 'SeriousDlqin2yrs' in test_df.columns:
            y = test_df['SeriousDlqin2yrs'].values
            X = test_df.drop('SeriousDlqin2yrs', axis=1)
        else:
            y = None
            X = test_df.copy()
        
        # Apply preprocessing
        X = self.handle_missing_values(X)
        X = self.handle_outliers(X)
        X = self.engineer_features(X)
        
        # Scale
        X_scaled = self.scaler.transform(X)
        X_processed = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        return X_processed, y


# ============================================================================
# 5. MODEL TRAINING
# ============================================================================

class CreditScoreModel:
    """
    XGBoost model for credit default prediction
    """
    
    def __init__(self, random_state=42):
        """
        Initialize model with optimized hyperparameters
        """
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.best_threshold = 0.5
        
        # XGBoost parameters (optimized for imbalanced data)
        self.params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'min_child_weight': 5,
            'gamma': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 10,  # Handle class imbalance
            'random_state': random_state,
            'n_jobs': -1,
            'tree_method': 'hist'
        }
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train XGBoost model
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : np.array
            Training target
        X_val : pd.DataFrame (optional)
            Validation features
        y_val : np.array (optional)
            Validation target
            
        Returns:
        --------
        model : xgb.XGBClassifier
            Trained model
        """
        print("\n" + "=" * 80)
        print("MODEL TRAINING")
        print("=" * 80 + "\n")
        
        print("Training XGBoost model...")
        print(f"Parameters: {self.params}\n")
        
        self.feature_names = list(X_train.columns)
        
        # Initialize model
        self.model = xgb.XGBClassifier(**self.params)
        
        # Prepare eval set
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        # Train model with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=50
        )
        
        print("\n✓ Model training completed\n")
        
        return self.model
    
    def optimize_threshold(self, X_val, y_val):
        """
        Optimize classification threshold for best F1 score
        
        Parameters:
        -----------
        X_val : pd.DataFrame
            Validation features
        y_val : np.array
            Validation target
            
        Returns:
        --------
        best_threshold : float
            Optimal threshold
        """
        print("Optimizing classification threshold...")
        
        # Get probabilities
        y_proba = self.model.predict_proba(X_val)[:, 1]
        
        # Try different thresholds
        thresholds = np.arange(0.1, 0.9, 0.05)
        f1_scores = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            f1 = f1_score(y_val, y_pred)
            f1_scores.append(f1)
        
        # Find best threshold
        best_idx = np.argmax(f1_scores)
        self.best_threshold = thresholds[best_idx]
        
        print(f"✓ Optimal threshold: {self.best_threshold:.3f} (F1: {f1_scores[best_idx]:.4f})\n")
        
        return self.best_threshold
    
    def predict_proba(self, X):
        """
        Predict default probabilities
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
            
        Returns:
        --------
        probabilities : np.array
            Default probabilities
        """
        return self.model.predict_proba(X)[:, 1]
    
    def predict(self, X, threshold=None):
        """
        Predict default labels
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        threshold : float (optional)
            Classification threshold
            
        Returns:
        --------
        predictions : np.array
            Binary predictions
        """
        if threshold is None:
            threshold = self.best_threshold
        
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)


# ============================================================================
# 6. CREDIT SCORE CONVERSION
# ============================================================================

class CreditScoreConverter:
    """
    Convert default probability to credit score (300-850 scale)
    """
    
    @staticmethod
    def probability_to_score(probability, min_score=300, max_score=850):
        """
        Convert default probability to credit score
        
        Formula: Credit Score = max_score - (probability * (max_score - min_score))
        Higher probability of default -> Lower credit score
        
        Parameters:
        -----------
        probability : float or np.array
            Default probability (0-1)
        min_score : int
            Minimum credit score
        max_score : int
            Maximum credit score
            
        Returns:
        --------
        credit_score : float or np.array
            Credit score (300-850)
        """
        # Ensure probability is in valid range
        probability = np.clip(probability, 0, 1)
        
        # Convert to credit score (inverse relationship)
        credit_score = max_score - (probability * (max_score - min_score))
        
        # Round to nearest integer
        credit_score = np.round(credit_score).astype(int)
        
        return credit_score
    
    @staticmethod
    def score_to_category(credit_score):
        """
        Categorize credit score
        
        Parameters:
        -----------
        credit_score : int or np.array
            Credit score
            
        Returns:
        --------
        category : str or np.array
            Credit category
        """
        # Accept scalars (including numpy scalars), lists, numpy arrays, and pandas Series
        # If input is a pandas Series, use its values for vectorized operations but preserve index if needed
        try:
            import pandas as _pd
        except Exception:
            _pd = None

        # Detect scalar (including numpy scalar types)
        is_scalar = np.isscalar(credit_score) or isinstance(credit_score, (np.generic,))

        if is_scalar:
            cs = int(credit_score)
            if cs >= 800:
                return "Exceptional"
            elif cs >= 740:
                return "Very Good"
            elif cs >= 670:
                return "Good"
            elif cs >= 580:
                return "Fair"
            else:
                return "Poor"

        # For non-scalars, convert to numpy array for vectorized comparisons
        if _pd is not None and isinstance(credit_score, _pd.Series):
            arr = credit_score.values
        else:
            arr = np.array(credit_score)

        # Ensure integer type for comparisons
        try:
            arr_int = arr.astype(int)
        except Exception:
            # Fallback: compare as-is
            arr_int = arr

        categories = np.empty(arr_int.shape, dtype=object)
        categories[arr_int >= 800] = "Exceptional"
        categories[(arr_int >= 740) & (arr_int < 800)] = "Very Good"
        categories[(arr_int >= 670) & (arr_int < 740)] = "Good"
        categories[(arr_int >= 580) & (arr_int < 670)] = "Fair"
        categories[arr_int < 580] = "Poor"

        return categories


# ============================================================================
# 7. SHAP EXPLAINABILITY
# ============================================================================

class SHAPExplainer:
    """
    SHAP-based model explainability
    """
    
    def __init__(self, model, X_train):
        """
        Initialize SHAP explainer
        
        Parameters:
        -----------
        model : xgb.XGBClassifier
            Trained model
        X_train : pd.DataFrame
            Training data for background
        """
        print("\n" + "=" * 80)
        print("SHAP EXPLAINABILITY INITIALIZATION")
        print("=" * 80 + "\n")
        
        print("Creating SHAP explainer...")
        self.model = model
        self.feature_names = list(X_train.columns)
        
        # Use TreeExplainer for XGBoost (faster and exact)
        self.explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values for training data sample (for global analysis)
        sample_size = min(1000, len(X_train))
        self.X_sample = X_train.sample(n=sample_size, random_state=42)
        self.shap_values_sample = self.explainer.shap_values(self.X_sample)
        
        print("✓ SHAP explainer initialized\n")
    
    def explain_instance(self, X_instance):
        """
        Get SHAP explanation for a single instance
        
        Parameters:
        -----------
        X_instance : pd.DataFrame or np.array
            Single instance (1 row)
            
        Returns:
        --------
        shap_values : np.array
            SHAP values for the instance
        """
        if isinstance(X_instance, pd.DataFrame):
            X_instance = X_instance.values
        
        if len(X_instance.shape) == 1:
            X_instance = X_instance.reshape(1, -1)
        
        shap_values = self.explainer.shap_values(X_instance)
        return shap_values[0]  # Return first row
    
    def get_top_features(self, shap_values, n_top=5):
        """
        Get top contributing features for an instance
        
        Parameters:
        -----------
        shap_values : np.array
            SHAP values
        n_top : int
            Number of top features
            
        Returns:
        --------
        top_features : list of tuples
            (feature_name, shap_value) sorted by absolute impact
        """
        # Get absolute SHAP values
        abs_shap = np.abs(shap_values)
        
        # Get top indices
        top_indices = np.argsort(abs_shap)[-n_top:][::-1]
        
        # Create list of (feature, shap_value)
        top_features = [
            (self.feature_names[idx], shap_values[idx])
            for idx in top_indices
        ]
        
        return top_features
    
    def generate_text_explanation(self, X_instance, shap_values, credit_score, default_prob):
        """
        Generate human-readable explanation
        
        Parameters:
        -----------
        X_instance : pd.Series or np.array
            Instance features
        shap_values : np.array
            SHAP values
        credit_score : int
            Predicted credit score
        default_prob : float
            Default probability
            
        Returns:
        --------
        explanation : str
            Human-readable explanation
        """
        if isinstance(X_instance, np.ndarray):
            X_instance = pd.Series(X_instance, index=self.feature_names)
        
        # Get top features
        top_features = self.get_top_features(shap_values, n_top=5)
        
        # Build explanation
        explanation = f"\n{'=' * 80}\n"
        explanation += f"CREDIT SCORE EXPLANATION\n"
        explanation += f"{'=' * 80}\n\n"
        explanation += f"Credit Score: {credit_score} ({CreditScoreConverter.score_to_category(credit_score)})\n"
        explanation += f"Default Probability: {default_prob:.2%}\n\n"
        explanation += f"Top Contributing Factors:\n"
        explanation += f"{'-' * 80}\n"
        
        for i, (feature, shap_val) in enumerate(top_features, 1):
            impact = "increases" if shap_val > 0 else "decreases"
            direction = "negatively" if shap_val > 0 else "positively"
            
            explanation += f"{i}. {feature}\n"
            explanation += f"   - SHAP Value: {shap_val:.4f}\n"
            explanation += f"   - Impact: This feature {impact} default risk, affecting credit score {direction}\n"
            
            # Add feature-specific insights
            if 'Late' in feature or 'Delinquency' in feature:
                explanation += f"   - Insight: Payment history is a critical factor in credit scoring\n"
            elif 'Income' in feature:
                explanation += f"   - Insight: Income level affects ability to repay debts\n"
            elif 'Debt' in feature:
                explanation += f"   - Insight: Debt burden impacts creditworthiness\n"
            elif 'Utilization' in feature:
                explanation += f"   - Insight: Credit utilization affects available credit capacity\n"
            
            explanation += "\n"
        
        return explanation
    
    def plot_global_importance(self, save_path=None):
        """
        Plot global feature importance
        
        Parameters:
        -----------
        save_path : str (optional)
            Path to save plot
        """
        print("Generating global feature importance plot...")
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values_sample,
            self.X_sample,
            plot_type="bar",
            show=False
        )
        plt.title("Global Feature Importance (SHAP)", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        
        plt.show()
    
    def plot_instance_explanation(self, X_instance, save_path=None):
        """
        Plot SHAP waterfall for single instance
        
        Parameters:
        -----------
        X_instance : pd.DataFrame
            Single instance
        save_path : str (optional)
            Path to save plot
        """
        print("Generating instance explanation plot...")
        
        # Get SHAP values
        shap_values = self.explain_instance(X_instance)
        
        # Create explanation object
        explanation = shap.Explanation(
            values=shap_values,
            base_values=self.explainer.expected_value,
            data=X_instance.values[0] if isinstance(X_instance, pd.DataFrame) else X_instance,
            feature_names=self.feature_names
        )
        
        # Plot waterfall
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(explanation, show=False)
        plt.title("SHAP Waterfall Plot - Instance Explanation", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        
        plt.show()


# ============================================================================
# 8. WHAT-IF ANALYSIS
# ============================================================================

class WhatIfAnalyzer:
    """
    What-if analysis for credit score simulation
    """
    
    def __init__(self, model, preprocessor, shap_explainer):
        """
        Initialize what-if analyzer
        
        Parameters:
        -----------
        model : CreditScoreModel
            Trained credit score model
        preprocessor : DataPreprocessor
            Fitted preprocessor
        shap_explainer : SHAPExplainer
            SHAP explainer
        """
        self.model = model
        self.preprocessor = preprocessor
        self.shap_explainer = shap_explainer
        self.converter = CreditScoreConverter()
    
    def simulate_changes(self, X_original, feature_changes):
        """
        Simulate changes to features and predict new credit score
        
        Parameters:
        -----------
        X_original : pd.DataFrame or pd.Series
            Original instance (before preprocessing)
        feature_changes : dict
            Dictionary of {feature_name: new_value}
            
        Returns:
        --------
        results : dict
            Simulation results
        """
        # Convert to DataFrame if Series
        if isinstance(X_original, pd.Series):
            X_original = X_original.to_frame().T
        
        # Make a copy
        X_modified = X_original.copy()
        
        # Apply changes
        for feature, new_value in feature_changes.items():
            if feature in X_modified.columns:
                X_modified[feature] = new_value
        
        # Preprocess (note: this will re-engineer features)
        X_modified_processed, _ = self.preprocessor.transform(X_modified)
        X_original_processed, _ = self.preprocessor.transform(X_original)
        
        # Predict
        prob_original = self.model.predict_proba(X_original_processed)[0]
        prob_modified = self.model.predict_proba(X_modified_processed)[0]
        
        score_original = self.converter.probability_to_score(prob_original)
        score_modified = self.converter.probability_to_score(prob_modified)
        
        # Get SHAP explanations
        shap_original = self.shap_explainer.explain_instance(X_original_processed)
        shap_modified = self.shap_explainer.explain_instance(X_modified_processed)
        
        # Calculate changes
        score_change = score_modified - score_original
        prob_change = prob_modified - prob_original
        
        results = {
            'original_score': score_original,
            'modified_score': score_modified,
            'score_change': score_change,
            'original_probability': prob_original,
            'modified_probability': prob_modified,
            'probability_change': prob_change,
            'feature_changes': feature_changes,
            'shap_original': shap_original,
            'shap_modified': shap_modified
        }
        
        return results
    
    def generate_what_if_report(self, results):
        """
        Generate human-readable what-if report
        
        Parameters:
        -----------
        results : dict
            Simulation results
            
        Returns:
        --------
        report : str
            What-if analysis report
        """
        report = f"\n{'=' * 80}\n"
        report += f"WHAT-IF ANALYSIS REPORT\n"
        report += f"{'=' * 80}\n\n"
        
        report += f"CURRENT SITUATION:\n"
        report += f"{'-' * 80}\n"
        report += f"Credit Score: {results['original_score']}\n"
        report += f"Default Probability: {results['original_probability']:.2%}\n"
        report += f"Rating: {self.converter.score_to_category(results['original_score'])}\n\n"
        
        report += f"SIMULATED CHANGES:\n"
        report += f"{'-' * 80}\n"
        for feature, value in results['feature_changes'].items():
            report += f"  • {feature}: Changed to {value}\n"
        report += "\n"
        
        report += f"PROJECTED OUTCOME:\n"
        report += f"{'-' * 80}\n"
        report += f"New Credit Score: {results['modified_score']}\n"
        report += f"New Default Probability: {results['modified_probability']:.2%}\n"
        report += f"New Rating: {self.converter.score_to_category(results['modified_score'])}\n\n"
        
        report += f"IMPACT ANALYSIS:\n"
        report += f"{'-' * 80}\n"
        report += f"Credit Score Change: {results['score_change']:+d} points\n"
        report += f"Probability Change: {results['probability_change']:+.2%}\n"
        
        if results['score_change'] > 0:
            report += f"\n✓ Improvement detected! These changes would positively impact your credit score.\n"
        elif results['score_change'] < 0:
            report += f"\n⚠ Warning! These changes would negatively impact your credit score.\n"
        else:
            report += f"\n→ Neutral impact. These changes have minimal effect on your credit score.\n"
        
        return report
    
    def suggest_improvements(self, X_instance, target_score_increase=50):
        """
        Suggest feature changes to improve credit score
        
        Parameters:
        -----------
        X_instance : pd.DataFrame or pd.Series
            Original instance
        target_score_increase : int
            Desired score increase
            
        Returns:
        --------
        suggestions : list of dict
            List of suggested changes
        """
        if isinstance(X_instance, pd.Series):
            X_instance = X_instance.to_frame().T
        
        suggestions = []
        
        # Suggestion 1: Reduce debt
        suggestions.append({
            'description': 'Reduce Debt Ratio',
            'changes': {'DebtRatio': max(0.1, X_instance['DebtRatio'].values[0] * 0.7)},
            'rationale': 'Lowering debt burden improves creditworthiness'
        })
        
        # Suggestion 2: Improve payment history
        late_payment_cols = [
            'NumberOfTime30-59DaysPastDueNotWorse',
            'NumberOfTime60-89DaysPastDueNotWorse',
            'NumberOfTimes90DaysLate'
        ]
        changes = {col: 0 for col in late_payment_cols if X_instance[col].values[0] > 0}
        if changes:
            suggestions.append({
                'description': 'Eliminate Late Payments',
                'changes': changes,
                'rationale': 'Payment history is the most important factor in credit scoring'
            })
        
        # Suggestion 3: Reduce credit utilization
        if X_instance['RevolvingUtilizationOfUnsecuredLines'].values[0] > 0.3:
            suggestions.append({
                'description': 'Reduce Credit Utilization',
                'changes': {'RevolvingUtilizationOfUnsecuredLines': 0.2},
                'rationale': 'Keeping utilization below 30% is considered ideal'
            })
        
        # Suggestion 4: Increase income
        suggestions.append({
            'description': 'Increase Monthly Income',
            'changes': {'MonthlyIncome': X_instance['MonthlyIncome'].values[0] * 1.2},
            'rationale': 'Higher income improves debt-to-income ratio'
        })
        
        # Simulate each suggestion
        for suggestion in suggestions:
            results = self.simulate_changes(X_instance, suggestion['changes'])
            suggestion['projected_impact'] = results['score_change']
            suggestion['new_score'] = results['modified_score']
        
        # Sort by impact
        suggestions = sorted(suggestions, key=lambda x: x['projected_impact'], reverse=True)
        
        return suggestions


# ============================================================================
# 9. MODEL EVALUATION
# ============================================================================

class ModelEvaluator:
    """
    Comprehensive model evaluation
    """
    
    @staticmethod
    def evaluate_classification(y_true, y_pred, y_proba, title="Classification Metrics"):
        """
        Evaluate classification performance
        
        Parameters:
        -----------
        y_true : np.array
            True labels
        y_pred : np.array
            Predicted labels
        y_proba : np.array
            Predicted probabilities
        title : str
            Evaluation title
            
        Returns:
        --------
        metrics : dict
            Evaluation metrics
        """
        print(f"\n{'=' * 80}")
        print(f"{title}")
        print(f"{'=' * 80}\n")
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc_roc = roc_auc_score(y_true, y_proba)
        
        metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'AUC-ROC': auc_roc
        }
        
        # Print metrics
        for metric_name, value in metrics.items():
            print(f"{metric_name:15s}: {value:.4f}")
        
        # Confusion matrix
        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['No Default', 'Default']))
        
        return metrics
    
    @staticmethod
    def evaluate_regression(credit_scores_true, credit_scores_pred, title="Credit Score Metrics"):
        """
        Evaluate credit score predictions as regression
        
        Parameters:
        -----------
        credit_scores_true : np.array
            True credit scores
        credit_scores_pred : np.array
            Predicted credit scores
        title : str
            Evaluation title
            
        Returns:
        --------
        metrics : dict
            Regression metrics
        """
        print(f"\n{'=' * 80}")
        print(f"{title}")
        print(f"{'=' * 80}\n")
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(credit_scores_true, credit_scores_pred))
        mae = mean_absolute_error(credit_scores_true, credit_scores_pred)
        r2 = r2_score(credit_scores_true, credit_scores_pred)
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R² Score': r2
        }
        
        # Print metrics
        for metric_name, value in metrics.items():
            print(f"{metric_name:15s}: {value:.4f}")
        
        return metrics
    
    @staticmethod
    def plot_roc_curve(y_true, y_proba, save_path=None):
        """
        Plot ROC curve
        
        Parameters:
        -----------
        y_true : np.array
            True labels
        y_proba : np.array
            Predicted probabilities
        save_path : str (optional)
            Path to save plot
        """
        print("\nGenerating ROC curve...")
        
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_score = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, linewidth=2, label=f'XGBoost (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('ROC Curve - Credit Default Prediction', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ ROC curve saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_score_distribution(credit_scores, title="Credit Score Distribution", save_path=None):
        """
        Plot credit score distribution
        
        Parameters:
        -----------
        credit_scores : np.array
            Credit scores
        title : str
            Plot title
        save_path : str (optional)
            Path to save plot
        """
        print("\nGenerating credit score distribution plot...")
        
        plt.figure(figsize=(12, 6))
        
        # Histogram
        plt.hist(credit_scores, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Credit Score', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.title(title, fontsize=16, fontweight='bold')
        
        # Add vertical lines for categories
        plt.axvline(x=580, color='red', linestyle='--', linewidth=2, label='Fair')
        plt.axvline(x=670, color='orange', linestyle='--', linewidth=2, label='Good')
        plt.axvline(x=740, color='green', linestyle='--', linewidth=2, label='Very Good')
        plt.axvline(x=800, color='blue', linestyle='--', linewidth=2, label='Exceptional')
        
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Distribution plot saved to {save_path}")
        
        plt.show()


# ============================================================================
# 10. MAIN PIPELINE
# ============================================================================

def main(train_path='cs-training.csv', test_path='cs-test.csv'):
    """
    Main execution pipeline
    
    Parameters:
    -----------
    train_path : str
        Path to training data
    test_path : str
        Path to test data
    """
    
    # ========== LOAD DATA ==========
    train_df, test_df = load_data(train_path, test_path)
    
    # ========== EXPLORATORY DATA ANALYSIS ==========
    perform_eda(train_df, "Training Data")
    
    # ========== PREPROCESS DATA ==========
    preprocessor = DataPreprocessor()
    X_train_full, y_train_full = preprocessor.fit_transform(train_df)
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train_full
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    
    # ========== TRAIN MODEL ==========
    model = CreditScoreModel(random_state=RANDOM_STATE)
    model.train(X_train, y_train, X_val, y_val)
    
    # Optimize threshold
    model.optimize_threshold(X_val, y_val)
    
    # ========== MAKE PREDICTIONS ==========
    print("\n" + "=" * 80)
    print("GENERATING PREDICTIONS")
    print("=" * 80 + "\n")
    
    # Validation predictions
    y_val_proba = model.predict_proba(X_val)
    y_val_pred = model.predict(X_val)
    
    # Convert to credit scores
    converter = CreditScoreConverter()
    credit_scores_val = converter.probability_to_score(y_val_proba)
    
    # ========== EVALUATE MODEL ==========
    evaluator = ModelEvaluator()
    
    # Classification metrics
    classification_metrics = evaluator.evaluate_classification(
        y_val, y_val_pred, y_val_proba, 
        title="Validation Set - Classification Metrics"
    )
    
    # If we have actual credit scores (simulated here)
    # For demonstration, convert true labels to scores
    credit_scores_true = converter.probability_to_score(y_val)
    regression_metrics = evaluator.evaluate_regression(
        credit_scores_true, credit_scores_val,
        title="Validation Set - Credit Score Metrics"
    )
    
    # Plot ROC curve
    evaluator.plot_roc_curve(y_val, y_val_proba, save_path='roc_curve.png')
    
    # Plot score distribution
    evaluator.plot_score_distribution(
        credit_scores_val, 
        title="Predicted Credit Score Distribution",
        save_path='score_distribution.png'
    )
    
    # ========== SHAP EXPLAINABILITY ==========
    shap_explainer = SHAPExplainer(model.model, X_train)
    
    # Global importance
    shap_explainer.plot_global_importance(save_path='shap_global_importance.png')
    
    # Example instance explanation
    example_idx = 0
    X_example = X_val.iloc[[example_idx]]
    shap_values_example = shap_explainer.explain_instance(X_example)
    
    # Generate text explanation
    explanation = shap_explainer.generate_text_explanation(
        X_example.iloc[0],
        shap_values_example,
        credit_scores_val[example_idx],
        y_val_proba[example_idx]
    )
    print(explanation)
    
    # Plot instance explanation
    shap_explainer.plot_instance_explanation(X_example, save_path='shap_instance_explanation.png')
    
    # ========== WHAT-IF ANALYSIS ==========
    print("\n" + "=" * 80)
    print("WHAT-IF ANALYSIS")
    print("=" * 80 + "\n")
    
    # Get original instance (before preprocessing)
    original_instance = train_df.drop('SeriousDlqin2yrs', axis=1).iloc[[example_idx]]
    
    # Initialize what-if analyzer
    what_if_analyzer = WhatIfAnalyzer(model, preprocessor, shap_explainer)
    
    # Example: What if debt ratio is reduced by 30%?
    current_debt_ratio = original_instance['DebtRatio'].values[0]
    simulation_results = what_if_analyzer.simulate_changes(
        original_instance,
        {'DebtRatio': current_debt_ratio * 0.7}
    )
    
    # Generate report
    what_if_report = what_if_analyzer.generate_what_if_report(simulation_results)
    print(what_if_report)
    
    # Get improvement suggestions
    print("\n" + "=" * 80)
    print("IMPROVEMENT SUGGESTIONS")
    print("=" * 80 + "\n")
    
    suggestions = what_if_analyzer.suggest_improvements(original_instance)
    
    for i, suggestion in enumerate(suggestions, 1):
        print(f"\nSuggestion {i}: {suggestion['description']}")
        print(f"Rationale: {suggestion['rationale']}")
        print(f"Projected Impact: {suggestion['projected_impact']:+d} points")
        print(f"New Score: {suggestion['new_score']}")
        print(f"Changes: {suggestion['changes']}")
    
    # ========== PROCESS TEST DATA ==========
    print("\n" + "=" * 80)
    print("PROCESSING TEST DATA")
    print("=" * 80 + "\n")
    
    X_test, y_test = preprocessor.transform(test_df)
    
    # Predictions
    y_test_proba = model.predict_proba(X_test)
    y_test_pred = model.predict(X_test)
    credit_scores_test = converter.probability_to_score(y_test_proba)
    credit_categories_test = converter.score_to_category(credit_scores_test)
    
    # SHAP values for test set (sample)
    test_sample_size = min(100, len(X_test))
    X_test_sample = X_test.sample(n=test_sample_size, random_state=42)
    shap_values_test = shap_explainer.explainer.shap_values(X_test_sample)
    
    # ========== CREATE FINAL OUTPUT DATAFRAME ==========
    print("\n" + "=" * 80)
    print("CREATING FINAL OUTPUT")
    print("=" * 80 + "\n")
    
    # Get top features for each user
    top_features_list = []
    for i in range(len(X_test)):
        shap_vals = shap_explainer.explain_instance(X_test.iloc[[i]])
        top_features = shap_explainer.get_top_features(shap_vals, n_top=3)
        top_features_str = "; ".join([f"{feat}: {val:.4f}" for feat, val in top_features])
        top_features_list.append(top_features_str)
    
    # Create output DataFrame
    output_df = pd.DataFrame({
        'user_id': range(1, len(X_test) + 1),
        'predicted_credit_score': credit_scores_test,
        'credit_category': credit_categories_test,
        'default_probability': y_test_proba,
        'default_prediction': y_test_pred,
        'top_contributing_features': top_features_list
    })
    
    # Save to CSV
    output_df.to_csv('credit_score_predictions.csv', index=False)
    print("✓ Predictions saved to 'credit_score_predictions.csv'")
    
    # Display sample
    print("\nSample Predictions:")
    print(output_df.head(10))
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80 + "\n")
    
    print(f"Total Users Evaluated: {len(output_df)}")
    print(f"\nCredit Score Distribution:")
    print(f"  Mean: {credit_scores_test.mean():.2f}")
    print(f"  Median: {np.median(credit_scores_test):.2f}")
    print(f"  Std Dev: {credit_scores_test.std():.2f}")
    print(f"  Min: {credit_scores_test.min()}")
    print(f"  Max: {credit_scores_test.max()}")
    
    print(f"\nCategory Distribution:")
    category_counts = pd.Series(credit_categories_test).value_counts()
    for category, count in category_counts.items():
        pct = 100 * count / len(credit_categories_test)
        print(f"  {category}: {count} ({pct:.2f}%)")
    
    print(f"\nDefault Prediction Distribution:")
    default_counts = pd.Series(y_test_pred).value_counts()
    print(f"  No Default: {default_counts.get(0, 0)} ({100 * default_counts.get(0, 0) / len(y_test_pred):.2f}%)")
    print(f"  Default: {default_counts.get(1, 0)} ({100 * default_counts.get(1, 0) / len(y_test_pred):.2f}%)")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    return {
        'model': model,
        'preprocessor': preprocessor,
        'shap_explainer': shap_explainer,
        'what_if_analyzer': what_if_analyzer,
        'output_df': output_df,
        'metrics': {
            'classification': classification_metrics,
            'regression': regression_metrics
        }
    }


# ============================================================================
# 11. EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run main pipeline
    results = main(
        train_path='cs-training.csv',
        test_path='cs-test.csv'
    )
    
    # Access results
    model = results['model']
    preprocessor = results['preprocessor']
    shap_explainer = results['shap_explainer']
    what_if_analyzer = results['what_if_analyzer']
    output_df = results['output_df']
    
    print("\n✓ All components are ready for use!")
    print("\nYou can now:")
    print("  1. Use model.predict_proba(X) for predictions")
    print("  2. Use shap_explainer.explain_instance(X) for explanations")
    print("  3. Use what_if_analyzer.simulate_changes(X, changes) for what-if analysis")
    print("  4. Access output_df for final predictions")

