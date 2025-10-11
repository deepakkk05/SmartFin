import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from datetime import datetime
import shap
import pickle
import os
from collections import deque
import random

########################################
# 1. DATA LOADING & PREPROCESSING
########################################
def load_data():
    asset_df = pd.read_csv("FAR-Trans-Data/asset_information.csv")
    customer_df = pd.read_csv("FAR-Trans-Data/customer_information.csv")
    transactions_df = pd.read_csv("FAR-Trans-Data/transactions.csv")
    limit_prices_df = pd.read_csv("FAR-Trans-Data/limit_prices.csv")
    return asset_df, customer_df, transactions_df, limit_prices_df

def preprocess_data(transactions_df):
    buys = transactions_df[transactions_df.transactionType == "Buy"].copy()
    buys['timestamp'] = pd.to_datetime(buys.timestamp)
    buys = buys.sort_values('timestamp')
    return buys

def leave_one_out_split(buys):
    train_list, test_list = [], []
    for uid, grp in buys.groupby('customerID'):
        if len(grp) < 2:
            train_list.append(grp)
        else:
            train_list.append(grp.iloc[:-1])
            test_list.append(grp.iloc[-1:])
    train_df = pd.concat(train_list)
    test_df = pd.concat(test_list) if test_list else pd.DataFrame(columns=buys.columns)
    return train_df, test_df

def build_rating_matrix(train_df):
    rating_df = train_df.groupby(['customerID','ISIN']).size().reset_index(name='count')
    rating_matrix = rating_df.pivot(index='customerID', columns='ISIN', values='count').fillna(0)
    return rating_matrix, rating_df

########################################
# 2. COLLABORATIVE FILTERING COMPONENT
########################################
def matrix_factorization(rating_matrix, n_components=5):
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    U = svd.fit_transform(rating_matrix)
    V = svd.components_.T
    pred_ratings = np.dot(U, V.T)
    pred_df = pd.DataFrame(pred_ratings, index=rating_matrix.index, columns=rating_matrix.columns)
    return pred_df

########################################
# 3. CONTENT-BASED FILTERING COMPONENT
########################################
def content_based_scores(customer_id, rating_df, asset_df, limit_prices_df):
    asset_features = asset_df[['ISIN', 'assetCategory', 'assetSubCategory', 'sector', 'industry', 'marketID']].copy()
    asset_features = asset_features.merge(
        limit_prices_df[['ISIN', 'profitability']],
        on='ISIN',
        how='left'
    )
    
    asset_features['profitability'] = asset_features['profitability'].fillna(asset_features['profitability'].median())
    asset_features['sector'] = asset_features['sector'].fillna('Unknown')
    asset_features['industry'] = asset_features['industry'].fillna('Unknown')
    
    feature_cols = ['assetCategory', 'assetSubCategory', 'sector', 'industry', 'marketID']
    encoded_features = pd.get_dummies(asset_features[feature_cols])
    encoded_features['profitability'] = asset_features['profitability']
    encoded_features.index = asset_features['ISIN']
    
    user_assets = rating_df[rating_df['customerID'] == customer_id]['ISIN'].unique().tolist()
    user_assets = [asset for asset in user_assets if asset in encoded_features.index]
    
    if len(user_assets) == 0:
        return pd.Series(0.5, index=encoded_features.index)
    
    user_profile = encoded_features.loc[user_assets].mean()
    similarity_scores = cosine_similarity(
        user_profile.values.reshape(1, -1),
        encoded_features.values
    )[0]
    
    content_scores = pd.Series(similarity_scores, index=encoded_features.index)
    return content_scores

########################################
# 4. DEMOGRAPHIC-BASED COMPONENT
########################################
def demographic_score(customer_id, customer_df, asset_df):
    def normalize_label(label):
        if pd.isna(label) or label == "Not_Available":
            return None
        return label.replace("Predicted_", "")
    
    risk_map = {"Conservative": 1, "Income": 2, "Balanced": 3, "Aggressive": 4}
    cap_map = {"CAP_LT30K": 1, "CAP_30K_80K": 2, "CAP_80K_300K": 3, "CAP_GT300K": 4}
    
    customer_df_sorted = customer_df.sort_values("timestamp").drop_duplicates("customerID", keep="last")
    user_info = customer_df_sorted[customer_df_sorted["customerID"] == customer_id]
    
    if user_info.empty:
        return pd.Series(0.5, index=asset_df["ISIN"])
    
    risk = normalize_label(user_info["riskLevel"].values[0])
    cap = normalize_label(user_info["investmentCapacity"].values[0])
    customer_type = user_info["customerType"].values[0]
    
    if risk not in risk_map or cap not in cap_map:
        return pd.Series(0.5, index=asset_df["ISIN"])
    
    user_vector = np.array([
        risk_map[risk],
        cap_map[cap],
        1 if customer_type == "Premium" else 0,
        1 if customer_type == "Professional" else 0,
    ])
    
    asset_scores = []
    for cat in asset_df["assetCategory"].unique():
        demographics = customer_df.copy()
        demographics["riskLevel"] = demographics["riskLevel"].apply(normalize_label)
        demographics["investmentCapacity"] = demographics["investmentCapacity"].apply(normalize_label)
        demographics = demographics.dropna(subset=["riskLevel", "investmentCapacity"])
        demographics = demographics[
            demographics["riskLevel"].isin(risk_map) &
            demographics["investmentCapacity"].isin(cap_map)
        ]
        
        if demographics.empty:
            avg_vector = np.array([2.5, 2.5, 0.5, 0.5])
        else:
            avg_vector = np.array([
                demographics["riskLevel"].map(risk_map).mean(),
                demographics["investmentCapacity"].map(cap_map).mean(),
                (demographics["customerType"] == "Premium").mean(),
                (demographics["customerType"] == "Professional").mean()
            ])
        
        weights = np.array([0.4, 0.3, 0.2, 0.1])
        sim = 1 - np.sqrt(np.sum(weights * (user_vector - avg_vector) ** 2)) / np.sqrt(np.sum(weights * np.array([3, 3, 1, 1]) ** 2))
        asset_scores.append((cat, sim))
    
    category_sim_map = dict(asset_scores)
    scores = asset_df["assetCategory"].map(category_sim_map).fillna(0.5)
    return pd.Series(scores.values, index=asset_df["ISIN"])

########################################
# 5. REINFORCEMENT LEARNING AGENT (PPO)
########################################
class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def size(self):
        return len(self.buffer)

# BETTER FIX: Replace the entire RLAgent class (starting around line 173)

class RLAgent:
    """
    Simplified Reinforcement Learning Agent for recommendation optimization.
    Uses a simple neural network approach with experience replay.
    """
    def __init__(self, state_dim, learning_rate=0.001, gamma=0.99, epsilon=0.1):
        self.state_dim = state_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Simple weight vector for policy (outputs single adjustment factor)
        self.weights = np.random.randn(state_dim) * 0.01
        self.bias = 0.0
        
        # Experience replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # Training history
        self.rewards_history = []
    
    def get_state(self, customer_id, cf_score, cb_score, demo_score, customer_profile):
        """Convert recommendation scores and customer profile to state vector"""
        return np.array([
            cf_score,
            cb_score,
            demo_score,
            customer_profile.get('risk_numeric', 2.5),
            customer_profile.get('capacity_numeric', 2.5),
            customer_profile.get('is_premium', 0),
            customer_profile.get('is_professional', 0)
        ])
    
    def predict_action_value(self, state):
        """Predict adjustment value given state"""
        return np.dot(state, self.weights) + self.bias
    
    def select_action(self, state, explore=True):
        """Select action using epsilon-greedy policy"""
        if explore and np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.randn()
        else:
            # Exploitation: best predicted action
            return self.predict_action_value(state)
    
    def update_scores(self, scores, state):
        """
        Adjust recommendation scores using RL policy.
        Returns adjusted scores as numpy array.
        """
        # Get single adjustment factor from state
        action_value = self.select_action(state, explore=False)
        
        # Normalize action value to [-0.2, 0.2] range for adjustment
        adjustment_factor = np.tanh(action_value) * 0.2
        
        # Apply uniform adjustment to all scores
        adjusted_scores = scores * (1 + adjustment_factor)
        adjusted_scores = np.clip(adjusted_scores, 0, 1)
        
        return adjusted_scores
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def train_step(self, batch_size=32):
        """Perform one training step using experience replay"""
        if self.replay_buffer.size() < batch_size:
            return None
        
        batch = self.replay_buffer.sample(batch_size)
        total_loss = 0
        
        for state, action, reward, next_state, done in batch:
            # Compute target Q-value
            if done:
                target = reward
            else:
                next_q_value = self.predict_action_value(next_state)
                target = reward + self.gamma * next_q_value
            
            # Compute current Q-value
            current_q = np.dot(state, self.weights) + self.bias
            
            # Compute loss (MSE)
            loss = (target - current_q) ** 2
            total_loss += loss
            
            # Gradient descent update
            grad_weights = -2 * (target - current_q) * state
            grad_bias = -2 * (target - current_q)
            
            self.weights -= self.learning_rate * grad_weights
            self.bias -= self.learning_rate * grad_bias
        
        avg_loss = total_loss / batch_size
        return avg_loss
    
    def save(self, filepath):
        """Save agent parameters"""
        np.savez(filepath, weights=self.weights, bias=self.bias)
    
    def load(self, filepath):
        """Load agent parameters"""
        if os.path.exists(filepath):
            data = np.load(filepath)
            self.weights = data['weights']
            self.bias = data['bias']
            return True
        return False


########################################
# 6. EXPLAINABILITY LAYER
########################################
class ExplainabilityLayer:
    """
    Provides interpretable explanations for recommendations using
    feature importance and score decomposition.
    """
    def __init__(self):
        self.feature_names = ['CF Score', 'CB Score', 'Demo Score', 
                             'Risk Level', 'Investment Capacity', 
                             'Premium Status', 'Professional Status']
    
    def explain_recommendation(self, isin, cf_score, cb_score, demo_score, 
                              final_score, customer_profile, asset_info):
        """
        Generate human-readable explanation for why an asset was recommended.
        """
        explanations = []
        
        # Score contribution analysis
        score_contributions = {
            'Collaborative Filtering': cf_score,
            'Content-Based': cb_score,
            'Demographic Matching': demo_score
        }
        
        # Find dominant factor
        dominant_factor = max(score_contributions, key=score_contributions.get)
        dominant_value = score_contributions[dominant_factor]
        
        explanations.append(f"**Primary Reason:** {dominant_factor} (score: {dominant_value:.3f})")
        
        # Asset-specific reasons
        if cb_score > 0.7:
            explanations.append(
                f"This asset matches your investment preferences in **{asset_info.get('sector', 'N/A')}** sector "
                f"and **{asset_info.get('assetCategory', 'N/A')}** category."
            )
        
        if cf_score > 0.7:
            explanations.append(
                "Similar investors with profiles like yours have shown strong interest in this asset."
            )
        
        if demo_score > 0.7:
            explanations.append(
                f"This asset aligns well with your **{customer_profile.get('risk_level', 'N/A')}** risk profile "
                f"and **{customer_profile.get('investment_capacity', 'N/A')}** investment capacity."
            )
        
        # Profitability information
        if 'profitability' in asset_info and asset_info['profitability'] is not None:
            prof = asset_info['profitability']
            if prof > 0.1:
                explanations.append(f"Strong profitability: **{prof:.2%}** return.")
            elif prof > 0:
                explanations.append(f"Moderate profitability: **{prof:.2%}** return.")
        
        return explanations
    
    def get_feature_importance(self, scores_dict, weights):
        """
        Calculate feature importance for the recommendation.
        """
        importance = {}
        total = sum(scores_dict[k] * weights[i] for i, k in enumerate(scores_dict.keys()))
        
        for i, key in enumerate(scores_dict.keys()):
            contribution = (scores_dict[key] * weights[i]) / total if total > 0 else 0
            importance[key] = contribution
        
        return importance
    
    def visualize_explanation(self, isin, feature_importance, explanations):
        """
        Create visualization for explanation (for Streamlit display).
        Returns formatted explanation text.
        """
        explanation_text = f"### Explanation for Asset: {isin}\n\n"
        
        # Feature importance
        explanation_text += "**Score Breakdown:**\n"
        for feature, importance in feature_importance.items():
            bar = "█" * int(importance * 20)
            explanation_text += f"- {feature}: {bar} ({importance:.1%})\n"
        
        # Detailed reasons
        explanation_text += "\n**Why This Asset?**\n"
        for exp in explanations:
            explanation_text += f"- {exp}\n"
        
        return explanation_text

########################################
# 7. HYBRID RECOMMENDATION WITH RL & EXPLAINABILITY
########################################
def normalize_scores(s):
    if s.max() - s.min() > 0:
        return (s - s.min()) / (s.max() - s.min())
    else:
        return s

def hybrid_recommendation_with_rl(customer_id, rating_matrix, pred_df, rating_df, asset_df,
                                   customer_df, limit_prices_df, weights, top_n, 
                                   rl_agent=None, explainer=None):
    """
    Enhanced hybrid recommendation with RL and explainability.
    """
    # 1. Collaborative Filtering
    if customer_id in pred_df.index:
        cf_scores = pred_df.loc[customer_id]
    else:
        cf_scores = pd.Series(0, index=rating_matrix.columns)
    
    # 2. Content-based Scores
    content_scores = content_based_scores(customer_id, rating_df, asset_df, limit_prices_df)
    
    # 3. Demographic-based Scores
    demo_scores = demographic_score(customer_id, customer_df, asset_df)
    
    # Normalize each score component to [0,1]
    cf_norm = normalize_scores(cf_scores)
    cb_norm = normalize_scores(content_scores)
    demo_norm = normalize_scores(demo_scores)
    
    # Weighted hybrid score
    final_score = weights[0]*cf_norm + weights[1]*cb_norm + weights[2]*demo_norm
    
    # Get customer profile for RL
    customer_profile = get_customer_profile(customer_id, customer_df)
    
    # Apply RL agent if provided
    if rl_agent is not None:
        # Convert to numpy array for RL processing
        score_array = final_score.values
        
        # Create state for each asset (simplified: use mean scores)
        mean_state = rl_agent.get_state(
            customer_id,
            cf_norm.mean(),
            cb_norm.mean(),
            demo_norm.mean(),
            customer_profile
        )
        
        # Adjust scores using RL
        adjusted_scores = rl_agent.update_scores(score_array, mean_state)
        final_score = pd.Series(adjusted_scores, index=final_score.index)
    
    # Exclude assets that the customer has already bought
    bought_assets = rating_df[rating_df['customerID'] == customer_id]['ISIN'].unique() if not rating_df[rating_df['customerID'] == customer_id].empty else []
    final_score = final_score.drop(labels=bought_assets, errors='ignore')
    
    recommendations = final_score.sort_values(ascending=False).head(top_n)
    
    # Generate explanations if explainer provided
    explanations = {}
    if explainer is not None:
        for isin in recommendations.index:
            asset_info = asset_df[asset_df['ISIN'] == isin].iloc[0].to_dict()
            asset_info['profitability'] = limit_prices_df[limit_prices_df['ISIN'] == isin]['profitability'].values[0] if isin in limit_prices_df['ISIN'].values else None
            
            cf_score = cf_norm.get(isin, 0)
            cb_score = cb_norm.get(isin, 0)
            demo_score = demo_norm.get(isin, 0)
            
            score_dict = {
                'Collaborative Filtering': cf_score,
                'Content-Based': cb_score,
                'Demographic': demo_score
            }
            
            feature_importance = explainer.get_feature_importance(score_dict, weights)
            explanation_list = explainer.explain_recommendation(
                isin, cf_score, cb_score, demo_score,
                recommendations[isin], customer_profile, asset_info
            )
            
            explanations[isin] = {
                'importance': feature_importance,
                'reasons': explanation_list
            }
    
    return recommendations, explanations

def get_customer_profile(customer_id, customer_df):
    """Extract customer profile as dictionary"""
    customer_df_sorted = customer_df.sort_values("timestamp").drop_duplicates("customerID", keep="last")
    user_info = customer_df_sorted[customer_df_sorted["customerID"] == customer_id]
    
    if user_info.empty:
        return {
            'risk_level': 'Unknown',
            'risk_numeric': 2.5,
            'investment_capacity': 'Unknown',
            'capacity_numeric': 2.5,
            'is_premium': 0,
            'is_professional': 0
        }
    
    risk_map = {"Conservative": 1, "Income": 2, "Balanced": 3, "Aggressive": 4}
    cap_map = {"CAP_LT30K": 1, "CAP_30K_80K": 2, "CAP_80K_300K": 3, "CAP_GT300K": 4}
    
    risk = user_info["riskLevel"].values[0]
    cap = user_info["investmentCapacity"].values[0]
    customer_type = user_info["customerType"].values[0]
    
    return {
        'risk_level': risk,
        'risk_numeric': risk_map.get(risk, 2.5),
        'investment_capacity': cap,
        'capacity_numeric': cap_map.get(cap, 2.5),
        'is_premium': 1 if customer_type == "Premium" else 0,
        'is_professional': 1 if customer_type == "Professional" else 0
    }

########################################
# 8. FEEDBACK AND RL TRAINING
########################################
def process_user_feedback(customer_id, recommended_isin, action_type, rl_agent, 
                         cf_score, cb_score, demo_score, customer_profile):
    """
    Process user feedback and update RL agent.
    action_type: 'click', 'buy', 'ignore', 'reject'
    """
    # Create state
    state = rl_agent.get_state(customer_id, cf_score, cb_score, demo_score, customer_profile)
    
    # Define rewards based on action type
    reward_map = {
        'buy': 1.0,
        'click': 0.5,
        'ignore': -0.1,
        'reject': -0.5
    }
    
    reward = reward_map.get(action_type, 0)
    
    # For simplicity, next_state is similar (in real scenario, would be updated)
    next_state = state
    done = (action_type == 'buy')
    
    # Store experience
    action = np.array([cf_score, cb_score, demo_score])
    rl_agent.store_experience(state, action, reward, next_state, done)
    
    # Train agent
    loss = rl_agent.train_step(batch_size=32)
    
    return reward, loss

########################################
# 9. EVALUATION METRICS (Keeping existing)
########################################
def compute_rmse(pred_df, test_df):
    if test_df.empty:
        return None
    y_true, y_pred = [], []
    for _, row in test_df.iterrows():
        u, i = row['customerID'], row['ISIN']
        if (u in pred_df.index) and (i in pred_df.columns):
            y_true.append(1.0)
            y_pred.append(pred_df.at[u,i])
    if not y_true:
        return None
    return np.sqrt(mean_squared_error(y_true, y_pred))

def compute_roi_at_k(recommendations, limit_prices_df, k=10):
    if recommendations is None or len(recommendations) == 0:
        return None
    top_k = recommendations.head(k)
    roi_values = limit_prices_df.set_index('ISIN')['profitability'].loc[top_k.index]
    avg_roi = roi_values.mean()
    return avg_roi

def compute_ndcg_at_k(recommendations, test_df, k=10):
    if recommendations is None or len(recommendations) == 0:
        return None
    top_k = recommendations.head(k)
    relevance = [1 if isin in test_df['ISIN'].values else 0 for isin in top_k.index]
    dcg = sum((2 ** rel - 1) / np.log2(i + 2) for i, rel in enumerate(relevance))
    num_relevant = sum(relevance)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(num_relevant, k)))
    ndcg = dcg / idcg if idcg > 0 else 0
    return ndcg

########################################
# 10. QUESTIONNAIRE PROCESSING (Keeping existing)
########################################
def process_questionnaire_responses(responses):
    risk_questions = {'q16': 0.3, 'q17': 0.3, 'q18': 0.2, 'q19': 0.2}
    risk_score = 0
    for q, weight in risk_questions.items():
        if q in responses:
            answer = responses[q]
            risk_score += weight * {'a': 4, 'b': 3, 'c': 2, 'd': 1, 'e': 0}[answer]
    
    if risk_score >= 3.5:
        risk_level = "Aggressive"
    elif risk_score >= 2.5:
        risk_level = "Balanced"
    elif risk_score >= 1.5:
        risk_level = "Income"
    else:
        risk_level = "Conservative"
    
    if 'q13' in responses:
        investment = responses['q13']
        if investment == 'a':
            investment_capacity = "CAP_GT300K"
        elif investment == 'b':
            investment_capacity = "CAP_80K_300K"
        elif investment == 'c':
            investment_capacity = "CAP_30K_80K"
        else:
            investment_capacity = "CAP_LT30K"
    else:
        investment_capacity = "CAP_LT30K"
    
    return risk_level, investment_capacity

def update_customer_profile(customer_id, risk_level, investment_capacity, customer_df):
    new_row = pd.DataFrame({
        'customerID': [customer_id],
        'customerType': ['Mass'],
        'riskLevel': [risk_level],
        'investmentCapacity': [investment_capacity],
        'lastQuestionnaireDate': [datetime.now().strftime('%Y-%m-%d')],
        'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    })
    updated_df = pd.concat([customer_df, new_row], ignore_index=True)
    return updated_df

########################################
# 11. STREAMLIT APP WITH RL & EXPLAINABILITY
########################################
def main():
    st.title("🚀 Advanced FAR-Trans Investment Recommender")
    st.write("Hybrid recommendation system with **Reinforcement Learning** and **Explainability** layers")
    
   # Initialize session state
    if 'rl_agent' not in st.session_state:
        st.session_state.rl_agent = RLAgent(state_dim=7, learning_rate=0.001)  # Remove action_dim
        st.session_state.rl_agent.load('rl_agent_weights.npz')

    
    if 'explainer' not in st.session_state:
        st.session_state.explainer = ExplainabilityLayer()
    
    if 'questionnaire_responses' not in st.session_state:
        st.session_state.questionnaire_responses = {}
    
    if 'feedback_log' not in st.session_state:
        st.session_state.feedback_log = []
    
    # Load data
    asset_df, customer_df, transactions_df, limit_prices_df = load_data()
    buys = preprocess_data(transactions_df)
    train_df, test_df = leave_one_out_split(buys)
    rating_matrix, rating_df = build_rating_matrix(train_df)
    pred_ratings = matrix_factorization(rating_matrix, n_components=5)
    
    # Sidebar controls
    st.sidebar.header("⚙️ Settings")
    customer_list = list(rating_matrix.index)
    customer_id_input = st.sidebar.selectbox("Customer ID", customer_list)
    N = st.sidebar.number_input("Top N Recommendations", min_value=1, max_value=20, value=10)
    
    st.sidebar.subheader("Component Weights")
    if 'weights' not in st.session_state:
        st.session_state.weights = [0.4, 0.3, 0.3]
    
    cf_weight = st.sidebar.slider("CF Weight", 0.0, 1.0, st.session_state.weights[0], 0.1)
    cb_weight = st.sidebar.slider("CB Weight", 0.0, 1.0, st.session_state.weights[1], 0.1)
    demo_weight = st.sidebar.slider("Demo Weight", 0.0, 1.0, st.session_state.weights[2], 0.1)
    
    st.session_state.weights = [cf_weight, cb_weight, demo_weight]
    weights = tuple(st.session_state.weights)
    
    use_rl = st.sidebar.checkbox("Enable RL Agent", value=True)
    show_explanations = st.sidebar.checkbox("Show Explanations", value=True)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Recommendations", "📝 Risk Assessment", "🧠 RL Training", "📈 Analytics"])
    
    with tab1:
        st.header("Investment Recommendations")
        
        if st.button("🎯 Generate Recommendations", type="primary"):
            with st.spinner("Generating personalized recommendations..."):
                # Get recommendations with RL and explanations
                recs, explanations = hybrid_recommendation_with_rl(
                    customer_id_input, rating_matrix, pred_ratings, rating_df, asset_df,
                    customer_df, limit_prices_df, weights, top_n=int(N),
                    rl_agent=st.session_state.rl_agent if use_rl else None,
                    explainer=st.session_state.explainer if show_explanations else None
                )
                
                # Store in session state
                st.session_state.current_recommendations = recs
                st.session_state.current_explanations = explanations
                
                # Display recommendations
                st.success(f"✅ Generated {len(recs)} recommendations for customer {customer_id_input}")
                
                # Create detailed DataFrame
                rec_details = pd.DataFrame({
                    'Rank': range(1, len(recs) + 1),
                    'ISIN': recs.index,
                    'Score': recs.values,
                    'Asset Name': asset_df.set_index('ISIN')['assetName'].loc[recs.index].values,
                    'Category': asset_df.set_index('ISIN')['assetCategory'].loc[recs.index].values,
                    'Sector': asset_df.set_index('ISIN')['sector'].loc[recs.index].values,
                    'Profitability': limit_prices_df.set_index('ISIN')['profitability'].loc[recs.index].values
                })
                
                # Display table
                st.dataframe(
                    rec_details.style.format({
                        'Score': '{:.4f}',
                        'Profitability': '{:.2%}'
                    }).background_gradient(subset=['Score'], cmap='RdYlGn'),
                    use_container_width=True
                )
                
                # Show explanations
                if show_explanations and explanations:
                    st.subheader("🔍 Why These Recommendations?")
                    
                    for i, isin in enumerate(list(recs.index)[:5]):  # Show top 5 explanations
                        with st.expander(f"#{i+1}: {asset_df[asset_df['ISIN']==isin]['assetName'].values[0]}"):
                            exp_data = explanations[isin]
                            
                            # Feature importance
                            st.write("**Score Breakdown:**")
                            for feature, importance in exp_data['importance'].items():
                                st.progress(importance, text=f"{feature}: {importance:.1%}")
                            
                            # Reasons
                            st.write("**Key Reasons:**")
                            for reason in exp_data['reasons']:
                                st.markdown(f"• {reason}")
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                roi = compute_roi_at_k(recs, limit_prices_df, k=10)
                ndcg = compute_ndcg_at_k(recs, test_df, k=10)
                
                with col1:
                    st.metric("ROI@10", f"{roi:.2%}" if roi else "N/A")
                with col2:
                    st.metric("nDCG@10", f"{ndcg:.4f}" if ndcg else "N/A")
                with col3:
                    st.metric("RL Epsilon", f"{st.session_state.rl_agent.epsilon:.3f}")
        
        # Feedback section
        if 'current_recommendations' in st.session_state:
            st.subheader("💬 Provide Feedback")
            st.write("Help improve recommendations by providing feedback:")
            
            feedback_isin = st.selectbox(
                "Select Asset", 
                st.session_state.current_recommendations.index
            )
            
            feedback_action = st.radio(
                "Your Action",
                ['buy', 'click', 'ignore', 'reject'],
                horizontal=True
            )
            
            if st.button("Submit Feedback"):
                # Get customer profile
                customer_profile = get_customer_profile(customer_id_input, customer_df)
                
                # Get scores for the asset
                cf_score = pred_ratings.loc[customer_id_input, feedback_isin] if customer_id_input in pred_ratings.index else 0
                cb_score = content_based_scores(customer_id_input, rating_df, asset_df, limit_prices_df).get(feedback_isin, 0)
                demo_score = demographic_score(customer_id_input, customer_df, asset_df).get(feedback_isin, 0)
                
                # Process feedback
                reward, loss = process_user_feedback(
                    customer_id_input, feedback_isin, feedback_action,
                    st.session_state.rl_agent, cf_score, cb_score, demo_score,
                    customer_profile
                )
                
                # Log feedback
                st.session_state.feedback_log.append({
                    'customer': customer_id_input,
                    'asset': feedback_isin,
                    'action': feedback_action,
                    'reward': reward,
                    'loss': loss,
                    'timestamp': datetime.now()
                })
                
                st.success(f"✅ Feedback recorded! Reward: {reward:.2f}")
                if loss:
                    st.info(f"RL Agent trained. Loss: {loss:.4f}")
    
    with tab2:
        st.header("📝 Risk Assessment Questionnaire")
        st.write("Answer these questions to update your investment profile:")
        
        questions = {
            'q16': "How would you rate your appetite for 'risk'?",
            'q17': "Which sentence best fits your investment expectations?",
            'q18': "Are you more concerned with potential losses or gains?",
            'q19': "If your investment declines by 20%, your reaction would be:",
            'q13': "Amount of funds available to invest:",
        }
        
        options = {
            'q16': {
                'a': "Particularly high. I really like to take risk.",
                'b': "Probably high. I usually like to take risks.",
                'c': "Moderate. I like to take the occasional risk.",
                'd': "Low. I usually don't like to take risks.",
                'e': "Too low. I don't like to take risks"
            },
            'q17': {
                'a': "Willing to take more risk for much higher returns.",
                'b': "Accept capital reductions for significant long-term profits.",
                'c': "Desire steady income with some fluctuations.",
                'd': "Achieve stable income with small ups and downs.",
                'e': "Maintain value of original capital."
            },
            'q18': {
                'a': "Always potential profits",
                'b': "Usually potential profits",
                'c': "Both gains and losses",
                'd': "Usually potential losses",
                'e': "Always potential losses"
            },
            'q19': {
                'a': "Opportunity for significant new placements",
                'b': "Opportunity for a little repositioning",
                'c': "I wouldn't do anything",
                'd': "Liquidate part of investment",
                'e': "Liquidate entire investment"
            },
            'q13': {
                'a': "Above €1 million",
                'b': "€300,001 to €1 million",
                'c': "€80,001 to €300,000",
                'd': "€30,001 to €80,000",
                'e': "Up to €30,000"
            }
        }
        
        for q_id, question in questions.items():
            st.subheader(question)
            response = st.radio(
                f"Select answer:",
                options=list(options[q_id].keys()),
                format_func=lambda x, q=q_id: options[q][x],
                key=q_id,
                horizontal=False
            )
            st.session_state.questionnaire_responses[q_id] = response
        
        if st.button("💾 Submit Questionnaire", type="primary"):
            risk_level, investment_capacity = process_questionnaire_responses(
                st.session_state.questionnaire_responses
            )
            customer_df = update_customer_profile(
                customer_id_input, risk_level, investment_capacity, customer_df
            )
            st.success(f"✅ Profile updated! Risk: **{risk_level}**, Capacity: **{investment_capacity}**")
    
    with tab3:
        st.header("🧠 Reinforcement Learning Training")
        st.write("Monitor and control the RL agent's learning process:")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Experience Buffer Size", st.session_state.rl_agent.replay_buffer.size())
            st.metric("Total Feedback Actions", len(st.session_state.feedback_log))
        
        with col2:
            if st.button("🔄 Train RL Agent (Batch)"):
                with st.spinner("Training RL agent..."):
                    losses = []
                    for _ in range(10):  # 10 training iterations
                        loss = st.session_state.rl_agent.train_step(batch_size=10)
                        if loss is not None:
                            losses.append(loss)
                    
                    if losses:
                        avg_loss = np.mean(losses)
                        st.success(f"✅ Training complete! Avg Loss: {avg_loss:.4f}")
                    else:
                        st.warning("Not enough experiences for training yet.")
            
            if st.button("💾 Save RL Agent"):
                st.session_state.rl_agent.save('rl_agent_weights.npz')
                st.success("✅ RL Agent saved successfully!")
        
        # Show recent feedback
        if st.session_state.feedback_log:
            st.subheader("Recent Feedback Log")
            feedback_df = pd.DataFrame(st.session_state.feedback_log)
            st.dataframe(feedback_df.tail(10), use_container_width=True)
    
    with tab4:
        st.header("📈 System Analytics")
        
        if st.session_state.feedback_log:
            feedback_df = pd.DataFrame(st.session_state.feedback_log)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Action Distribution")
                action_counts = feedback_df['action'].value_counts()
                st.bar_chart(action_counts)
            
            with col2:
                st.subheader("Average Reward by Action")
                avg_rewards = feedback_df.groupby('action')['reward'].mean()
                st.bar_chart(avg_rewards)
            
            st.subheader("Reward Over Time")
            feedback_df['reward_ma'] = feedback_df['reward'].rolling(window=10, min_periods=1).mean()
            st.line_chart(feedback_df[['reward', 'reward_ma']])
        else:
            st.info("No analytics data available yet. Start by generating recommendations and providing feedback!")
    
    # Footer
    st.markdown("---")
    st.markdown("Created with ❤️ using FAR-Trans Dataset | Enhanced with RL & Explainability")

if __name__ == '__main__':
    main()
