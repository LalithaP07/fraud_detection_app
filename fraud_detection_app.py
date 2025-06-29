import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Tuple, Optional
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import joblib
warnings.filterwarnings('ignore')

class FraudEmbedder:
    """Advanced fraud pattern embedding using BERT and custom features"""
    
    def __init__(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = AutoModel.from_pretrained('bert-base-uncased')
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.bert_model.to(self.device)
            self.bert_model.eval()  # Set to evaluation mode
            self.scaler = StandardScaler()
            self.bert_available = True
        except Exception as e:
            st.error(f"Error initializing BERT models: {str(e)}")
            # Fallback to simple embeddings
            self.tokenizer = None
            self.bert_model = None
            self.sentence_model = None
            self.device = torch.device('cpu')
            self.scaler = StandardScaler()
            self.bert_available = False
        
    def create_transaction_narrative(self, row: pd.Series) -> str:
        """Create natural language description of transaction for BERT processing"""
        narrative = f"Transaction of {row.get('amount', 0)} dollars "
        narrative += f"at {row.get('merchant_category', 'unknown')} merchant "
        narrative += f"using {row.get('payment_method', 'card')} "
        narrative += f"in {row.get('location', 'unknown location')} "
        narrative += f"at {row.get('time_of_day', 'unknown time')} "
        narrative += f"on {row.get('day_of_week', 'unknown day')}"
        
        if row.get('velocity_1h', 0) > 0:
            narrative += f" with {row.get('velocity_1h', 0)} transactions in past hour"
            
        return narrative
    
    def get_bert_embeddings(self, narratives: List[str]) -> np.ndarray:
        """Get BERT embeddings for narrative text."""
        if not self.bert_available or self.bert_model is None or self.tokenizer is None:
            # Return dummy embeddings if BERT is not available
            st.warning("BERT not available, returning dummy embeddings")
            return np.random.rand(len(narratives), 768)
        
        try:
            # Process in batches to avoid memory issues
            batch_size = 8
            all_embeddings = []
            
            for i in range(0, len(narratives), batch_size):
                batch_narratives = narratives[i:i+batch_size]
                
                # Tokenize input
                inputs = self.tokenizer(
                    batch_narratives,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                # Move inputs to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate embeddings (no gradients to save memory)
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                    # Return [CLS] token embeddings
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    all_embeddings.append(batch_embeddings)
            
            return np.vstack(all_embeddings) if all_embeddings else np.array([])
            
        except Exception as e:
            st.warning(f"Error generating BERT embeddings: {str(e)}. Using fallback method.")
            # Return dummy embeddings
            return np.random.rand(len(narratives), 768)

    def calculate_user_velocity(self, group):
        """Calculate velocity features for user transactions"""
        try:
            # Ensure 'timestamp' is in datetime format first
            if 'timestamp' not in group.columns:
                group['velocity_1h'] = 0
                group['velocity_24h'] = 0
                group['amount_velocity_1h'] = 0
                return group
                
            group['timestamp'] = pd.to_datetime(group['timestamp'], errors='coerce')
            group = group.dropna(subset=['timestamp'])  # Remove invalid timestamps
            
            if len(group) == 0:
                return group
                
            group = group.sort_values('timestamp').set_index('timestamp')
        
            # Calculate rolling features using count() instead of size()
            group['velocity_1h'] = group['amount'].rolling('1h').count()
            group['velocity_24h'] = group['amount'].rolling('24h').count()
            group['amount_velocity_1h'] = group['amount'].rolling('1h').sum()
        
            return group.reset_index()
        except Exception as e:
            st.warning(f"Error calculating velocity features: {str(e)}")
            # Return group with zero velocity features
            group['velocity_1h'] = 0
            group['velocity_24h'] = 0
            group['amount_velocity_1h'] = 0
            return group
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer advanced fraud detection features"""
        df = df.copy()
        
        try:
            # Time-based features
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                # Remove rows with invalid timestamps
                df = df.dropna(subset=['timestamp'])
                
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
                df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
            
            # Amount-based features
            if 'amount' in df.columns:
                # Ensure amount is numeric
                df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
                df = df.dropna(subset=['amount'])
                
                df['amount_log'] = np.log1p(df['amount'])
                df['amount_zscore'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
                df['is_round_amount'] = (df['amount'] % 1 == 0).astype(int)
                
            # Calculate velocity using a sliding window approach
            if 'user_id' in df.columns and 'timestamp' in df.columns:
                # Convert timestamp to datetime and sort
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df = df.dropna(subset=['timestamp'])
                df = df.sort_values(['user_id', 'timestamp'])
                
                # Apply velocity calculation with error handling
                try:
                    df = df.groupby('user_id', group_keys=False).apply(self.calculate_user_velocity)
                    # Reset index if it became a MultiIndex
                    if isinstance(df.index, pd.MultiIndex):
                        df = df.reset_index(drop=True)
                except Exception as e:
                    st.warning(f"Error in velocity calculation: {str(e)}. Setting default values.")
                    df['velocity_1h'] = 0
                    df['velocity_24h'] = 0
                    df['amount_velocity_1h'] = 0
            
            # Location-based features
            if 'location' in df.columns:
                location_counts = df['location'].value_counts()
                df['location_frequency'] = df['location'].map(location_counts)
                df['is_rare_location'] = (df['location_frequency'] <= 2).astype(int)
            
            # Merchant category features
            if 'merchant_category' in df.columns:
                high_risk_categories = ['ATM', 'Online', 'Gas Station', 'Gambling']
                df['is_high_risk_merchant'] = df['merchant_category'].isin(high_risk_categories).astype(int)
            
        except Exception as e:
            st.error(f"Error in feature engineering: {str(e)}")
        
        return df

class FraudRAGSystem:
    """RAG system specialized for fraud detection knowledge"""
    
    def __init__(self, embedder: FraudEmbedder):
        self.embedder = embedder
        self.fraud_knowledge_base = self._build_fraud_knowledge_base()
        self.index = None
        try:
            self.qa_pipeline = pipeline("question-answering", 
                                       model="deepset/roberta-base-squad2")
        except Exception as e:
            st.warning(f"Could not load QA pipeline: {str(e)}")
            self.qa_pipeline = None
        self._build_knowledge_index()
        
    def _build_fraud_knowledge_base(self) -> List[str]:
        """Build comprehensive fraud detection knowledge base"""
        knowledge = [
            "Credit card fraud patterns include unusual spending amounts significantly higher than normal patterns",
            "Geographic anomalies in transactions indicate potential fraud when purchases occur in distant locations within short timeframes",
            "Velocity fraud involves multiple rapid transactions in short time periods to maximize damage before detection",
            "Round dollar amounts in transactions often indicate synthetic or test transactions used by fraudsters",
            "Night-time and weekend transactions have higher fraud rates due to reduced monitoring",
            "ATM and online transactions carry higher fraud risk compared to in-person retail transactions",
            "Gambling and adult entertainment merchants have elevated fraud rates",
            "Sequential card numbers being used simultaneously indicates potential batch fraud",
            "Dormant account reactivation with high-value transactions suggests account takeover",
            "Multiple failed authentication attempts followed by successful transaction indicates credential stuffing",
            "Transactions just below reporting thresholds may indicate structuring to avoid detection",
            "Cross-border transactions, especially to high-risk countries, require enhanced monitoring",
            "Device fingerprinting anomalies help identify compromised accounts and new device usage",
            "Social engineering attacks often precede account takeover fraud by days or weeks",
            "Micro-transaction testing followed by large purchases is common in card testing fraud"
        ]
        return knowledge
    
    def _build_knowledge_index(self):
        """Build FAISS index for fraud knowledge retrieval"""
        try:
            if self.embedder.sentence_model is not None:
                embeddings = self.embedder.sentence_model.encode(self.fraud_knowledge_base)
                
                dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)
                
                faiss.normalize_L2(embeddings)
                self.index.add(embeddings)
            else:
                self.index = None
        except Exception as e:
            st.warning(f"Could not build knowledge index: {str(e)}")
            self.index = None
    
    def get_fraud_insights(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """Retrieve relevant fraud detection insights"""
        if self.index is None or self.embedder.sentence_model is None:
            # Return simple keyword matching if embedding search is not available
            results = []
            query_lower = query.lower()
            for knowledge in self.fraud_knowledge_base:
                score = sum(1 for word in query_lower.split() if word in knowledge.lower())
                if score > 0:
                    results.append((knowledge, float(score)))
            return sorted(results, key=lambda x: x[1], reverse=True)[:k]
        
        try:
            query_embedding = self.embedder.sentence_model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            scores, indices = self.index.search(query_embedding, k)
            
            results = []
            for idx, score in zip(indices[0], scores[0]):
                if idx < len(self.fraud_knowledge_base):
                    results.append((self.fraud_knowledge_base[idx], float(score)))
            
            return results
        except Exception as e:
            st.warning(f"Error in fraud insights search: {str(e)}")
            return []

class FraudDetectionModel:
    """Advanced fraud detection using multiple ML approaches"""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
        self.training_feature_count = 0

    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare numerical features for ML models"""
        feature_cols = ['amount', 'hour', 'day_of_week', 'is_weekend', 'is_night',
                       'amount_log', 'amount_zscore', 'is_round_amount', 'velocity_1h',
                       'velocity_24h', 'amount_velocity_1h', 'location_frequency',
                       'is_rare_location', 'is_high_risk_merchant']
        
        # Only use columns that exist in the dataframe
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if not available_cols:
            st.error("No suitable features found for fraud detection")
            return np.array([])
        
        self.feature_columns = available_cols
        X = df[available_cols].fillna(0)
        
        return X.values
    
    def train_unsupervised(self, X: np.ndarray):
        """Train unsupervised fraud detection model"""
        if len(X) == 0:
            return
            
        try:
            X_scaled = self.scaler.fit_transform(X)
            self.isolation_forest.fit(X_scaled)
            self.is_trained = True
        except Exception as e:
            st.error(f"Error training unsupervised model: {str(e)}")
    
    def train_supervised(self, X: np.ndarray, y: np.ndarray):
        """Train supervised fraud detection model"""
        if len(X) == 0:
            return
            
        try:
            X_scaled = self.scaler.fit_transform(X)
            self.random_forest.fit(X_scaled, y)
            self.is_trained = True
        except Exception as e:
            st.error(f"Error training supervised model: {str(e)}")
    
    def predict_fraud_probability(self, X: np.ndarray) -> np.ndarray:
        """Predict fraud probability for transactions"""
        if not self.is_trained or len(X) == 0:
            return np.array([])
            
        try:
            X_scaled = self.scaler.transform(X)
            
            # Isolation Forest anomaly scores
            anomaly_scores = self.isolation_forest.decision_function(X_scaled)
            # Convert to probability-like scores (0-1)
            fraud_probs = 1 / (1 + np.exp(anomaly_scores))
            
            return fraud_probs
        except Exception as e:
            st.error(f"Error predicting fraud probability: {str(e)}")
            return np.zeros(len(X))

class FraudAnalytics:
    """Advanced analytics for fraud detection"""
    
    def __init__(self):
        pass
    
    def analyze_fraud_patterns(self, df: pd.DataFrame, fraud_scores: np.ndarray) -> Dict:
        """Comprehensive fraud pattern analysis"""
        if len(fraud_scores) == 0:
            return {}
            
        df = df.copy()
        
        # Ensure we have the same number of scores as rows
        if len(fraud_scores) != len(df):
            st.warning(f"Mismatch between fraud scores ({len(fraud_scores)}) and dataframe rows ({len(df)})")
            min_len = min(len(fraud_scores), len(df))
            df = df.iloc[:min_len]
            fraud_scores = fraud_scores[:min_len]
        
        df['fraud_score'] = fraud_scores
        
        # Define high-risk threshold
        threshold = np.percentile(fraud_scores, 95)
        df['is_high_risk'] = (df['fraud_score'] >= threshold).astype(int)
        
        analysis = {}
        
        try:
            # Basic statistics
            analysis['total_transactions'] = len(df)
            analysis['high_risk_count'] = df['is_high_risk'].sum()
            analysis['high_risk_percentage'] = (analysis['high_risk_count'] / analysis['total_transactions']) * 100
            analysis['avg_fraud_score'] = fraud_scores.mean()
            
            # Amount analysis
            if 'amount' in df.columns:
                high_risk_amounts = df[df['is_high_risk'] == 1]['amount']
                normal_amounts = df[df['is_high_risk'] == 0]['amount']
                
                analysis['high_risk_avg_amount'] = high_risk_amounts.mean() if len(high_risk_amounts) > 0 else 0
                analysis['normal_avg_amount'] = normal_amounts.mean() if len(normal_amounts) > 0 else 0
                analysis['amount_risk_ratio'] = (analysis['high_risk_avg_amount'] / analysis['normal_avg_amount'] 
                                               if analysis['normal_avg_amount'] > 0 else 0)
            
            # Time-based patterns
            if 'hour' in df.columns:
                hour_risk = df.groupby('hour')['is_high_risk'].mean()
                analysis['riskiest_hours'] = hour_risk.nlargest(3).to_dict()
            
            # Location patterns
            if 'location' in df.columns:
                location_risk = df.groupby('location')['is_high_risk'].mean()
                analysis['riskiest_locations'] = location_risk.nlargest(5).to_dict()
            
            # Merchant category patterns
            if 'merchant_category' in df.columns:
                merchant_risk = df.groupby('merchant_category')['is_high_risk'].mean()
                analysis['riskiest_merchants'] = merchant_risk.nlargest(5).to_dict()
                
        except Exception as e:
            st.error(f"Error in fraud pattern analysis: {str(e)}")
        
        return analysis
    
    def create_fraud_visualizations(self, df: pd.DataFrame, fraud_scores: np.ndarray):
        """Create comprehensive fraud detection visualizations"""
        if len(fraud_scores) == 0:
            st.error("No fraud scores available for visualization")
            return
            
        try:
            df = df.copy()
            
            # Ensure we have the same number of scores as rows
            if len(fraud_scores) != len(df):
                min_len = min(len(fraud_scores), len(df))
                df = df.iloc[:min_len]
                fraud_scores = fraud_scores[:min_len]
                
            df['fraud_score'] = fraud_scores
            threshold = np.percentile(fraud_scores, 95)
            # Ensure threshold is within valid range and bins are strictly increasing
            bin_edges = sorted(set([0, 0.3, 0.7, threshold, 1.0]))
            if len(bin_edges) < 2:
                st.warning("Invalid bin configuration for fraud score visualization.")
                return

            # Adjust labels to match number of bins - 1
            labels = ['Low', 'Medium', 'High', 'Critical'][:len(bin_edges)-1]
            
            df['risk_level'] = pd.cut(df['fraud_score'], 
                                      bins=bin_edges, 
                                      labels=labels,
                                      include_lowest=True)
            
                        
            # Create dashboard layout
            col1, col2 = st.columns(2)
                        
            with col1:
                # Fraud Score Distribution
                fig_dist = px.histogram(df, x='fraud_score', nbins=50,
                                        title='Fraud Score Distribution',
                                        labels={'fraud_score': 'Fraud Score', 'count': 'Number of Transactions'})
                fig_dist.add_vline(x=threshold, line_dash="dash", line_color="red",
                                              annotation_text="High Risk Threshold")
                st.plotly_chart(fig_dist, use_container_width=True)
                            
                # Risk Level Pie Chart
                risk_counts = df['risk_level'].value_counts()
                fig_pie = px.pie(values=risk_counts.values, names=risk_counts.index,
                                           title='Transaction Risk Level Distribution')
                st.plotly_chart(fig_pie, use_container_width=True)
                        
                with col2:
                    # Amount vs Fraud Score Scatter
                    if 'amount' in df.columns:
                        fig_scatter = px.scatter(df, x='amount', y='fraud_score',
                                                       color='risk_level', 
                                                       title='Transaction Amount vs Fraud Score',
                                                       labels={'amount': 'Transaction Amount ($)', 
                                                              'fraud_score': 'Fraud Score'})
                        st.plotly_chart(fig_scatter, use_container_width=True)
                            
                    # Hourly Risk Pattern
                    if 'hour' in df.columns:
                        hourly_risk = df.groupby('hour')['fraud_score'].mean().reset_index()
                        fig_hourly = px.line(hourly_risk, x='hour', y='fraud_score',
                                                   title='Average Fraud Score by Hour of Day',
                                                   labels={'hour': 'Hour of Day', 'fraud_score': 'Average Fraud Score'})
                        st.plotly_chart(fig_hourly, use_container_width=True)
                                
        except Exception as e:
            st.error(f"Error creating visualizations: {str(e)}")

def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate realistic sample transaction data for demonstration"""
    np.random.seed(42)
    
    # Generate base transaction data
    data = {
        'transaction_id': [f'TXN_{i:06d}' for i in range(n_samples)],
        'user_id': np.random.randint(1000, 9999, n_samples),
        'amount': np.random.lognormal(3, 1.5, n_samples),
        'merchant_category': np.random.choice(['Grocery', 'Gas Station', 'Restaurant', 'Online', 'ATM', 'Retail', 'Gambling'], n_samples),
        'location': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio'], n_samples),
        'payment_method': np.random.choice(['Credit Card', 'Debit Card', 'Mobile Payment'], n_samples, p=[0.6, 0.3, 0.1])
    }
    
    # Generate timestamps
    start_date = datetime.now() - timedelta(days=30)
    timestamps = [start_date + timedelta(minutes=np.random.randint(0, 43200)) for _ in range(n_samples)]
    data['timestamp'] = timestamps
    
    # Add some fraudulent patterns
    fraud_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    df = pd.DataFrame(data)
    
    # Make some transactions look fraudulent
    for idx in fraud_indices:
        df.loc[idx, 'amount'] *= np.random.uniform(3, 10)  # Unusually high amounts
        if np.random.random() > 0.5:
            df.loc[idx, 'merchant_category'] = np.random.choice(['ATM', 'Online', 'Gambling'])
        if np.random.random() > 0.5:
            df.loc[idx, 'timestamp'] = df.loc[idx, 'timestamp'].replace(hour=np.random.randint(0, 6))
    
    return df

def main():
    st.set_page_config(
        page_title="Advanced Fraud Detection Analytics",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Advanced Fraud Detection Analytics System")
    st.markdown("### Professional Banking & Credit Card Fraud Detection with BERT, RAG & ML")
    
    # Initialize session state
    if 'embedder' not in st.session_state:
        with st.spinner("Initializing AI models and fraud detection systems..."):
            try:
                st.session_state.embedder = FraudEmbedder()
                st.session_state.rag_system = FraudRAGSystem(st.session_state.embedder)
                st.session_state.fraud_model = FraudDetectionModel()
                st.session_state.analytics = FraudAnalytics()
                st.success("AI models initialized successfully!")
            except Exception as e:
                st.error(f"Error initializing models: {str(e)}")
                st.stop()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose Analysis Type", 
                               ["Data Upload & Processing", "Fraud Detection", "Pattern Analysis", "AI Insights"])
    
    if page == "Data Upload & Processing":
        st.header("📊 Transaction Data Processing")
        
        # Data upload options
        data_source = st.radio("Choose data source:", ["Upload CSV", "Use Sample Data"])
        
        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Upload transaction data (CSV)", type=['csv'])
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.df = df
                    st.success(f"Successfully loaded {len(df)} transactions")
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
                    df = None
            else:
                df = None
        else:
            n_samples = st.slider("Number of sample transactions", 500, 5000, 1000)
            if st.button("Generate Sample Data"):
                with st.spinner("Generating sample transaction data..."):
                    df = generate_sample_data(n_samples)
                    st.session_state.df = df
                    st.success(f"Generated {len(df)} sample transactions")
            df = st.session_state.get('df')
        
        if df is not None:
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            st.subheader("Data Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Transactions", len(df))
            with col2:
                st.metric("Unique Users", df['user_id'].nunique() if 'user_id' in df.columns else 'N/A')
            with col3:
                st.metric("Total Amount", f"${df['amount'].sum():,.2f}" if 'amount' in df.columns else 'N/A')
            with col4:
                if 'timestamp' in df.columns:
                    try:
                        date_range = (pd.to_datetime(df['timestamp']).max() - pd.to_datetime(df['timestamp']).min()).days
                        st.metric("Date Range", f"{date_range} days")
                    except:
                        st.metric("Date Range", "N/A")
                else:
                    st.metric("Date Range", "N/A")
            
            # Feature engineering
            if st.button("Engineer Features"):
                with st.spinner("Engineering advanced fraud detection features..."):
                    df_engineered = st.session_state.embedder.engineer_features(df)
                    st.session_state.df_engineered = df_engineered
                    st.success("Feature engineering completed!")
                    
                    # Show new features
                    new_features = [col for col in df_engineered.columns if col not in df.columns]
                    if new_features:
                        st.subheader("Engineered Features")
                        st.write(new_features)
    
    elif page == "Fraud Detection":
        st.header("🚨 Fraud Detection Analysis")
        
        if 'df_engineered' not in st.session_state:
            st.warning("Please process data first in the 'Data Upload & Processing' section")
            return
        
        df = st.session_state.df_engineered
        
        # Model selection
        detection_method = st.selectbox("Choose detection method:", 
                                       ["Unsupervised (Isolation Forest)", "BERT + ML Hybrid"])
        
        if st.button("Run Fraud Detection"):
            with st.spinner("Running fraud detection analysis..."):
                # Prepare features
                X = st.session_state.fraud_model.prepare_features(df)
                
                if len(X) > 0:
                    if detection_method == "Unsupervised (Isolation Forest)":
                        st.session_state.fraud_model.train_unsupervised(X)
                        fraud_scores = st.session_state.fraud_model.predict_fraud_probability(X)
                    else:
                        # BERT + ML Hybrid approach
                        try:
                            narratives = [st.session_state.embedder.create_transaction_narrative(row) 
                                        for _, row in df.iterrows()]
                            # Limit for demo to avoid memory issues
                            sample_size = min(100, len(df))
                            bert_embeddings = st.session_state.embedder.get_bert_embeddings(narratives[:sample_size])
                            
                            # Combine BERT embeddings with traditional features
                            if len(bert_embeddings) > 0 and bert_embeddings.shape[1] > 0:
                                combined_features = np.hstack([X[:len(bert_embeddings)], bert_embeddings])
                                st.session_state.fraud_model.train_unsupervised(combined_features)
                                fraud_scores_partial = st.session_state.fraud_model.predict_fraud_probability(combined_features)
                                # For remaining transactions, generate embeddings and predict
                                if len(df) > sample_size:
                                    remaining_narratives = narratives[sample_size:]
                                    remaining_embeddings = st.session_state.embedder.get_bert_embeddings(remaining_narratives)
                                    if remaining_embeddings.shape[0] == len(X[sample_size:]):
                                        remaining_combined = np.hstack([X[sample_size:], remaining_embeddings])
                                        remaining_scores = st.session_state.fraud_model.predict_fraud_probability(remaining_combined)
                                        fraud_scores = np.concatenate([fraud_scores_partial, remaining_scores])
                                    else:
                                        st.warning("Mismatch in remaining embeddings. Using only first batch.")
                                        fraud_scores = fraud_scores_partial
                                else:
                                    fraud_scores = fraud_scores_partial

                            else:
                                st.warning("BERT embeddings failed, using traditional ML only")
                                st.session_state.fraud_model.train_unsupervised(X)
                                fraud_scores = st.session_state.fraud_model.predict_fraud_probability(X)
                        except Exception as e:
                            st.warning(f"BERT hybrid method failed: {str(e)}. Using traditional ML.")
                            st.session_state.fraud_model.train_unsupervised(X)
                            fraud_scores = st.session_state.fraud_model.predict_fraud_probability(X)
                    
                    st.session_state.fraud_scores = fraud_scores
                    st.success("Fraud detection completed!")
                    
                    # Display results
                    if len(fraud_scores) > 0:
                        threshold = np.percentile(fraud_scores, 95)
                        high_risk_count = np.sum(fraud_scores >= threshold)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("High Risk Transactions", high_risk_count)
                        with col2:
                            st.metric("Risk Rate", f"{(high_risk_count/len(df)*100):.2f}%")
                        with col3:
                            st.metric("Average Risk Score", f"{fraud_scores.mean():.3f}")
                else:
                    st.error("Unable to prepare features for fraud detection")
    
    elif page == "Pattern Analysis":
        st.header("📈 Fraud Pattern Analysis")
        
        if 'fraud_scores' not in st.session_state:
            st.warning("Please run fraud detection first")
            return
        
        df = st.session_state.df_engineered
        fraud_scores = st.session_state.fraud_scores
        
        # Generate comprehensive analysis
        analysis = st.session_state.analytics.analyze_fraud_patterns(df, fraud_scores)
        
        if analysis:
            # Key metrics
            st.subheader("Key Fraud Metrics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Transactions", analysis['total_transactions'])
            with col2:
                st.metric("High Risk Count", analysis['high_risk_count'])
            with col3:
                st.metric("Risk Percentage", f"{analysis['high_risk_percentage']:.2f}%")
            with col4:
                st.metric("Avg Risk Score", f"{analysis['avg_fraud_score']:.3f}")
            
            # Visualizations
            st.subheader("Fraud Pattern Visualizations")
            st.session_state.analytics.create_fraud_visualizations(df, fraud_scores)
            
            # Pattern insights
            st.subheader("Pattern Insights")
            
            if 'riskiest_hours' in analysis:
                st.write("**Riskiest Hours of Day:**")
                for hour, risk in analysis['riskiest_hours'].items():
                    st.write(f"- Hour {hour}: {risk:.3f} risk rate")
            
            if 'riskiest_locations' in analysis:
                st.write("**Riskiest Locations:**")
                for location, risk in analysis['riskiest_locations'].items():
                    st.write(f"- {location}: {risk:.3f} risk rate")
            
            if 'riskiest_merchants' in analysis:
                st.write("**Riskiest Merchant Categories:**")
                for merchant, risk in analysis['riskiest_merchants'].items():
                    st.write(f"- {merchant}: {risk:.3f} risk rate")
    
    elif page == "AI Insights":
        st.header("🤖 AI-Powered Fraud Insights")
        
        # RAG-based Q&A
        st.subheader("Fraud Detection Knowledge Base")
        
        question = st.text_input("Ask about fraud patterns or detection methods:",
                               placeholder="e.g., What are common credit card fraud patterns?")
        
        if question:
            with st.spinner("Searching fraud knowledge base..."):
                insights = st.session_state.rag_system.get_fraud_insights(question)
                
                st.subheader("Relevant Fraud Detection Insights:")
                for i, (insight, score) in enumerate(insights, 1):
                    st.write(f"**{i}. Relevance Score: {score:.3f}**")
                    st.write(insight)
                    st.write("---")
        
        # Transaction narrative analysis
        st.subheader("Transaction Narrative Analysis")
        
        if 'df_engineered' in st.session_state:
            df = st.session_state.df_engineered
            
            # Select a random transaction for analysis
            if st.button("Analyze Random Transaction"):
                random_idx = np.random.randint(0, len(df))
                transaction = df.iloc[random_idx]
                
                narrative = st.session_state.embedder.create_transaction_narrative(transaction)
                
                st.write("**Transaction Narrative:**")
                st.write(narrative)
                
                # Get fraud score if available
                if 'fraud_scores' in st.session_state:
                    fraud_score = st.session_state.fraud_scores[random_idx]
                    st.metric("Fraud Risk Score", f"{fraud_score:.3f}")
                    
                    if fraud_score > 0.7:
                        st.error("🚨 HIGH RISK TRANSACTION")
                    elif fraud_score > 0.4:
                        st.warning("⚠️ MEDIUM RISK TRANSACTION")
                    else:
                        st.success("✅ LOW RISK TRANSACTION")
        
        # Model performance insights
        st.subheader("Model Performance Insights")
        
        if st.session_state.fraud_model.is_trained:
            st.success("✅ Fraud detection model is trained and ready")
            st.write(f"**Features used:** {len(st.session_state.fraud_model.feature_columns)}")
            st.write(f"**Feature names:** {', '.join(st.session_state.fraud_model.feature_columns)}")
        else:
            st.warning("⚠️ Fraud detection model not yet trained")

if __name__ == "__main__":
    main()