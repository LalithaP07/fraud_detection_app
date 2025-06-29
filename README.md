# fraud_detection_app
# 🔍 AI-Powered Fraud Detection & Analytics System

A cutting-edge, production-ready fraud detection and analytics platform that combines **Natural Language Processing (NLP)**, **Machine Learning**, and **interactive dashboards** to detect and analyze financial fraud in real-time.

Built using state-of-the-art models like **BERT**, an intelligent **RAG-style retrieval system**, and a multi-model ML pipeline, this system goes beyond traditional fraud detection to **understand** behavior, **visualize risk**, and **offer explainable insights**.

---

## 🚀 Features

### 🧠 AI & ML Core

- **BERT-Powered NLP**: Converts transaction data into natural language narratives and embeds them for semantic analysis.
- **Hybrid Detection Engine**:
  - *Unsupervised*: Isolation Forest to detect unknown anomalies.
  - *Supervised*: Random Forest to learn known fraud patterns.
- **Custom Feature Engineering**: Dynamic time, amount, merchant, and geolocation-based features.
- **Velocity Modeling**: Tracks rapid transaction patterns using rolling windows.

### 🤖 RAG-Based Insight System

- **FAISS vector search** over a curated fraud knowledge base.
- Ask **questions** like: _“What are velocity attacks?”_ and get domain-specific answers.
- Retrieves top-k relevant insights contextualized to user behavior.

### 📊 Interactive Dashboards

- Visual fraud score distributions, temporal trends, and anomaly clusters.
- Explore high-risk merchants, locations, and time patterns.
- Integrated using **Plotly** + **Streamlit** for a smooth UX.

---

## 🧰 Tech Stack

| Category        | Libraries & Tools                          |
|----------------|--------------------------------------------|
| ML/NLP          | `scikit-learn`, `transformers`, `sentence-transformers`, `torch` |
| Vector Search   | `faiss-cpu`                                |
| Dashboard       | `Streamlit`, `Plotly`, `Pandas`            |
| Visualization   | `Matplotlib`, `Seaborn`                    |
| Backend         | `Python 3.10+`                             |

---

## 📂 Project Structure


.
├── fraud_detection_app.py      # Main Streamlit app
├── requirements.txt            # All required Python packages
├── sample_data.csv             # Optional test dataset
├── README.md                   # You're here!
└── .streamlit/
    └── config.toml             # Theme settings for Streamlit



# Run the app
streamlit run fraud_detection_app.py

💡 How It Works
Upload or simulate transaction data
Engineer features (amount patterns, time-of-day, velocity, merchant risk)
Generate BERT-based narratives for NLP embeddings
Run hybrid ML model to detect high-risk anomalies
Visualize patterns and get RAG-based insights

⚠️ Use Cases
🏦 Banks & Credit Unions
💳 Payment Gateways & Fintech Platforms
🧪 Fraud Research & ML Experimentation
📊 AI-powered Compliance Dashboards

🧠 Sample Queries You Can Ask
“What makes online transactions high risk?”
“How does velocity fraud work?”
“Why are round dollar transactions suspicious?”
“Explain time-based fraud detection”

🙌 Acknowledgements
HuggingFace Transformers
Streamlit
FAISS by Facebook AI
Scikit-learn
