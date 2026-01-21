# 🩺 AI-Powered Health Risk Prediction System

---

## 📌 Project Description

This project is an **AI-powered health risk prediction system** that predicts the risk of six major diseases:

- Diabetes  
- Hypertension  
- Heart Disease  
- Kidney Disease  
- Liver Disease  
- Asthma  

The system combines **Machine Learning (ML)** for disease risk prediction with **Retrieval-Augmented Generation (RAG)** to provide **medically grounded explanations** for each prediction.

Unlike traditional ML classification projects that act as black boxes, this system focuses on **interpretability, transparency, and responsible AI**, which are critical in healthcare applications.

---

## 🛠️ Tech Stack

### Machine Learning & Data Processing
- Python  
- NumPy  
- Pandas  
- Scikit-learn  
- XGBoost  
- Imbalanced-learn (SMOTE)

### Explainability & RAG
- LangChain  
- FAISS (Vector Database)  
- Hugging Face Sentence Transformers  
- Transformers

### Deployment & Visualization
- Streamlit
- Tableau

### Model & Artifact Management
- Joblib (for saving models and preprocessors)

### Version Control
- Git  
- GitHub

---

## ⭐ Key Features

- Predicts disease risk using **XGBoost classifiers**
- Handles **class imbalance** using SMOTE
- Uses **F1-score** as the primary evaluation metric
- Implements **threshold tuning** for Low / Medium / High risk categorization
- Integrates **RAG** to explain predictions using medical knowledge
- Interactive **Streamlit web application**
- Modular, scalable, and production-style project structure

---

## 🧠 Project Architecture / Flow

1. User enters health parameters via the Streamlit interface  
2. Input data is preprocessed using saved imputer and scaler  
3. XGBoost models predict disease risk probabilities  
4. Risk is categorized using tuned thresholds  
5. RAG retrieves relevant medical explanations from a FAISS vector database  
6. Predictions and explanations are displayed to the user  

This architecture clearly separates **prediction (ML)** and **reasoning (RAG)**, making the system interpretable and extensible.

---

## 🚀 Deployment

The application is deployed using **Streamlit**, providing a single public URL for real-time interaction.

- Users input health parameters
- Disease risk predictions are generated
- Medical explanations are shown alongside predictions

This setup is ideal for **demos, interviews, and proof-of-concept deployments**.

---

## ⚠️ Disclaimer

This project uses **AI-assisted synthetic clinical data** for academic and learning purposes only.  
It is **not intended for real-world medical diagnosis or clinical decision-making**.

