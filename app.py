import streamlit as st
import joblib
import numpy as np

# -------- RAG imports --------
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# =============================
# LOAD MODELS & PREPROCESSORS
# =============================
models = joblib.load("xgb_models.pkl")
scaler = joblib.load("scaler.pkl")
feature_cols = joblib.load("feature_cols.pkl")
feature_means = joblib.load("feature_means.pkl")

# =============================
# LOAD RAG VECTOR DATABASE
# =============================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = FAISS.load_local(
    "rag/vector_db",
    embeddings,
    allow_dangerous_deserialization=True
)

# =============================
# STREAMLIT UI
# =============================
st.title("🔍 AI-Powered Health Risk Prediction System")
st.write("Predict disease risks and get medical explanations using ML + RAG")

# -------- INPUT FORM --------
with st.form("input_form"):
    age = st.number_input("Age", 1, 120, 50)
    glucose = st.number_input("Blood Glucose (mg/dL)", 50, 400, 120)
    hba1c = st.number_input("HbA1c (%)", 3.0, 12.0, 6.0)
    systolic = st.number_input("Systolic BP", 80, 200, 130)
    diastolic = st.number_input("Diastolic BP", 60, 150, 90)
    cholesterol_ratio = st.number_input("Cholesterol Ratio", 1.0, 10.0, 4.5)
    egfr = st.number_input("eGFR", 20.0, 150.0, 90.0)
    creatinine = st.number_input("Creatinine (mg/dL)", 0.3, 15.0, 1.0)

    submitted = st.form_submit_button("Predict")

# =============================
# PREDICTION + RAG
# =============================
if submitted:

    input_data = {
        "age": age,
        "blood_glucose_mg_dl": glucose,
        "hba1c": hba1c,
        "bp_systolic": systolic,
        "bp_diastolic": diastolic,
        "cholesterol_ratio": cholesterol_ratio,
        "egfr": egfr,
        "creatinine_mg_dl": creatinine
    }

    # Fill missing features safely
    full_input = [
        input_data.get(col, feature_means[col])
        for col in feature_cols
    ]

    full_input = np.array(full_input).reshape(1, -1)
    full_input_scaled = scaler.transform(full_input)

    st.subheader("📌 Disease Risk Predictions")

    for disease, model in models.items():
        prob = model.predict_proba(full_input_scaled)[0][1]

        if prob > 0.85:
            risk = "🔴 High Risk"
        elif prob > 0.50:
            risk = "🟡 Medium Risk"
        else:
            risk = "🟢 Low Risk"

        st.markdown(f"### {disease}")
        st.write(f"**Risk Level:** {risk} ({prob:.2f})")

        # -------- RAG EXPLANATION --------
        query = f"Explain medical reasons and risk factors for {disease}"
        docs = vector_db.similarity_search(query, k=1)

        st.markdown("**Medical Explanation (RAG):**")
        for d in docs:
            st.write("- ", d.page_content)

        st.divider()
