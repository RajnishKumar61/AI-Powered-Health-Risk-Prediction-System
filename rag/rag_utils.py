from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local("rag/vector_db", embeddings)

llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    model_kwargs={"temperature": 0.3}
)

def generate_rag_explanation(disease, probability, user_features):
    docs = db.similarity_search(disease, k=3)
    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
    Disease: {disease}
    Prediction Probability: {probability}

    Patient input:
    {user_features}

    Medical knowledge:
    {context}

    Explain the risk in simple language and give general advice.
    """

    return llm(prompt)
