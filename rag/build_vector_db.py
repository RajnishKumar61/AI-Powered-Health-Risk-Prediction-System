import os
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

documents = []

for file in os.listdir("rag/rag_docs"):
    loader = TextLoader(f"rag/rag_docs/{file}", encoding="utf-8")
    documents.extend(loader.load())

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = FAISS.from_documents(documents, embeddings)
vector_db.save_local("rag/vector_db")

print("✅ Vector database created successfully")
