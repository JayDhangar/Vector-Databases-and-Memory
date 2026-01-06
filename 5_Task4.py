from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

path=r"C:\Users\sad\Downloads\sample.pdf"

loader = PyPDFLoader(path)   
docs = loader.load()


embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

db = FAISS.from_documents(docs, embeddings)

retriever = db.as_retriever(search_kwargs={"k": 1})

query = "What is machine learning?"
results = retriever.invoke(query)

print(results[0].page_content)
