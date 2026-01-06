from sentence_transformers import SentenceTransformer
import chromadb
from transformers import pipeline

client = chromadb.Client()
collection = client.create_collection(name="docs")

documents = [
    "Computer Science is the study of computation and algorithms.",
    "Artificial Intelligence enables machines to mimic human thinking.",
    "Machine Learning is a subset of AI that learns from data.",
    "Python is the most popular language for AI and ML development.",
    "Deep learning uses neural networks with multiple layers."
]

model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = model.encode(documents)

collection.add(
    documents=documents,
    embeddings=doc_embeddings.tolist(),
    ids=[str(i) for i in range(len(documents))]
)

query = "what is the subset of AI?"
q_emb = model.encode([query])

result = collection.query(
    query_embeddings=q_emb.tolist(),
    n_results=1
)

context = result["documents"][0][0]  

prompt =f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}

Answer:
"""
 
generator = pipeline("text-generation", model="gpt2", device=-1)
response = generator(prompt,do_sample=True,temperature=0.7,max_new_tokens=30,truncation=True)

#split("Answer:")[-1].strip()

print(response[0]["generated_text"])






