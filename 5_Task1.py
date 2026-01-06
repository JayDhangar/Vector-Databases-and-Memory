from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "AI is transforming the world with development",
    "Machine learning is powerful",
    "I love playing volleyball"
]

embeddings = model.encode(sentences)
print(embeddings)  
