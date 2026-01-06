from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model= SentenceTransformer("all-MiniLM-L6-v2")

doc=["Computer Science is the best subject",
     "AI and ML is the future of development",
     "Python is most popular language",
     "Machine Learning improves predictions"]

embeddings= model.encode(doc)

index=faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

query="Which is most popular language"
q_embeddings=model.encode([query])

distance, indices =index.search(q_embeddings,k=1)

print("Similarity search-")
for i in indices[0]:
    print(doc[i])