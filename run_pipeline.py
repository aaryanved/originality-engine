from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode("This idea measures originality of ideas.")
print("Embedding shape:", embedding.shape)