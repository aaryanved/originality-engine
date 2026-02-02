import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticOriginality:
    def __init__(
        self,
        index_path="embeddings/index.faiss",
        chunks_path="embeddings/chunks.pkl",
        model_name="all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(index_path)

        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

    def embed(self, text):
        return self.model.encode(text).astype("float32").reshape(1, -1)

    def nearest_neighbors(self, text, k=10):
        emb = self.embed(text)
        distances, indices = self.index.search(emb, k)

        neighbors = []
        for i in range(k):
            neighbors.append(
                {
                    "distance": float(distances[0][i]),
                    "text": self.chunks[indices[0][i]],
                }
            )

        return neighbors

    def semantic_novelty_score(self, text, k=10):
        neighbors = self.nearest_neighbors(text, k)
        distances = [n["distance"] for n in neighbors]

        avg_distance = float(np.mean(distances))
        min_distance = float(np.min(distances))

        return {
            "avg_distance": avg_distance,
            "min_distance": min_distance,
            "neighbors": neighbors,
        }