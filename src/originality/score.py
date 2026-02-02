import numpy as np
import pickle
import faiss
from sentence_transformers import SentenceTransformer


class OriginalityScorer:
    def __init__(
        self,
        embeddings_path="embeddings/embeddings.pkl",
        index_path="embeddings/index.faiss",
        model_name="all-MiniLM-L6-v2",
    ):
        with open(embeddings_path, "rb") as f:
            self.embeddings = pickle.load(f).astype("float32")

        self.index = faiss.read_index(index_path)
        self.model = SentenceTransformer(model_name)

    def local_baseline_distribution(self, k=5, sample_size=500):
        """
        Computes how far corpus points are from their own neighbors.
        This is the correct baseline for novelty.
        """
        distances = []

        indices = np.random.choice(
            len(self.embeddings),
            size=min(sample_size, len(self.embeddings)),
            replace=False,
        )

        for idx in indices:
            emb = self.embeddings[idx].reshape(1, -1)
            dists, _ = self.index.search(emb, k + 1)
            distances.append(np.mean(dists[0][1:]))  # exclude self

        return np.array(distances)

    def originality_percentile(self, text, baseline_distances, k=5):
        emb = self.model.encode(text).astype("float32").reshape(1, -1)
        dists, _ = self.index.search(emb, k)
        score = float(np.mean(dists))

        percentile = float(np.mean(baseline_distances < score) * 100)
        return score, percentile