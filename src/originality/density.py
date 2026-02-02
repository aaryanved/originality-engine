import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class DensityEstimator:
    def __init__(
        self,
        index_path="embeddings/index.faiss",
        model_name="all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(index_path)

    def local_density(self, text, radius=1.2):
        emb = self.model.encode(text).astype("float32").reshape(1, -1)

        # Range search finds all points within radius
        lims, _, _ = self.index.range_search(emb, radius)

        count = lims[1] - lims[0]
        return int(count)