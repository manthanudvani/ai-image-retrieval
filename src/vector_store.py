import faiss
import numpy as np
from pathlib import Path
import json

class VectorStore:
    def __init__(self, index_path: Path, mapping_path: Path, dim: int = 512):
        self.index_path = index_path
        self.mapping_path = mapping_path
        self.dim = dim
        self.index = None
        self.id_mapping = [] # Maps faiss integer index to string image_id
        
        self.load_index()

    def load_index(self):
        if self.index_path.exists() and self.mapping_path.exists():
            print(f"[INFO] Loading FAISS index from {self.index_path}")
            self.index = faiss.read_index(str(self.index_path))
            with open(self.mapping_path, 'r') as f:
                self.id_mapping = json.load(f)
        else:
            print("[INFO] Creating new FAISS index (Inner Product for Cosine Sim on normalized vectors)...")
            self.index = faiss.IndexFlatIP(self.dim)
            self.id_mapping = []

    def add_embeddings(self, image_ids: list, embeddings: np.ndarray):
        if len(image_ids) == 0:
            return
            
        # Ensure embeddings are normalized for IP to act as Cosine Similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.id_mapping.extend(image_ids)
        self.save_index()

    def save_index(self):
        faiss.write_index(self.index, str(self.index_path))
        with open(self.mapping_path, 'w') as f:
            json.dump(self.id_mapping, f)

    def search(self, query_embedding: np.ndarray, top_k: int = 10):
        if self.index is None or self.index.ntotal == 0:
            return [], []
            
        faiss.normalize_L2(query_embedding)
        distances, indices = self.index.search(query_embedding, top_k * 2) # Get more candidates for hybrid ranking
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1 and idx < len(self.id_mapping):
                results.append({
                    "image_id": self.id_mapping[idx],
                    "clip_score": float(dist)
                })
        return results
