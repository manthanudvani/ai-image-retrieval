import numpy as np
from typing import List, Dict, Any
from .config import config
from .clip_engine import CLIPEngine
from .vector_store import VectorStore
from .metadata import MetadataStore
from .face_engine import FaceEngine
from .ranking import HybridRanker

class SearchEngine:
    def __init__(self):
        self.clip = CLIPEngine(config.clip_model, config.device)
        self.vector_store = VectorStore(
            index_path=config.faiss_dir / "faiss.index",
            mapping_path=config.faiss_dir / "mapping.json"
        )
        self.metadata = MetadataStore(config.metadata_dir / "metadata.db")
        self.face = FaceEngine(config.face_model, config.models_dir, config.device)
        self.face.load_reference_faces(config.reference_faces_dir)
        self.ranker = HybridRanker()
        
        # Simple common objects list for YOLO keyword extraction
        self.common_objects = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def _extract_query_objects(self, query: str) -> List[str]:
        q_lower = query.lower()
        found = []
        for obj in self.common_objects:
            if obj in q_lower:
                found.append(obj)
        return found

    def search(self, query: str, top_k: int = config.top_k) -> List[Dict[str, Any]]:
        # 1. Text embedding
        query_emb = self.clip.get_text_embeddings([query])[0]
        query_emb = np.expand_dims(query_emb, axis=0)
        
        # 2. FAISS search for candidates
        candidates = self.vector_store.search(query_emb, top_k=top_k*2)
        
        # 3. Analyze query
        is_me_query = "me" in query.lower().split() or "myself" in query.lower().split()
        query_objects = self._extract_query_objects(query)

        # 4. Enrich candidates with metadata and dynamic scores
        enriched = []
        for cand in candidates:
            image_id = cand['image_id']
            meta = self.metadata.get_image(image_id)
            if not meta:
                continue
                
            cand['metadata'] = meta
            
            # Dynamic face scoring if it's a "me" query and references exist
            if is_me_query and self.face.reference_embeddings and meta.get('has_faces'):
                # Extract faces dynamically. In production, store them.
                face_embs = self.face.get_face_embeddings(meta['file_path'])
                face_score = self.face.compute_similarity(face_embs)
                cand['face_score'] = face_score
            else:
                cand['face_score'] = 0.0
                
            enriched.append(cand)
            
        # 5. Hybrid Ranking
        ranked = self.ranker.rank(enriched, query, query_objects, is_me_query)
        
        return ranked[:top_k]
