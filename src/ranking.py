from typing import List, Dict, Any
from .config import config

class HybridRanker:
    def __init__(self, clip_weight: float = config.clip_weight, face_weight: float = config.face_weight, object_weight: float = config.object_weight):
        self.clip_weight = clip_weight
        self.face_weight = face_weight
        self.object_weight = object_weight

    def rank(self, candidates: List[Dict[str, Any]], query: str, query_objects: List[str] = None, is_me_query: bool = False) -> List[Dict[str, Any]]:
        """
        candidates: List of dicts, each containing:
        - image_id
        - clip_score
        - face_score (optional)
        - metadata (objects, etc)
        """
        for cand in candidates:
            clip_s = cand.get('clip_score', 0.0)
            face_s = cand.get('face_score', 0.0)
            
            # Object matching score
            obj_s = 0.0
            if query_objects and 'metadata' in cand:
                detected_objects = [obj['label'].lower() for obj in cand['metadata'].get('objects', [])]
                matches = sum(1 for q_obj in query_objects if q_obj.lower() in detected_objects)
                if matches > 0:
                    obj_s = min(1.0, matches * 0.3) # Arbitrary boost per matched object

            # Adjust weights based on query type
            cw = self.clip_weight
            fw = self.face_weight if is_me_query else 0.0
            ow = self.object_weight if query_objects else 0.0
            
            # Normalize weights dynamically
            total_w = cw + fw + ow
            if total_w > 0:
                cw /= total_w
                fw /= total_w
                ow /= total_w

            final_score = (cw * clip_s) + (fw * face_s) + (ow * obj_s)
            cand['final_score'] = final_score
            cand['object_score'] = obj_s

        # Sort by final score descending
        ranked = sorted(candidates, key=lambda x: x['final_score'], reverse=True)
        return ranked
