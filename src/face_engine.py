from insightface.app import FaceAnalysis
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional

class FaceEngine:
    def __init__(self, model_name: str, models_dir: Path, device: str):
        print(f"[INFO] Loading InsightFace Model ({model_name})...")
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
            self.app = FaceAnalysis(name=model_name, root=str(models_dir), providers=providers)
            self.app.prepare(ctx_id=0 if device == 'cuda' else -1, det_size=(640, 640))
        except Exception as e:
            print(f"[WARNING] InsightFace failed to initialize: {e}. Face recognition will be disabled.")
            self.app = None
        self.reference_embeddings = []

    def load_reference_faces(self, ref_dir: Path):
        print(f"[INFO] Loading reference faces from {ref_dir}...")
        self.reference_embeddings = []
        if not ref_dir.exists() or self.app is None:
            return

        for img_path in ref_dir.glob("*.[jp][pn]*"):
            img = cv2.imread(str(img_path))
            if img is not None:
                faces = self.app.get(img)
                if faces:
                    # Take the most prominent face
                    faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
                    self.reference_embeddings.append(faces[0].normed_embedding)
        
        if self.reference_embeddings:
            print(f"[INFO] Loaded {len(self.reference_embeddings)} reference face embeddings.")

    def get_face_embeddings(self, image_path: str) -> List[np.ndarray]:
        if self.app is None:
            return []
        img = cv2.imread(image_path)
        if img is None:
            return []
        faces = self.app.get(img)
        return [face.normed_embedding for face in faces]

    def compute_similarity(self, img_embeddings: List[np.ndarray]) -> float:
        """Compute max cosine similarity against reference embeddings"""
        if not self.reference_embeddings or not img_embeddings:
            return 0.0
            
        max_sim = 0.0
        for emb in img_embeddings:
            for ref in self.reference_embeddings:
                sim = np.dot(emb, ref)
                if sim > max_sim:
                    max_sim = sim
        return float(max_sim)
