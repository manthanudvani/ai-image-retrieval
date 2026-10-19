import os
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

from .config import config
from .clip_engine import CLIPEngine
from .object_detector import ObjectDetector
from .face_engine import FaceEngine
from .metadata import MetadataStore
from .vector_store import VectorStore

class Preprocessor:
    def __init__(self):
        self.clip = CLIPEngine(config.clip_model, config.device)
        self.yolo = ObjectDetector(config.yolo_model, config.models_dir)
        self.face = FaceEngine(config.face_model, config.models_dir, config.device)
        self.metadata = MetadataStore(config.metadata_dir / "metadata.db")
        
        # We need to determine CLIP dimension. ViT-B/32 has 512.
        self.vector_store = VectorStore(
            index_path=config.faiss_dir / "faiss.index",
            mapping_path=config.faiss_dir / "mapping.json",
            dim=512 
        )

    def process_directory(self, input_dir: Path):
        print(f"\n[INFO] Processing images in {input_dir}")
        if not input_dir.exists():
            print(f"[WARNING] Directory does not exist: {input_dir}")
            return

        image_paths = list(input_dir.glob("*.[jp][pn]*"))
        if not image_paths:
            print("[INFO] No images found.")
            return

        # Check existing to avoid reprocessing
        existing_images = {img['image_id'] for img in self.metadata.get_all_images()}
        
        to_process = []
        for p in image_paths:
            image_id = p.stem
            if image_id not in existing_images:
                to_process.append(p)

        if not to_process:
            print("[INFO] All images already processed.")
            return

        print(f"[INFO] Found {len(to_process)} new images to process.")
        
        batch_size = 32
        for i in tqdm(range(0, len(to_process), batch_size), desc="Processing Batches"):
            batch_paths = to_process[i:i+batch_size]
            batch_images = []
            valid_paths = []
            valid_ids = []
            
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    batch_images.append(img)
                    valid_paths.append(str(p))
                    valid_ids.append(p.stem)
                except Exception as e:
                    print(f"\n[ERROR] Failed to load {p}: {e}")

            if not batch_images:
                continue

            # 1. CLIP Embeddings
            embeddings = self.clip.get_image_embeddings(batch_images)
            self.vector_store.add_embeddings(valid_ids, embeddings)

            # 2. Metadata, YOLO, Faces
            for j, p_str in enumerate(valid_paths):
                image_id = valid_ids[j]
                img_obj = batch_images[j]
                
                # YOLO
                try:
                    objects = self.yolo.detect(p_str)
                except Exception as e:
                    print(f"YOLO failed on {p_str}: {e}")
                    objects = []
                    
                # Faces
                try:
                    face_embs = self.face.get_face_embeddings(p_str)
                    has_faces = len(face_embs) > 0
                    
                    # We could store face embeddings to disk, but for this project, 
                    # we will compute similarity dynamically during query if it's a "me" query,
                    # OR we can precompute similarity against reference faces right now and store in metadata!
                    # Precomputing is much faster for search.
                except Exception as e:
                    print(f"Face detection failed on {p_str}: {e}")
                    has_faces = False

                meta_dict = {
                    "width": img_obj.width,
                    "height": img_obj.height,
                    "format": img_obj.format or "JPEG",
                    "objects": objects,
                    "has_faces": has_faces
                }
                
                # In a more advanced system, we'd store the raw face embeddings in another FAISS index.
                # For simplicity here, we will just store whether it has a face.
                
                self.metadata.add_image(image_id, p_str, meta_dict)

        print("[INFO] Processing complete.")
