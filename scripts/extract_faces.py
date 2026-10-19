import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.face_engine import FaceEngine
from src.metadata import MetadataStore
from src.config import config
from tqdm import tqdm
import json
import sqlite3
import cv2

def main():
    print("[INFO] Initializing Face Engine...")
    face_engine = FaceEngine(config.face_model, config.models_dir, config.device)
    if face_engine.app is None:
        print("[ERROR] Face engine failed to load. Aborting.")
        return
        
    metadata = MetadataStore(config.metadata_dir / "metadata.db")
    
    with sqlite3.connect(metadata.db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT image_id, file_path, faces FROM images")
        rows = cursor.fetchall()
        
    images_to_process = [dict(r) for r in rows if r['faces'] == '[]' or r['faces'] is None]
    print(f"[INFO] Found {len(images_to_process)} images without face data.")
    
    updated_count = 0
    for img in tqdm(images_to_process, desc="Extracting faces"):
        try:
            image_np = cv2.imread(img['file_path'])
            if image_np is None: continue
            
            # Detect faces
            faces = face_engine.get_faces(image_np)
            if not faces: continue
            
            # Convert face data to JSON
            face_data = []
            for face in faces:
                face_data.append({
                    "bbox": face.bbox.tolist(),
                    "embedding": face.normed_embedding.tolist()
                })
            
            # Update DB
            with sqlite3.connect(metadata.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE images SET faces = ? WHERE image_id = ?", (json.dumps(face_data), img['image_id']))
                conn.commit()
            updated_count += 1
        except Exception as e:
            print(f"Error on {img['image_id']}: {e}")
            
    print(f"[INFO] Successfully updated {updated_count} images with face data.")

if __name__ == "__main__":
    main()
