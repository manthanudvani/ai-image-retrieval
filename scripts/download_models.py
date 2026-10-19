import sys
from pathlib import Path
import os
os.environ["YOLO_CONFIG_DIR"] = str(Path(__file__).resolve().parent.parent / "models")
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import config

def download_models():
    print(f"[INFO] Initializing model downloads to cache them locally...")
    
    # 1. CLIP
    print(f"[INFO] Downloading CLIP model: {config.clip_model}")
    from transformers import CLIPProcessor, CLIPModel
    try:
        model = CLIPModel.from_pretrained(config.clip_model)
        processor = CLIPProcessor.from_pretrained(config.clip_model)
        print("[INFO] CLIP model cached successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to download CLIP: {e}")

    # 2. YOLO
    print(f"[INFO] Downloading YOLO model: {config.yolo_model}")
    try:
        from ultralytics import YOLO
        yolo_path = config.models_dir / config.yolo_model
        model = YOLO(config.yolo_model)
        if not yolo_path.exists() and Path(config.yolo_model).exists():
            import shutil
            shutil.copy(config.yolo_model, yolo_path)
        print("[INFO] YOLO model cached successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to download YOLO: {e}")

    # 3. InsightFace
    print(f"[INFO] Downloading InsightFace model: {config.face_model}")
    try:
        from insightface.app import FaceAnalysis
        # Providers: use CUDA if available
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if config.device == 'cuda' else ['CPUExecutionProvider']
        app = FaceAnalysis(name=config.face_model, root=str(config.models_dir), providers=providers)
        app.prepare(ctx_id=0 if config.device == 'cuda' else -1, det_size=(640, 640))
        print("[INFO] InsightFace model cached successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to download InsightFace: {e}")

if __name__ == "__main__":
    download_models()
