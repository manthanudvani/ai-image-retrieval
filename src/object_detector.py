from ultralytics import YOLO
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any

class ObjectDetector:
    def __init__(self, model_name: str, models_dir: Path):
        print(f"[INFO] Loading YOLO Model ({model_name})...")
        model_path = models_dir / model_name
        if not model_path.exists():
             self.model = YOLO(model_name)
        else:
             self.model = YOLO(str(model_path))

    def detect(self, image_path: str, conf_threshold: float = 0.25) -> List[Dict[str, Any]]:
        results = self.model(image_path, conf=conf_threshold, verbose=False)
        objects = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = result.names[cls_id]
                objects.append({
                    "label": label,
                    "confidence": conf
                })
        return objects
