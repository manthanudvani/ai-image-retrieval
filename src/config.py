import yaml
from pathlib import Path
import torch

class Config:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.load_config()

        # Paths
        self.base_dir = Path(__file__).resolve().parent.parent
        self.data_dir = self.base_dir / "data"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.personal_dir = self.data_dir / "personal"
        self.reference_faces_dir = self.data_dir / "reference_faces"
        
        self.models_dir = self.base_dir / "models"
        
        self.storage_dir = self.base_dir / "storage"
        self.faiss_dir = self.storage_dir / "faiss"
        self.embeddings_dir = self.storage_dir / "embeddings"
        self.metadata_dir = self.storage_dir / "metadata"

        self._create_directories()

    def load_config(self):
        with open(self.config_path, "r") as f:
            self._config = yaml.safe_load(f)

    def _create_directories(self):
        directories = [
            self.raw_dir, self.processed_dir, self.personal_dir, self.reference_faces_dir,
            self.models_dir, self.faiss_dir, self.embeddings_dir, self.metadata_dir
        ]
        for d in directories:
            d.mkdir(parents=True, exist_ok=True)

    @property
    def dataset_name(self):
        return self._config.get("dataset", {}).get("name", "coco")

    @property
    def max_images(self):
        return self._config.get("dataset", {}).get("max_images", 2000)

    @property
    def top_k(self):
        return self._config.get("retrieval", {}).get("top_k", 10)

    @property
    def clip_model(self):
        return self._config.get("models", {}).get("clip", "openai/clip-vit-base-patch32")

    @property
    def yolo_model(self):
        return self._config.get("models", {}).get("yolo", "yolov8n.pt")

    @property
    def face_model(self):
        return self._config.get("models", {}).get("face", "buffalo_l")

    @property
    def clip_weight(self):
        return self._config.get("weights", {}).get("clip_weight", 0.6)

    @property
    def face_weight(self):
        return self._config.get("weights", {}).get("face_weight", 0.25)

    @property
    def object_weight(self):
        return self._config.get("weights", {}).get("object_weight", 0.15)

    @property
    def device(self):
        auto = self._config.get("device", {}).get("auto", True)
        if auto:
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cpu"

config = Config()
