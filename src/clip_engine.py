import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from typing import Union, List

class CLIPEngine:
    def __init__(self, model_name: str, device: str):
        self.device = device
        print(f"[INFO] Loading CLIP Model ({model_name}) on {self.device}...")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def get_image_embeddings(self, images: Union[Image.Image, List[Image.Image]]) -> np.ndarray:
        if not isinstance(images, list):
            images = [images]
            
        with torch.no_grad():
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            image_features = self.model.get_image_features(**inputs)
            if hasattr(image_features, 'pooler_output'):
                image_features = image_features.pooler_output
            elif isinstance(image_features, tuple):
                image_features = image_features[1]
            # Normalize
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            return image_features.cpu().numpy()

    def get_text_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        if not isinstance(texts, list):
            texts = [texts]
            
        with torch.no_grad():
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            text_features = self.model.get_text_features(**inputs)
            if hasattr(text_features, 'pooler_output'):
                text_features = text_features.pooler_output
            elif isinstance(text_features, tuple):
                text_features = text_features[1]
            # Normalize
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            return text_features.cpu().numpy()
