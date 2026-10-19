import os
import shutil
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
import sys

# Add src to path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import config

def download_coco_subset():
    print(f"[INFO] Downloading COCO dataset subset (max {config.max_images} images)...")
    
    raw_dir = config.raw_dir
    os.makedirs(raw_dir, exist_ok=True)
    
    existing_images = list(raw_dir.glob("*.jpg"))
    if len(existing_images) >= config.max_images:
        print(f"[INFO] Found {len(existing_images)} images already. Skipping download.")
        return

    print(f"[INFO] Fetching images into {raw_dir} directly...")
    
    import urllib.request
    
    count = 0
    for i in tqdm(range(1, config.max_images + 1), desc="Downloading images"):
        img_id = 391895 + i  # some arbitrary COCO image ids
        url = f"http://images.cocodataset.org/val2017/{img_id:012d}.jpg"
        img_path = raw_dir / f"{img_id:012d}.jpg"
        
        if not img_path.exists():
            try:
                urllib.request.urlretrieve(url, img_path)
                count += 1
            except Exception as e:
                pass # skip if id doesn't exist
                
    print(f"[INFO] Dataset download complete. {count} images saved.")

if __name__ == "__main__":
    download_coco_subset()
