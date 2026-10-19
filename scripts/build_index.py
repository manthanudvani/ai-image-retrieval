import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import config
from src.preprocessing import Preprocessor

def main():
    preprocessor = Preprocessor()
    
    # Process COCO raw dataset
    if config.raw_dir.exists():
        preprocessor.process_directory(config.raw_dir)
        
    # Process personal images
    if config.personal_dir.exists():
        preprocessor.process_directory(config.personal_dir)

if __name__ == "__main__":
    main()
