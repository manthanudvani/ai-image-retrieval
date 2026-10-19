import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    print(f"\n{'='*50}")
    print(f"[STAGE] {description}")
    print(f"{'='*50}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Command failed: {e}")
        sys.exit(1)

def main():
    base_dir = Path(__file__).resolve().parent
    python_exe = sys.executable

    print("\n--- AI Image Retrieval System ---")
    print("Initializing components...")

    # 1. Download models
    run_command([python_exe, str(base_dir / "scripts" / "download_models.py")], "Downloading Models")

    # 2. Download dataset
    run_command([python_exe, str(base_dir / "scripts" / "download_dataset.py")], "Downloading Dataset")

    # 3. Build index
    run_command([python_exe, str(base_dir / "scripts" / "build_index.py")], "Building Index & Extracting Features")

    # 4. Start web app
    print(f"\n{'='*50}")
    print(f"[STAGE] Starting Web Application")
    print(f"{'='*50}")
    
    # We use uvicorn to run the FastAPI app
    try:
        subprocess.run([python_exe, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"])
    except KeyboardInterrupt:
        print("\n[INFO] Application stopped by user.")

if __name__ == "__main__":
    main()
