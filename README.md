# AI-Powered Semantic Image Retrieval System

## Project Overview
This is a comprehensive machine learning project that allows users to search through a large image collection using natural language queries. It leverages state-of-the-art models for semantic understanding, object detection, and facial recognition to provide highly relevant search results.

## Features
- **Semantic Search**: Use natural language to find images (e.g., "dog playing in grass"). Powered by OpenAI's CLIP model.
- **Object Detection**: Extracts explicit objects from images using YOLO for object-aware retrieval.
- **Facial Recognition**: Can identify specific individuals (e.g., "me wearing a black shirt") using InsightFace embeddings.
- **Vector Search Engine**: Uses FAISS for incredibly fast similarity search.
- **Hybrid Ranking**: Dynamically combines CLIP scores, YOLO object matches, and Face embeddings for accurate results.
- **Modern UI**: A fully responsive web interface built with HTML/CSS and FastAPI.

## Architecture
1. **Preprocessing Pipeline**: Images are ingested, resized, and processed through CLIP (for semantic embeddings), YOLO (for object labels), and InsightFace (for facial embeddings).
2. **Vector Store**: Image embeddings are stored in a FAISS index.
3. **Metadata DB**: Object labels, face presence, and image metadata are stored in a SQLite database.
4. **Search Engine**: User queries are embedded via CLIP, compared in FAISS, and then re-ranked using the hybrid ranker.

## Installation
1. Clone the repository.
2. Create a virtual environment: `python -m venv .venv`
3. Activate it: `.venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121`
5. Run the application: `python run.py`

## Adding Personal Images
To search your own images, place them in the `data/personal/` directory.

## Adding Reference Faces
To enable the "me" search (e.g., "me at the beach"), place a few clear, front-facing photos of yourself in the `data/reference_faces/` directory.

## Methodology
- **Why CLIP?**: CLIP natively understands the mapping between images and text, allowing for zero-shot natural language search.
- **Why FAISS?**: It allows for extremely fast nearest-neighbor search in high-dimensional vector spaces.
- **Why Hybrid Ranking?**: CLIP is great for general semantics, but YOLO ensures specific objects are present, and InsightFace guarantees identity. Combining them yields the most robust system.
