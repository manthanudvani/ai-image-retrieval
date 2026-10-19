from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import config
from src.search_engine import SearchEngine

app = FastAPI(title="AI Image Retrieval System")
templates_dir = Path(__file__).resolve().parent / "templates"
static_dir = Path(__file__).resolve().parent / "static"
templates = Jinja2Templates(directory=str(templates_dir))
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Mount data directories to serve images directly
app.mount("/data/raw", StaticFiles(directory=str(config.raw_dir)), name="raw_data")
app.mount("/data/personal", StaticFiles(directory=str(config.personal_dir)), name="personal_data")

engine = None

@app.on_event("startup")
async def startup_event():
    global engine
    engine = SearchEngine()

import random

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    raw_files = list(config.raw_dir.glob("*.jpg"))
    # The user requested ALL images to be visible on the homepage
    gallery_images = [f"/data/raw/{f.name}" for f in raw_files]
    return templates.TemplateResponse(request=request, name="index.html", context={"results": [], "query": "", "gallery_images": gallery_images, "total_count": len(raw_files)})

@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...), top_k: int = Form(config.top_k)):
    global engine
    try:
        results = engine.search(query, top_k=top_k)
    except Exception as e:
        print(f"[ERROR] Search failed: {e}")
        results = []

    # Map file_paths to static URLs
    for r in results:
        file_path = Path(r['metadata']['file_path'])
        if "personal" in file_path.parts:
            r['url'] = f"/data/personal/{file_path.name}"
        else:
            r['url'] = f"/data/raw/{file_path.name}"
            
    return templates.TemplateResponse(request=request, name="index.html", context={"results": results, "query": query})
