import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Any

class MetadataStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS images (
                    image_id TEXT PRIMARY KEY,
                    file_path TEXT,
                    filename TEXT,
                    width INTEGER,
                    height INTEGER,
                    format TEXT,
                    objects TEXT,
                    has_faces BOOLEAN
                )
            ''')
            conn.commit()

    def add_image(self, image_id: str, file_path: str, metadata: Dict[str, Any]):
        objects_json = json.dumps(metadata.get('objects', []))
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO images 
                (image_id, file_path, filename, width, height, format, objects, has_faces)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                image_id, 
                file_path, 
                Path(file_path).name,
                metadata.get('width', 0),
                metadata.get('height', 0),
                metadata.get('format', ''),
                objects_json,
                metadata.get('has_faces', False)
            ))
            conn.commit()

    def get_image(self, image_id: str) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM images WHERE image_id = ?', (image_id,))
            row = cursor.fetchone()
            if row:
                return {
                    'image_id': row[0],
                    'file_path': row[1],
                    'filename': row[2],
                    'width': row[3],
                    'height': row[4],
                    'format': row[5],
                    'objects': json.loads(row[6]),
                    'has_faces': bool(row[7])
                }
        return {}

    def get_all_images(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM images')
            rows = cursor.fetchall()
            results = []
            for row in rows:
                results.append({
                    'image_id': row[0],
                    'file_path': row[1],
                    'filename': row[2],
                    'width': row[3],
                    'height': row[4],
                    'format': row[5],
                    'objects': json.loads(row[6]),
                    'has_faces': bool(row[7])
                })
            return results
