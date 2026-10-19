import sys
from pathlib import Path
import json

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import config
from src.search_engine import SearchEngine

def main():
    print("[INFO] Starting Evaluation...")
    engine = SearchEngine()
    
    # We will use a mock evaluation set if evaluation/queries.json doesn't exist
    eval_dir = config.base_dir / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    queries_file = eval_dir / "queries.json"
    
    if not queries_file.exists():
        print(f"[INFO] Creating mock evaluation file at {queries_file}")
        mock_data = [
            {"query": "a dog", "relevant_images": ["000000000010", "000000000015"]},
            {"query": "person", "relevant_images": ["000000000020"]},
        ]
        with open(queries_file, 'w') as f:
            json.dump(mock_data, f, indent=4)
            
    with open(queries_file, 'r') as f:
        queries = json.load(f)
        
    print(f"[INFO] Evaluating {len(queries)} queries...")
    
    # Very basic evaluation
    total_p5 = 0
    total_p10 = 0
    
    for item in queries:
        q = item['query']
        relevant = set([str(x) for x in item.get('relevant_images', [])])
        if not relevant:
            continue
            
        results = engine.search(q, top_k=10)
        retrieved_ids = [r['image_id'] for r in results]
        
        # P@5
        p5_hits = sum(1 for id in retrieved_ids[:5] if id in relevant)
        p5 = p5_hits / 5.0
        total_p5 += p5
        
        # P@10
        p10_hits = sum(1 for id in retrieved_ids[:10] if id in relevant)
        p10 = p10_hits / 10.0
        total_p10 += p10
        
        print(f"Query: '{q}' -> P@5: {p5:.2f}, P@10: {p10:.2f}")
        
    print(f"\n[SUMMARY] Mean P@5:  {total_p5/len(queries):.2f}")
    print(f"[SUMMARY] Mean P@10: {total_p10/len(queries):.2f}")
    
if __name__ == "__main__":
    main()
