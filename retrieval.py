from rank_bm25 import BM25Okapi
from typing import List, Tuple
from utils import tokenize

def build_index(chunks: List[dict]) -> BM25Okapi:
    tokenized = [tokenize(chunk["text"]) for chunk in chunks]
    return BM25Okapi(tokenized)

def search(query: str, index: BM25Okapi, chunks: List[dict], top_k: int = 20) -> List[Tuple[dict, float]]:
    tokens = tokenize(query)
    scores = index.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    results = []
    for i in top_indices:
        chunk = chunks[i].copy()
        chunk["bm25_score"] = float(scores[i])
        results.append(chunk)
    return results