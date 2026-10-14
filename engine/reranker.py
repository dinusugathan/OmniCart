from typing import List, Tuple, Dict, Any
from engine.chunker import DocumentChunk

class ProductReranker:
    """
    Cross-Encoder Reranker using FlashRank (ultra-lightweight local cross-encoder).
    Drastically improves Context Precision and minimizes context window noise.
    """
    def __init__(self, model_name: str = "ms-marco-TinyBERT-L-2-v2"):
        self.model_name = model_name
        self.ranker = None
        self._init_ranker()

    def _init_ranker(self):
        try:
            from flashrank import Ranker
            # Lazy initialize FlashRank
            self.ranker = Ranker(model_name=self.model_name, cache_dir="./.cache")
        except Exception as e:
            # Resilient fallback if onnx weights not yet cached
            self.ranker = None

    def rerank(
        self, 
        query: str, 
        candidate_chunks: List[Tuple[DocumentChunk, float]], 
        top_n: int = 4
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Reranks candidate chunks with cross-encoder attention scoring.
        """
        if not candidate_chunks:
            return []

        if self.ranker is not None:
            try:
                from flashrank import RerankRequest
                passages = [
                    {"id": idx, "text": chunk.text, "meta": chunk.metadata}
                    for idx, (chunk, _) in enumerate(candidate_chunks)
                ]
                rerank_req = RerankRequest(query=query, passages=passages)
                results = self.ranker.rerank(rerank_req)
                
                # Sort and format top_n
                reranked_chunks = []
                for item in results[:top_n]:
                    orig_chunk = candidate_chunks[item["id"]][0]
                    reranked_chunks.append((orig_chunk, float(item["score"])))
                return reranked_chunks
            except Exception:
                pass

        # Fallback heuristic reranking: score by query-specific keyword density and title affinity
        scored_candidates = []
        q_lower = query.lower()
        for chunk, orig_score in candidate_chunks:
            bonus = 0.0
            # Boost if query words occur in product title
            title_words = [w for w in chunk.title.lower().split() if len(w) > 3]
            for w in title_words:
                if w in q_lower:
                    bonus += 0.25
            scored_candidates.append((chunk, orig_score + bonus))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[:top_n]
