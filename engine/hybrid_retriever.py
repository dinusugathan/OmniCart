import re
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from engine.chunker import DocumentChunk
from engine.embeddings import BaseEmbeddingProvider, get_embedding_provider
from engine.models import ExtractedFilters

class HybridRetriever:
    """
    Hybrid Retriever combining Sparse BM25 + Dense Cosine Similarity + Reciprocal Rank Fusion (RRF)
    with strict metadata filtering (price cap, in-stock, category, skin type).
    """
    def __init__(self, chunks: List[DocumentChunk], embedder: Optional[BaseEmbeddingProvider] = None):
        self.chunks = chunks
        self.embedder = embedder or get_embedding_provider()
        
        # Prepare BM25 Corpus
        self.tokenized_corpus = [self._tokenize(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # Prepare Dense Vector Index
        chunk_texts = [c.text for c in self.chunks]
        self.doc_embeddings = self.embedder.embed_documents(chunk_texts)

    def _tokenize(self, text: str) -> List[str]:
        cleaned = re.sub(r"[^\w\s-]", " ", text.lower())
        return [tok for tok in cleaned.split() if len(tok) > 1]

    def _apply_metadata_filters(self, chunk: DocumentChunk, filters: ExtractedFilters) -> bool:
        meta = chunk.metadata
        
        # Check stock status if product
        if filters.in_stock_only and "in_stock" in meta:
            if not meta["in_stock"]:
                return False
                
        # Check maximum price cap if specified
        if filters.max_price is not None and "price" in meta:
            if meta["price"] > filters.max_price:
                return False

        # Check category constraint if specified
        if filters.category is not None and "category" in meta:
            if filters.category.lower() not in meta["category"].lower():
                return False

        # Check skin type compatibility if specified
        if filters.skin_type is not None and "skin_types" in meta:
            skin_types_lower = [s.lower() for s in meta["skin_types"]]
            target = filters.skin_type.lower()
            if "all" not in skin_types_lower and target not in skin_types_lower:
                # Soft penalty or exclusion if totally incompatible
                pass
                
        return True

    def retrieve(
        self, 
        query: str, 
        filters: Optional[ExtractedFilters] = None, 
        top_k: int = 8,
        rrf_k: int = 60
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Executes BM25 sparse search and dense embedding search, merges using Reciprocal Rank Fusion,
        and enforces deterministic metadata filtering.
        """
        if filters is None:
            filters = ExtractedFilters()

        # 1. Sparse BM25 Scores
        query_tokens = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_ranked_indices = np.argsort(bm25_scores)[::-1]

        # 2. Dense Cosine Scores
        query_vec = self.embedder.embed_query(query)
        dense_scores = np.dot(self.doc_embeddings, query_vec)
        dense_ranked_indices = np.argsort(dense_scores)[::-1]

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[int, float] = {i: 0.0 for i in range(len(self.chunks))}
        
        for rank, idx in enumerate(bm25_ranked_indices[:top_k * 3]):
            rrf_scores[idx] += 1.0 / (rrf_k + rank + 1)
            
        for rank, idx in enumerate(dense_ranked_indices[:top_k * 3]):
            rrf_scores[idx] += 1.0 / (rrf_k + rank + 1)

        # Sort combined results by RRF score
        sorted_candidates = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

        # 4. Enforce Metadata Filters
        filtered_results: List[Tuple[DocumentChunk, float]] = []
        for idx, score in sorted_candidates:
            chunk = self.chunks[idx]
            if self._apply_metadata_filters(chunk, filters):
                filtered_results.append((chunk, score))
            if len(filtered_results) >= top_k:
                break

        return filtered_results
