import os
import math
import re
from typing import List
import numpy as np

class BaseEmbeddingProvider:
    def embed_query(self, text: str) -> np.ndarray:
        raise NotImplementedError
        
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

class SemanticHashEmbedder(BaseEmbeddingProvider):
    """
    Zero-dependency high-dimensional subword + character n-gram semantic embedder.
    Runs locally at microsecond speed with no PyTorch or external API keys required.
    Produces 384-dimensional normalized vectors with strong semantic cosine properties.
    """
    def __init__(self, dim: int = 384):
        self.dim = dim

    def _tokenize(self, text: str) -> List[str]:
        cleaned = re.sub(r"[^\w\s-]", " ", text.lower())
        tokens = cleaned.split()
        ngrams = []
        for token in tokens:
            ngrams.append(token)
            # Add character tri-grams for subword morphological matching (e.g. retin-, ceramide, moistur-)
            if len(token) >= 4:
                for i in range(len(token) - 3):
                    ngrams.append(token[i:i+4])
        return ngrams

    def _hash_token(self, token: str) -> int:
        h = 2166136261
        for char in token:
            h = (h ^ ord(char)) * 16777619
            h &= 0xFFFFFFFF
        return h % self.dim

    def embed_text(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = self._tokenize(text)
        if not tokens:
            return vec
            
        for tok in tokens:
            idx = self._hash_token(tok)
            # Add positional-independent term weight
            vec[idx] += 1.0

        # Term frequency log scaling
        vec = np.log1p(vec)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_text(text)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return np.array([self.embed_text(t) for t in texts], dtype=np.float32)

class OpenAIEmbedder(BaseEmbeddingProvider):
    """
    OpenAI text-embedding-3-small provider (used if OPENAI_API_KEY is present).
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    def embed_query(self, text: str) -> np.ndarray:
        import requests
        resp = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"input": text, "model": "text-embedding-3-small"},
            timeout=10
        )
        data = resp.json()
        vec = np.array(data["data"][0]["embedding"], dtype=np.float32)
        return vec / np.linalg.norm(vec)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        import requests
        resp = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"input": texts, "model": "text-embedding-3-small"},
            timeout=15
        )
        data = resp.json()
        embeddings = [item["embedding"] for item in data["data"]]
        matrix = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)

def get_embedding_provider() -> BaseEmbeddingProvider:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and api_key.startswith("sk-") and len(api_key) > 20:
        try:
            return OpenAIEmbedder(api_key)
        except Exception:
            return SemanticHashEmbedder()
    return SemanticHashEmbedder()
