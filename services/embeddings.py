"""
Harbor — Local embedding service using sentence-transformers.
Uses BGE-large-en-v1.5 (1024 dims) for $0 cost embeddings.
Model is loaded once and cached in memory (~1.3GB).
"""
import asyncio
from functools import lru_cache
from typing import List

import structlog

log = structlog.get_logger()

MODEL_PATH = "/mnt/data/models/bge-large-en-v1.5"
EMBEDDING_DIM = 1024


@lru_cache(maxsize=1)
def _get_model():
    """Load BGE model once, cache forever."""
    from sentence_transformers import SentenceTransformer
    log.info("harbor.embeddings.loading_model", path=MODEL_PATH)
    model = SentenceTransformer(MODEL_PATH)
    log.info("harbor.embeddings.model_loaded", dim=EMBEDDING_DIM)
    return model


async def get_embedding(text: str) -> List[float]:
    """Get embedding for a single text."""
    text = text.replace("\n", " ").strip()
    if len(text) > 8000:
        text = text[:8000]
    model = _get_model()
    # BGE recommends prepending "Represent this sentence:" for retrieval
    embedding = await asyncio.to_thread(
        model.encode, f"Represent this sentence: {text}", normalize_embeddings=True
    )
    return embedding.tolist()


async def get_embeddings_batch(texts: List[str], batch_size: int = 64) -> List[List[float]]:
    """Get embeddings for multiple texts efficiently."""
    model = _get_model()
    cleaned = [t.replace("\n", " ").strip()[:8000] for t in texts]
    prefixed = [f"Represent this sentence: {t}" for t in cleaned]
    embeddings = await asyncio.to_thread(
        model.encode, prefixed, normalize_embeddings=True, batch_size=batch_size
    )
    return [e.tolist() for e in embeddings]
