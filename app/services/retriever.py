from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)


class Retriever:
    """FAISS HNSW vector store for semantic search."""
    
    def __init__(self) -> None:
        """Initialize the retriever with FAISS index and sentence transformer."""
        self.index_path = settings.INDEX_PATH
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.index: Optional[faiss.Index] = None
        self.documents: List[str] = []
        self._load_index()
    
    def _load_index(self) -> None:
        """Load the FAISS index and documents if they exist."""
        try:
            if self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path / "index.faiss"))
                with open(self.index_path / "documents.txt", "r") as f:
                    self.documents = [line.strip() for line in f]
                logger.info(f"Loaded index with {len(self.documents)} documents")
            else:
                self._create_index()
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise
    
    def _create_index(self) -> None:
        """Create a new FAISS HNSW index."""
        try:
            self.index = faiss.IndexHNSWFlat(
                settings.VECTOR_DIM,
                settings.HNSW_M,
                faiss.METRIC_INNER_PRODUCT
            )
            self.index.hnsw.efConstruction = settings.HNSW_EF_CONSTRUCTION
            logger.info("Created new FAISS HNSW index")
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise
    
    def add_documents(self, documents: List[str]) -> None:
        """
        Add documents to the index.
        
        Args:
            documents: List of document texts to add
        """
        try:
            # Get embeddings
            embeddings = self.embedding_model.encode(
                documents,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
            
            # Add to index
            self.index.add(embeddings)
            self.documents.extend(documents)
            
            # Save index and documents
            self._save_index()
            
            logger.info(f"Added {len(documents)} documents to index")
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def _save_index(self) -> None:
        """Save the index and documents to disk."""
        try:
            self.index_path.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(self.index_path / "index.faiss"))
            with open(self.index_path / "documents.txt", "w") as f:
                f.write("\n".join(self.documents))
            logger.info("Saved index and documents")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise
    
    def get_top_k(self, query: str, k: int = 4) -> List[Tuple[str, float]]:
        """
        Retrieve top-k most similar documents for a query.
        
        Args:
            query: The query text
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        try:
            # Get query embedding
            query_embedding = self.embedding_model.encode(
                query,
                convert_to_numpy=True,
            ).reshape(1, -1)
            
            # Search index
            scores, indices = self.index.search(query_embedding, k)
            
            # Return results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1:  # Valid index
                    results.append((self.documents[idx], float(score)))
            
            return results
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            raise


# Global retriever instance
retriever: Optional[Retriever] = None


def get_retriever() -> Retriever:
    """Get or create the global retriever instance."""
    global retriever
    if retriever is None:
        retriever = Retriever()
    return retriever 