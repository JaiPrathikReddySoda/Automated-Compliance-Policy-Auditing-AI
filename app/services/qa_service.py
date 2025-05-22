from __future__ import annotations

import logging
from typing import Dict, List, Optional

from app.models.model_loader import get_model
from app.services.retriever import get_retriever

logger = logging.getLogger(__name__)


class QAService:
    """High-level QA service combining retrieval and answer generation."""
    
    def __init__(self) -> None:
        """Initialize the QA service with model and retriever."""
        self.model = get_model()
        self.retriever = get_retriever()
    
    def answer_question(
        self,
        question: str,
        top_k: int = 4,
    ) -> Dict[str, any]:
        """
        Answer a question using retrieved context.
        
        Args:
            question: The question to answer
            top_k: Number of context chunks to retrieve
            
        Returns:
            Dict containing answer, citations, and metadata
        """
        try:
            # Retrieve relevant context
            context_chunks = self.retriever.get_top_k(question, k=top_k)
            
            if not context_chunks:
                return {
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "citations": [],
                    "metadata": {"confidence": 0.0},
                }
            
            # Combine context chunks
            context = "\n\n".join(chunk for chunk, _ in context_chunks)
            
            # Generate answer
            answer, metadata = self.model.answer(question, context)
            
            # Prepare citations
            citations = [
                {
                    "text": chunk,
                    "score": float(score),
                }
                for chunk, score in context_chunks
            ]
            
            return {
                "answer": answer,
                "citations": citations,
                "metadata": metadata,
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            raise


# Global QA service instance
qa_service: Optional[QAService] = None


def get_qa_service() -> QAService:
    """Get or create the global QA service instance."""
    global qa_service
    if qa_service is None:
        qa_service = QAService()
    return qa_service 