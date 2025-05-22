from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from optimum.intel import ORTModelForQuestionAnswering
from transformers import AutoTokenizer

from app.config import settings

logger = logging.getLogger(__name__)


class ModelLoader:
    """Loads and manages the quantized DeBERTa-v3 model for QA."""
    
    def __init__(self) -> None:
        """Initialize the model and tokenizer."""
        self.model_path = settings.MODEL_PATH
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the quantized ONNX model and tokenizer."""
        try:
            self.model = ORTModelForQuestionAnswering.from_pretrained(
                self.model_path,
                export=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME)
            logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def answer(self, question: str, context: str) -> tuple[str, dict]:
        """
        Generate an answer for the given question based on the context.
        
        Args:
            question: The question to answer
            context: The context to use for answering
            
        Returns:
            tuple[str, dict]: The answer and metadata including start/end positions
        """
        try:
            inputs = self.tokenizer(
                question,
                context,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,
            )
            
            outputs = self.model(**inputs)
            
            # Get the most likely answer span
            answer_start = outputs.start_logits.argmax().item()
            answer_end = outputs.end_logits.argmax().item()
            
            # Get the answer text
            answer = self.tokenizer.decode(
                inputs["input_ids"][0][answer_start:answer_end + 1]
            )
            
            metadata = {
                "start_pos": answer_start,
                "end_pos": answer_end,
                "confidence": float(
                    outputs.start_logits[0][answer_start] + 
                    outputs.end_logits[0][answer_end]
                ),
            }
            
            return answer, metadata
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise


# Global model instance
model_loader: Optional[ModelLoader] = None


def get_model() -> ModelLoader:
    """Get or create the global model instance."""
    global model_loader
    if model_loader is None:
        model_loader = ModelLoader()
    return model_loader 