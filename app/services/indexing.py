from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import pdfplumber
import requests
from bs4 import BeautifulSoup
from tika import parser
from trafilatura import extract

from app.config import settings
from app.services.retriever import get_retriever

logger = logging.getLogger(__name__)


class IndexingService:
    """Service for processing and indexing documents."""
    
    def __init__(self) -> None:
        """Initialize the indexing service."""
        self.retriever = get_retriever()
    
    def process_file(self, file_path: Path) -> List[str]:
        """
        Process a file and extract text chunks.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of text chunks
        """
        try:
            if file_path.suffix.lower() == ".pdf":
                return self._process_pdf(file_path)
            elif file_path.suffix.lower() == ".txt":
                return self._process_txt(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise
    
    def process_url(self, url: str) -> List[str]:
        """
        Process a URL and extract text chunks.
        
        Args:
            url: URL to process
            
        Returns:
            List of text chunks
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Try trafilatura first (better for news/articles)
            text = extract(response.text)
            if text:
                return self._chunk_text(text)
            
            # Fallback to BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            return self._chunk_text(text)
            
        except Exception as e:
            logger.error(f"Error processing URL {url}: {e}")
            raise
    
    def _process_pdf(self, file_path: Path) -> List[str]:
        """Process a PDF file and extract text chunks."""
        try:
            # Try pdfplumber first
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            
            if not text.strip():
                # Fallback to tika
                raw = parser.from_file(str(file_path))
                text = raw["content"]
            
            return self._chunk_text(text)
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            raise
    
    def _process_txt(self, file_path: Path) -> List[str]:
        """Process a text file and extract chunks."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            return self._chunk_text(text)
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {e}")
            raise
    
    def _chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            chunk_size: Target size of each chunk
            
        Returns:
            List of text chunks
        """
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), chunk_size // 2):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def index_document(self, source: str) -> None:
        """
        Process and index a document from file or URL.
        
        Args:
            source: Path to file or URL
        """
        try:
            # Check if source is URL
            if source.startswith(("http://", "https://")):
                chunks = self.process_url(source)
            else:
                chunks = self.process_file(Path(source))
            
            # Add to index
            self.retriever.add_documents(chunks)
            
            logger.info(f"Indexed {len(chunks)} chunks from {source}")
            
        except Exception as e:
            logger.error(f"Error indexing document {source}: {e}")
            raise


# Global indexing service instance
indexing_service: Optional[IndexingService] = None


def get_indexing_service() -> IndexingService:
    """Get or create the global indexing service instance."""
    global indexing_service
    if indexing_service is None:
        indexing_service = IndexingService()
    return indexing_service 