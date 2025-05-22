from __future__ import annotations

import logging
from typing import Dict, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, HttpUrl

from app.services.indexing import get_indexing_service
from app.services.qa_service import get_qa_service

logger = logging.getLogger(__name__)
router = APIRouter()


class Question(BaseModel):
    """Question model for the /ask endpoint."""
    question: str


class Answer(BaseModel):
    """Answer model for the /ask endpoint."""
    answer: str
    citations: list[Dict[str, any]]
    metadata: Dict[str, any]


class UploadResponse(BaseModel):
    """Response model for the /upload endpoint."""
    message: str
    chunks_indexed: int


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@router.post("/ask", response_model=Answer)
async def ask_question(question: Question) -> Answer:
    """
    Ask a question about the indexed documents.
    
    Args:
        question: The question to answer
        
    Returns:
        Answer with citations and metadata
    """
    try:
        qa_service = get_qa_service()
        result = qa_service.answer_question(question.question)
        return Answer(**result)
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: Optional[UploadFile] = File(None),
    url: Optional[HttpUrl] = None,
) -> UploadResponse:
    """
    Upload and index a document.
    
    Args:
        file: File to upload (PDF or TXT)
        url: URL to index
        
    Returns:
        Upload response with status
    """
    try:
        if not file and not url:
            raise HTTPException(
                status_code=400,
                detail="Either file or url must be provided",
            )
        
        indexing_service = get_indexing_service()
        
        if file:
            # Save uploaded file
            file_path = f"temp/{file.filename}"
            with open(file_path, "wb") as f:
                f.write(await file.read())
            
            # Index file
            chunks = indexing_service.process_file(file_path)
            
            return UploadResponse(
                message=f"Successfully indexed {file.filename}",
                chunks_indexed=len(chunks),
            )
        else:
            # Index URL
            chunks = indexing_service.process_url(str(url))
            
            return UploadResponse(
                message=f"Successfully indexed {url}",
                chunks_indexed=len(chunks),
            )
            
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 