#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import structlog
from app.services.indexing import get_indexing_service

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
)
logger = structlog.get_logger()


def find_documents(docs_dir: Path) -> List[Path]:
    """
    Find all supported documents in a directory.
    
    Args:
        docs_dir: Directory to search
        
    Returns:
        List of document paths
    """
    documents = []
    for ext in (".pdf", ".txt"):
        documents.extend(docs_dir.glob(f"**/*{ext}"))
    return documents


def main() -> None:
    """Main entry point for the indexing script."""
    parser = argparse.ArgumentParser(description="Build document index")
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("docs"),
        help="Directory containing documents to index",
    )
    parser.add_argument(
        "--s3-bucket",
        type=str,
        help="S3 bucket containing documents to index",
    )
    args = parser.parse_args()
    
    indexing_service = get_indexing_service()
    
    if args.s3_bucket:
        # TODO: Implement S3 indexing
        logger.warning("S3 indexing not implemented yet")
        return
    
    # Index local documents
    docs_dir = args.docs_dir
    if not docs_dir.exists():
        logger.error(f"Docs directory {docs_dir} does not exist")
        return
    
    documents = find_documents(docs_dir)
    if not documents:
        logger.warning(f"No documents found in {docs_dir}")
        return
    
    logger.info(f"Found {len(documents)} documents to index")
    
    for doc_path in documents:
        try:
            logger.info(f"Indexing {doc_path}")
            indexing_service.index_document(str(doc_path))
        except Exception as e:
            logger.error(f"Failed to index {doc_path}: {e}")
    
    logger.info("Indexing complete")


if __name__ == "__main__":
    main() 