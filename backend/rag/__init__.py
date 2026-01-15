"""
RAG AI Assistant - Core Module
"""

from .chunker import chunk_text
from .embedder import embed_chunks, embed_query, embed_query_hyde
from .llm import generate_response, generate_answer_hyde
from .parser import _extract_pdf_text, extract_text_from_file
from .retriever import search, search_with_query, get_relevant_context
from .vector_store import get_vector_store, vector_store
from .main import process_document, process_query, list_user_documents
from .mongodb import mongodb
from .auth import AuthManager, bcrypt, jwt, jwt_required, get_jwt_identity, init_auth
from .logging_config import setup_logger

__version__ = "1.0.0"
__all__ = [
    # Core functions
    "chunk_text",
    "embed_chunks",
    "embed_query",
    "embed_query_hyde",
    "generate_response",
    "generate_answer_hyde",
    "_extract_pdf_text",
    "extract_text_from_file",
    "search",
    "search_with_query",
    "get_relevant_context",
    "process_document",
    "process_query",
    "list_user_documents",
    
    # Database
    "get_vector_store",
    "vector_store",
    "mongodb",
    
    # Authentication
    "AuthManager",
    "init_auth",
    "bcrypt",
    "jwt",
    "jwt_required",
    "get_jwt_identity",
    
    # Logging
    "setup_logger",
]