import re
import numpy as np
from typing import List, Dict, Optional
from .logging_config import logger
from .vector_store import get_vector_store
from .embedder import embed_query

# Global reranker model
_reranker = None

def _load_reranker() -> Optional[object]:
    """Lazy load reranker model"""
    global _reranker
    
    # Check if reranker is disabled
    import os
    if os.getenv("DISABLE_RERANKER", "0") == "1":
        logger.info("Reranker disabled via DISABLE_RERANKER=1")
        return None
    
    if _reranker is not None:
        return _reranker
    
    try:
        logger.info("Loading reranker model...")
        from sentence_transformers import CrossEncoder
        
        # Use a small, efficient model
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("Reranker model loaded successfully")
        return _reranker
    
    except ImportError as e:
        logger.warning(f"sentence_transformers not available; skipping reranker: {e}")
        logger.info("Install with: pip install sentence-transformers")
        _reranker = None
        return None
    
    except Exception as e:
        logger.error(f"Failed to load reranker model: {e}")
        _reranker = None
        return None

def _calculate_keyword_score(query: str, text: str) -> float:
    """
    Calculate keyword overlap score between query and text
    
    Args:
        query: Search query
        text: Document text
    
    Returns:
        Keyword score (0.0 to 1.0)
    """
    # Extract words from query and text
    query_words = set(re.findall(r'\b\w+\b', query.lower()))
    text_words = set(re.findall(r'\b\w+\b', text.lower()))
    
    if not query_words:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = query_words.intersection(text_words)
    union = query_words.union(text_words)
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)

def _apply_reranker(query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    Apply reranking to search candidates
    
    Args:
        query: Search query
        candidates: List of candidate documents
        top_k: Number of top results to return
    
    Returns:
        Reranked list of candidates
    """
    reranker = _load_reranker()
    
    if not reranker or not candidates:
        return candidates[:top_k]
    
    try:
        # Prepare query-document pairs for reranking
        pairs = [(query, candidate.get("text", "")) for candidate in candidates]
        
        # Get reranking scores
        rerank_scores = reranker.predict(pairs)
        
        # Update candidates with rerank scores
        for i, candidate in enumerate(candidates):
            candidate["rerank_score"] = float(rerank_scores[i])
        
        # Sort by rerank score (descending)
        reranked = sorted(candidates, key=lambda x: x.get("rerank_score", 0), reverse=True)
        
        logger.info(f"Applied reranking to {len(candidates)} candidates")
        return reranked[:top_k]
    
    except Exception as e:
        logger.warning(f"Reranking failed: {e}. Using original ranking.")
        return candidates[:top_k]

def search(query: str, query_embedding: List[float], user_id: str = None, 
           document_ids: List[str] = None, k: int = 2, alpha: float = 1.0) -> List[Dict]:
    """
    Hybrid search combining vector similarity and keyword matching - STRICT k=2 (Rule 4)
    """
    try:
        logger.info(f"Starting hybrid search for query: {query[:50]}...")
        
        if not query_embedding:
            logger.warning("Empty query embedding provided")
            return []
        
        # Get vector store instance
        vector_store = get_vector_store()
        if not vector_store:
            logger.error("Vector store not available")
            return []
        
        # Step 1: Vector search - Limit to k (Rule 4)
        vector_results = vector_store.search_vectors(
            query_embedding=query_embedding,
            top_k=k,  
            user_id=user_id,
            document_ids=document_ids
        )
        
        if not vector_results:
            logger.warning("No vector search results found")
            return []
        
        logger.info(f"Vector search returned {len(vector_results)} candidates")
        
        # Format results
        formatted_results = []
        for result in vector_results[:k]:
            formatted = {
                "text": result.get("text", ""),
                "source_file": result.get("source_file", "unknown"),
                "document_id": result.get("document_id", "unknown"),
                "score": result.get("score", 0.0),
                "metadata": result.get("metadata", {})
            }
            formatted_results.append(formatted)
        
        logger.info(f"Search completed: found {len(formatted_results)} relevant chunks")
        return formatted_results
    
    except Exception as e:
        logger.exception(f"Error during search: {e}")
        return []

def search_with_query(query: str, user_id: str = None, 
                     document_ids: List[str] = None, k: int = 2) -> List[Dict]:
    """
    Simplified search function that handles embedding internally - STRICT k=2 (Rule 4)
    """
    try:
        # Generate query embedding
        query_embedding = embed_query(query)
        
        if not query_embedding:
            logger.error("Failed to generate query embedding")
            return []
        
        # Perform search
        return search(query, query_embedding, user_id, document_ids, k)
    
    except Exception as e:
        logger.exception(f"Error in search_with_query: {e}")
        return []

def get_relevant_context(query: str, user_id: str = None, 
                        document_ids: List[str] = None, 
                        max_chars: int = 4000) -> str:
    """
    Get relevant context for a query - STRICT max 2 chunks (Rule 4)
    """
    # Search for relevant chunks - STRICT k=2
    chunks = search_with_query(query, user_id, document_ids, k=2)
    
    if not chunks:
        return ""
    
    # Sort by score (already sorted by search)
    relevant_chunks = []
    total_chars = 0
    
    for chunk in chunks:
        chunk_text = chunk.get("text", "")
        chunk_length = len(chunk_text)
        
        # Check if adding this chunk would exceed max_chars
        if total_chars + chunk_length > max_chars:
            # Try to add part of the chunk
            remaining = max_chars - total_chars
            if remaining > 100:  # Only add if we have significant space
                chunk_text = chunk_text[:remaining] + "..."
                relevant_chunks.append({
                    "text": chunk_text,
                    "source": chunk.get("source_file", "unknown"),
                    "score": chunk.get("score", 0.0)
                })
            break
        
        relevant_chunks.append({
            "text": chunk_text,
            "source": chunk.get("source_file", "unknown"),
            "score": chunk.get("score", 0.0)
        })
        total_chars += chunk_length
    
    # Format context with sources
    context_parts = []
    for i, chunk in enumerate(relevant_chunks):
        context_parts.append(f"[Source: {chunk['source']}, Relevance: {chunk['score']:.2f}]")
        context_parts.append(chunk['text'])
        context_parts.append("---")
    
    context = "\n".join(context_parts).rstrip("---\n").strip()
    
    logger.info(f"Created context of {len(context)} characters from {len(relevant_chunks)} chunks")
    
    return context