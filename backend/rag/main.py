import os
import json
import uuid
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

from rag.parser import _extract_pdf_text
from rag.chunker import chunk_text
from rag.embedder import embed_chunks, embed_query_hyde
from rag.vector_store import vector_store, get_vector_store
from rag.logging_config import setup_logger
from rag.retriever import search
from rag.llm import generate_response
from rag.mongodb import mongodb

load_dotenv()
logger = setup_logger(__name__)

def initialize_system():
    """Initialize the RAG system components"""
    try:
        # Ensure vector store is initialized
        store = get_vector_store()
        if store:
            logger.info("MongoDB RAG system initialized successfully.")
            return True
        else:
            logger.error("Failed to initialize vector store")
            return False
    except Exception as e:
        logger.exception("Failed to initialize MongoDB RAG system.")
        return False

def process_document(pdf_file, user_id: str) -> Dict:
    """Process a PDF document and store it in the vector database"""
    try:
        logger.info(f"Starting PDF processing for user: {user_id}")

        # Determine filename and get content
        source_file = None
        content = None
        
        if isinstance(pdf_file, str) and os.path.exists(pdf_file):
            source_file = os.path.basename(pdf_file)
            with open(pdf_file, "rb") as f:
                content = f.read()
        elif hasattr(pdf_file, 'read'):
            # Handle file-like objects
            source_file = getattr(pdf_file, "filename", getattr(pdf_file, "name", "unknown_file.pdf"))
            # Reset pointer if possible
            if hasattr(pdf_file, "seek"):
                try:
                    pdf_file.seek(0)
                except:
                    pass
            content = pdf_file.read()
        else:
            content = pdf_file  # Assume it's already bytes

        if not content:
            logger.warning("No content found in the provided PDF file.")
            return {"success": False, "error": "No content found."}

        # Extract text
        text = _extract_pdf_text(content)
        
        # Fallback to Gemini Vision/PDF if text extraction failed (empty or very short)
        if not text or len(text.strip()) < 50:
            logger.warning("Standard text extraction low/failed. Trying Gemini Vision/PDF extraction...")
            try:
                from rag.llm import generate_with_fallback
                import google.genai.types as types
                
                logger.info("Sending PDF to Gemini for extraction...")
                
                # Prepare PDF part
                pdf_part = types.Part.from_bytes(
                    data=content,
                    mime_type="application/pdf"
                )
                
                prompt = "Please extract all text from this PDF document explicitly. Preserve the structure as much as possible."
                
                text = generate_with_fallback([prompt, pdf_part])
                if text:
                     logger.info(f"Gemini extracted {len(text)} characters.")
                else:
                    logger.warning("Gemini extraction returned no text.")
            except Exception as e:
                logger.error(f"Gemini PDF extraction failed: {e}")

        if not text:
            logger.warning("No text extracted from PDF.")
            return {"success": False, "error": "No text extracted from PDF. Please ensure the file is not corrupted or password protected."}

        logger.info(f"Extracted {len(text)} characters from PDF.")

        # Chunk text
        chunks = chunk_text(text)
        logger.info(f"Generated {len(chunks)} text chunks.")

        # Generate embeddings using Google embeddings
        embeddings = embed_chunks(chunks)
        logger.info(f"Generated {len(embeddings)} embeddings.")

        if len(chunks) != len(embeddings):
            logger.error(f"Mismatch between chunks ({len(chunks)}) and embeddings ({len(embeddings)}) count")
            # Use only chunks that have embeddings
            chunks = chunks[:len(embeddings)]
            if not chunks:
                return {"success": False, "error": "Embedding failed."}

        # Get vector store instance
        store = get_vector_store()
        if not store:
            return {"success": False, "error": "Vector store not available."}
        
        # Store in vector database
        metadata = {
            "filename": source_file,
            "original_filename": pdf_file.filename if hasattr(pdf_file, 'filename') else source_file,
            "chunk_count": len(chunks),
            "user_id": user_id
        }
        
        document_id = store.store_vectors(chunks, embeddings, metadata, user_id)
        
        if document_id:
            logger.info(f"Document '{source_file}' processed and stored successfully. Document ID: {document_id}")

            return {
                "success": True,
                "document_id": document_id,
                "filename": source_file,
                "chunks": len(chunks),
                "message": f"Document '{source_file}' processed successfully."
            }
        else:
            return {"success": False, "error": "Failed to store document in vector database."}

    except Exception as e:
        logger.exception(f"[ERROR] Failed in process_document: {e}")
        return {"success": False, "error": str(e)}

def process_query(query: str, user_id: str, language: str = "English") -> dict:
    """Process a query and return context and response using Google models"""
    try:
        logger.info(f"Processing query for user {user_id}: {query[:50]}...")

        # Get vector store instance
        store = get_vector_store()
        if not store:
            return {"context": "", "response": "Vector store not available."}

        # Generate query embedding using Google HYDE
        query_embedding = embed_query_hyde(query)
        if not query_embedding:
            logger.warning("No embedding generated for query.")
            return {"context": "", "response": "Error generating query embedding."}

        # Get user's documents
        user_docs = store.get_user_documents(user_id)
        document_ids = [doc["document_id"] for doc in user_docs] if user_docs else None

        # Retrieve relevant chunks
        top_chunks = search(query, query_embedding, user_id, document_ids)
        if not top_chunks:
            logger.warning("No results retrieved for query.")
            return {"context": "", "response": "No relevant information found in documents."}

        # Combine context
        context = "\n\n---\n\n".join(chunk["text"] for chunk in top_chunks)
        
        # Generate response using Google Gemini
        response = generate_response(query, context, language)
        
        # Add indicator that documents were analyzed
        if top_chunks:
            response = "🔍 *Analyzed your uploaded documents:*\n\n" + response
        
        # Save to chat history
        chat_data = {
            "user_id": user_id,
            "session_id": str(uuid.uuid4()),  # Generate or get from session
            "role": "user",
            "content": query,
            "response": response,
            "language": language,
            "tool_used": "rag",
            "sources": [{"file": chunk["source_file"], "score": chunk["score"]} for chunk in top_chunks]
        }
        
        mongodb.save_chat_message(chat_data)
        
        return {
            "context": context,
            "response": response,
            "sources": [{"file": chunk["source_file"], "score": chunk["score"]} for chunk in top_chunks]
        }

    except Exception as e:
        logger.exception(f"[ERROR] Failed in process_query: {e}")
        return {"context": "", "response": f"Error processing query: {str(e)}"}

def list_user_documents(user_id: str) -> list:
    """List all documents for a user"""
    store = get_vector_store()
    if store:
        return store.get_user_documents(user_id)
    return []