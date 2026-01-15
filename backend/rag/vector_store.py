from typing import List, Dict, Optional, Any, Union
import numpy as np
from datetime import datetime, timedelta
import uuid
import logging
import os
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.operations import SearchIndexModel
from bson import ObjectId
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class MongoDBVectorStore:
    """
    Vector store implementation using MongoDB's vector search
    """
    
    def __init__(self, connection_string: str = None, database_name: str = None):
        """
        Initialize MongoDB vector store
        
        Args:
            connection_string: MongoDB connection string
            database_name: Database name
        """
        self.connection_string = connection_string or os.getenv(
            "MONGODB_URI", 
            "mongodb://localhost:27017/Study-Mate"
        )
        self.database_name = database_name or "Study-Mate"
        self.vector_dimension = int(os.getenv("VECTOR_DIMENSION", 768))
        
        # MongoDB collections
        self.client = None
        self.db = None
        self.documents_collection = None
        self.chunks_collection = None
        self.users_collection = None
        self.use_atlas_search = True
        
        self._initialize_connection()
        self._ensure_indexes()
        self._ensure_vector_search_index()
    
    def _initialize_connection(self):
        """Initialize MongoDB connection"""
        try:
            logger.info(f"Connecting to MongoDB: {self.connection_string.split('@')[-1] if '@' in self.connection_string else self.connection_string}")
            
            self.client = MongoClient(
                self.connection_string,
                maxPoolSize=50,
                minPoolSize=10,
                connectTimeoutMS=30000,
                socketTimeoutMS=45000,
                serverSelectionTimeoutMS=30000
            )
            
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[self.database_name]
            
            # Initialize collections
            self.documents_collection = self.db.documents
            self.chunks_collection = self.db.chunks
            self.users_collection = self.db.users
            
            logger.info(f"Connected to MongoDB database: {self.database_name}")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def _ensure_indexes(self):
        """Ensure necessary indexes exist"""
        try:
            # Index for documents collection
            self.documents_collection.create_index([("user_id", ASCENDING), ("uploaded_at", DESCENDING)])
            self.documents_collection.create_index([("document_id", ASCENDING)], unique=True)
            self.documents_collection.create_index([("filename", ASCENDING)])
            
            # Index for chunks collection
            self.chunks_collection.create_index([("document_id", ASCENDING)])
            self.chunks_collection.create_index([("user_id", ASCENDING)])
            self.chunks_collection.create_index([("chunk_index", ASCENDING)])
            self.chunks_collection.create_index([("created_at", DESCENDING)])
            
            # Index for users collection
            self.users_collection.create_index([("email", ASCENDING)], unique=True)
            self.users_collection.create_index([("username", ASCENDING)], unique=True)
            
            logger.info("Database indexes ensured")
            
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
            raise
    
    def _ensure_vector_search_index(self):
        """Create vector search index if it doesn't exist"""
        try:
            # Check if vector search index already exists
            existing_indexes = list(self.db.command({"listSearchIndexes": "chunks"}))
            
            vector_index_exists = any(
                idx.get("name") == "vector_index" 
                for idx in existing_indexes
            )
            
            if not vector_index_exists:
                logger.info("Creating vector search index...")
                
                # Create vector search index
                index_definition = {
                    "name": "vector_index",
                    "definition": {
                        "fields": [
                            {
                                "type": "vector",
                                "path": "embedding",
                                "numDimensions": self.vector_dimension,
                                "similarity": "cosine"
                            },
                            {
                                "type": "filter",
                                "path": "user_id"
                            },
                            {
                                "type": "filter",
                                "path": "document_id"
                            }
                        ]
                    }
                }
                
                # Create the search index
                self.db.command({
                    "createSearchIndexes": "chunks",
                    "indexes": [index_definition]
                })
                
                logger.info("Vector search index created successfully")
            else:
                logger.info("Vector search index already exists")
                
        except Exception as e:
            # Check for 'CommandNotFound' (code 59) which means we are likely on local MongoDB
            # or a version that doesn't support Atlas Search
            is_command_not_found = (
                isinstance(e, Exception) and 
                (getattr(e, 'code', None) == 59 or 'command not found' in str(e).lower())
            )
            
            if is_command_not_found:
                logger.info("MongoDB Atlas Vector Search not available (running locally?). Using manual similarity calculation.")
                self.use_atlas_search = False
            else:
                logger.error(f"Failed to create vector search index: {e}")
                logger.warning("Falling back to manual similarity calculation")
                self.use_atlas_search = False
    
    def store_vectors(
        self, 
        chunks: List[str], 
        embeddings: List[List[float]], 
        metadata: Dict, 
        user_id: str
    ) -> str:
        """
        Store document chunks with embeddings in MongoDB
        
        Args:
            chunks: List of text chunks
            embeddings: List of embedding vectors
            metadata: Document metadata
            user_id: User ID who owns the document
        
        Returns:
            Document ID
        """
        if not chunks or not embeddings:
            logger.warning("No chunks or embeddings provided")
            return None
        
        if len(chunks) != len(embeddings):
            logger.error(f"Chunks and embeddings count mismatch: {len(chunks)} vs {len(embeddings)}")
            return None
        
        try:
            # Generate document ID
            document_id = metadata.get("document_id", str(uuid.uuid4()))
            filename = metadata.get("filename", "unknown")
            
            # Store document metadata
            document_data = {
                "document_id": document_id,
                "user_id": user_id,
                "filename": filename,
                "original_filename": metadata.get("original_filename", filename),
                "file_size": metadata.get("file_size", 0),
                "mime_type": metadata.get("mime_type", "application/pdf"),
                "chunk_count": len(chunks),
                "uploaded_at": datetime.utcnow(),
                "metadata": metadata,
                "status": "processed"
            }
            
            # Insert or update document
            self.documents_collection.update_one(
                {"document_id": document_id, "user_id": user_id},
                {"$set": document_data},
                upsert=True
            )
            
            logger.info(f"Storing {len(chunks)} chunks for document: {filename}")
            
            # Prepare chunks for bulk insertion
            chunk_documents = []
            for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = str(uuid.uuid4())
                
                # Validate embedding dimension
                if len(embedding) != self.vector_dimension:
                    logger.warning(f"Chunk {i} has wrong dimension: {len(embedding)}. Expected: {self.vector_dimension}")
                    # Truncate or pad embedding to correct dimension
                    if len(embedding) > self.vector_dimension:
                        embedding = embedding[:self.vector_dimension]
                    else:
                        embedding = embedding + [0.0] * (self.vector_dimension - len(embedding))
                
                chunk_doc = {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "user_id": user_id,
                    "chunk_index": i,
                    "text": chunk_text[:5000],  # Limit text length
                    "embedding": embedding,
                    "metadata": {
                        "source_file": filename,
                        "chunk_length": len(chunk_text),
                        **metadata
                    },
                    "created_at": datetime.utcnow()
                }
                chunk_documents.append(chunk_doc)
            
            # Insert chunks in batches
            batch_size = 100
            success_count = 0
            
            for i in range(0, len(chunk_documents), batch_size):
                batch = chunk_documents[i:i + batch_size]
                try:
                    result = self.chunks_collection.insert_many(batch, ordered=False)
                    success_count += len(result.inserted_ids)
                except Exception as batch_error:
                    logger.error(f"Error inserting batch {i//batch_size}: {batch_error}")
                    # Try inserting individually
                    for chunk in batch:
                        try:
                            self.chunks_collection.insert_one(chunk)
                            success_count += 1
                        except Exception as single_error:
                            logger.error(f"Error inserting single chunk: {single_error}")
            
            logger.info(f"Successfully stored {success_count}/{len(chunks)} chunks for document {document_id}")
            
            # Update document with actual stored chunk count
            self.documents_collection.update_one(
                {"document_id": document_id, "user_id": user_id},
                {"$set": {"stored_chunk_count": success_count}}
            )
            
            return document_id
            
        except Exception as e:
            logger.exception(f"Failed to store vectors: {e}")
            return None
    
    def get_user_documents(self, user_id: str, limit: int = 100) -> List[Dict]:
        """
        Get all documents for a user
        
        Args:
            user_id: User ID
            limit: Maximum number of documents to return
        
        Returns:
            List of document metadata
        """
        try:
            documents = list(self.documents_collection.find(
                {"user_id": user_id},
                {
                    "_id": 0,
                    "document_id": 1,
                    "filename": 1,
                    "original_filename": 1,
                    "file_size": 1,
                    "chunk_count": 1,
                    "uploaded_at": 1,
                    "status": 1,
                    "metadata": 1
                }
            ).sort("uploaded_at", DESCENDING).limit(limit))
            
            # Convert ObjectId and datetime to strings
            for doc in documents:
                if "uploaded_at" in doc and isinstance(doc["uploaded_at"], datetime):
                    doc["uploaded_at"] = doc["uploaded_at"].isoformat()
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to get user documents: {e}")
            return []
    
    def search_vectors(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        user_id: str = None,
        document_ids: List[str] = None,
        min_score: float = 0.0
    ) -> List[Dict]:
        """
        Search for similar vectors using MongoDB vector search
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            user_id: Filter by user ID
            document_ids: Filter by document IDs
            min_score: Minimum similarity score
        
        Returns:
            List of search results with scores
        """
        if not query_embedding:
            logger.warning("No query embedding provided")
            return []
        
        if len(query_embedding) != self.vector_dimension:
            logger.error(f"Query embedding dimension mismatch: {len(query_embedding)} != {self.vector_dimension}")
            return []
        
        if not self.use_atlas_search:
            return self._fallback_search(
                query_embedding, top_k, user_id, document_ids, min_score
            )

        try:
            # Build aggregation pipeline for vector search
            pipeline = []
            
            # Vector search stage (requires MongoDB 6.0.11+ with Atlas or Enterprise)
            vector_search_stage = {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": top_k * 10,
                    "limit": top_k
                }
            }
            
            # Add filters if provided
            if user_id or document_ids:
                filter_conditions = {}
                if user_id:
                    filter_conditions["user_id"] = user_id
                if document_ids:
                    filter_conditions["document_id"] = {"$in": document_ids}
                
                vector_search_stage["$vectorSearch"]["filter"] = filter_conditions
            
            pipeline.append(vector_search_stage)
            
            # Project stage to include metadata and score
            pipeline.append({
                "$project": {
                    "_id": 1,
                    "chunk_id": 1,
                    "document_id": 1,
                    "user_id": 1,
                    "chunk_index": 1,
                    "text": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            })
            
            # Execute search
            results = list(self.chunks_collection.aggregate(pipeline))
            
            # Filter by minimum score
            filtered_results = [
                result for result in results 
                if result.get("score", 0) >= min_score
            ]
            
            # Format results
            formatted_results = []
            for result in filtered_results[:top_k]:
                formatted_results.append({
                    "id": str(result.get("_id")),
                    "chunk_id": result.get("chunk_id"),
                    "document_id": result.get("document_id"),
                    "user_id": result.get("user_id"),
                    "chunk_index": result.get("chunk_index", 0),
                    "text": result.get("text", ""),
                    "score": result.get("score", 0.0),
                    "source_file": result.get("metadata", {}).get("source_file", "unknown"),
                    "metadata": result.get("metadata", {})
                })
            
            logger.info(f"Vector search found {len(formatted_results)} results")
            
            return formatted_results
            
        except Exception as e:
            # Check for 'CommandNotFound' (code 59) -> Local MongoDB
            is_command_not_found = (
                isinstance(e, Exception) and 
                (getattr(e, 'code', None) == 59 or 
                 (hasattr(e, 'details') and isinstance(e.details, dict) and e.details.get('code') == 59) or
                 'unrecognized pipeline stage name: $vectorSearch' in str(e) or
                 'command not found' in str(e).lower())
            )
            
            if is_command_not_found:
                 logger.info("Using manual fallback search (Atlas Vector Search not available)")
            else:
                 logger.warning(f"Vector search failed: {e}. Using fallback method.")
            
            return self._fallback_search(
                query_embedding, top_k, user_id, document_ids, min_score
            )
    
    def _fallback_search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        user_id: str = None,
        document_ids: List[str] = None,
        min_score: float = 0.0
    ) -> List[Dict]:
        """
        Fallback search using manual cosine similarity calculation
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            user_id: Filter by user ID
            document_ids: Filter by document IDs
            min_score: Minimum similarity score
        
        Returns:
            List of search results with scores
        """
        try:
            # Build query
            query = {}
            if user_id:
                query["user_id"] = user_id
            if document_ids:
                query["document_id"] = {"$in": document_ids}
            
            # Get all relevant chunks (warning: inefficient for large collections)
            chunks = list(self.chunks_collection.find(
                query,
                {
                    "_id": 1,
                    "chunk_id": 1,
                    "document_id": 1,
                    "user_id": 1,
                    "chunk_index": 1,
                    "text": 1,
                    "embedding": 1,
                    "metadata": 1
                }
            ).limit(1000))  # Limit to prevent memory issues
            
            if not chunks:
                return []
            
            # Convert query embedding to numpy array
            query_vec = np.array(query_embedding)
            query_norm = np.linalg.norm(query_vec)
            
            # Calculate similarities
            similarities = []
            for chunk in chunks:
                chunk_embedding = chunk.get("embedding")
                if chunk_embedding and len(chunk_embedding) == len(query_embedding):
                    chunk_vec = np.array(chunk_embedding)
                    chunk_norm = np.linalg.norm(chunk_vec)
                    
                    if query_norm > 0 and chunk_norm > 0:
                        similarity = np.dot(query_vec, chunk_vec) / (query_norm * chunk_norm)
                        
                        if similarity >= min_score:
                            similarities.append((similarity, chunk))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[0], reverse=True)
            
            # Format results
            formatted_results = []
            for similarity, chunk in similarities[:top_k]:
                formatted_results.append({
                    "id": str(chunk.get("_id")),
                    "chunk_id": chunk.get("chunk_id"),
                    "document_id": chunk.get("document_id"),
                    "user_id": chunk.get("user_id"),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "text": chunk.get("text", ""),
                    "score": float(similarity),
                    "source_file": chunk.get("metadata", {}).get("source_file", "unknown"),
                    "metadata": chunk.get("metadata", {})
                })
            
            logger.info(f"Fallback search found {len(formatted_results)} results")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
            return []
    
    def delete_document(self, document_id: str, user_id: str) -> bool:
        """
        Delete a document and all its chunks
        
        Args:
            document_id: Document ID
            user_id: User ID (for verification)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Start a session for transaction
            with self.client.start_session() as session:
                with session.start_transaction():
                    # Delete document metadata
                    doc_result = self.documents_collection.delete_one(
                        {"document_id": document_id, "user_id": user_id},
                        session=session
                    )
                    
                    if doc_result.deleted_count == 0:
                        logger.warning(f"Document not found or user mismatch: {document_id}")
                        session.abort_transaction()
                        return False
                    
                    # Delete associated chunks
                    chunk_result = self.chunks_collection.delete_many(
                        {"document_id": document_id, "user_id": user_id},
                        session=session
                    )
                    
                    logger.info(f"Deleted document {document_id} with {chunk_result.deleted_count} chunks")
                    
                    session.commit_transaction()
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return False
    
    def delete_user_data(self, user_id: str) -> bool:
        """
        Delete all data for a user
        
        Args:
            user_id: User ID
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Start a session for transaction
            with self.client.start_session() as session:
                with session.start_transaction():
                    # Delete user's documents
                    doc_result = self.documents_collection.delete_many(
                        {"user_id": user_id},
                        session=session
                    )
                    
                    # Delete user's chunks
                    chunk_result = self.chunks_collection.delete_many(
                        {"user_id": user_id},
                        session=session
                    )
                    
                    logger.info(f"Deleted {doc_result.deleted_count} documents and {chunk_result.deleted_count} chunks for user {user_id}")
                    
                    session.commit_transaction()
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to delete user data: {e}")
            return False
    
    def get_statistics(self, user_id: str = None) -> Dict:
        """
        Get vector store statistics
        
        Args:
            user_id: Optional user ID to filter statistics
        
        Returns:
            Dictionary of statistics
        """
        try:
            stats = {}
            
            # Build match stage for aggregation
            match_stage = {}
            if user_id:
                match_stage["user_id"] = user_id
            
            # Document statistics
            doc_pipeline = []
            if match_stage:
                doc_pipeline.append({"$match": match_stage})
            
            doc_pipeline.extend([
                {"$group": {
                    "_id": None,
                    "total_documents": {"$sum": 1},
                    "total_chunks": {"$sum": "$chunk_count"},
                    "avg_chunks_per_doc": {"$avg": "$chunk_count"}
                }}
            ])
            
            doc_stats = list(self.documents_collection.aggregate(doc_pipeline))
            if doc_stats:
                stats.update(doc_stats[0])
                del stats["_id"]
            
            # Chunk statistics
            chunk_pipeline = []
            if match_stage:
                chunk_pipeline.append({"$match": match_stage})
            
            chunk_pipeline.extend([
                {"$group": {
                    "_id": None,
                    "total_stored_chunks": {"$sum": 1},
                    "avg_text_length": {"$avg": {"$strLenCP": "$text"}}
                }}
            ])
            
            chunk_stats = list(self.chunks_collection.aggregate(chunk_pipeline))
            if chunk_stats:
                stats.update(chunk_stats[0])
                del stats["_id"]
            
            # Storage size (approximate)
            if user_id:
                user_docs = list(self.documents_collection.find(
                    {"user_id": user_id},
                    {"file_size": 1}
                ))
                stats["total_storage_bytes"] = sum(doc.get("file_size", 0) for doc in user_docs)
            else:
                all_docs = list(self.documents_collection.find({}, {"file_size": 1}))
                stats["total_storage_bytes"] = sum(doc.get("file_size", 0) for doc in all_docs)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def update_chunk_metadata(
        self, 
        chunk_id: str, 
        metadata: Dict, 
        user_id: str = None
    ) -> bool:
        """
        Update metadata for a specific chunk
        
        Args:
            chunk_id: Chunk ID
            metadata: New metadata
            user_id: Optional user ID for verification
        
        Returns:
            True if successful, False otherwise
        """
        try:
            query = {"chunk_id": chunk_id}
            if user_id:
                query["user_id"] = user_id
            
            result = self.chunks_collection.update_one(
                query,
                {"$set": {"metadata": metadata}}
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Failed to update chunk metadata: {e}")
            return False
    
    def cleanup_old_chunks(self, days_old: int = 30) -> int:
        """
        Clean up chunks older than specified days
        
        Args:
            days_old: Delete chunks older than this many days
        
        Returns:
            Number of chunks deleted
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            # Find documents with only old chunks
            pipeline = [
                {
                    "$match": {
                        "created_at": {"$lt": cutoff_date}
                    }
                },
                {
                    "$group": {
                        "_id": "$document_id",
                        "total_chunks": {"$sum": 1},
                        "old_chunks": {"$sum": 1}
                    }
                },
                {
                    "$match": {
                        "$expr": {"$eq": ["$total_chunks", "$old_chunks"]}
                    }
                }
            ]
            
            old_documents = list(self.chunks_collection.aggregate(pipeline))
            doc_ids = [doc["_id"] for doc in old_documents]
            
            # Delete old chunks
            result = self.chunks_collection.delete_many({
                "created_at": {"$lt": cutoff_date},
                "document_id": {"$in": doc_ids}
            })
            
            # Delete documents that have no chunks left
            self.documents_collection.delete_many({
                "document_id": {"$in": doc_ids}
            })
            
            logger.info(f"Cleaned up {result.deleted_count} chunks from {len(doc_ids)} documents")
            
            return result.deleted_count
            
        except Exception as e:
            logger.error(f"Failed to clean up old chunks: {e}")
            return 0
    
    def batch_embed_and_store(
        self,
        texts: List[str],
        metadata_list: List[Dict],
        user_id: str,
        batch_size: int = 50
    ) -> List[str]:
        """
        Batch embed and store texts
        
        Args:
            texts: List of texts to embed
            metadata_list: List of metadata for each text
            user_id: User ID
            batch_size: Batch size for embedding
        
        Returns:
            List of document IDs
        """
        from rag.embedder import embed_chunks
        
        document_ids = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_metadata = metadata_list[i:i + batch_size]
            
            # Generate embeddings
            embeddings = embed_chunks(batch_texts)
            
            if not embeddings:
                logger.warning(f"Failed to generate embeddings for batch starting at index {i}")
                continue
            
            # Store each document
            for j, (text, metadata, embedding) in enumerate(zip(batch_texts, batch_metadata, embeddings)):
                doc_id = self.store_vectors(
                    chunks=[text],
                    embeddings=[embedding],
                    metadata=metadata,
                    user_id=user_id
                )
                
                if doc_id:
                    document_ids.append(doc_id)
            
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        
        return document_ids
    
    def hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        user_id: str = None,
        document_ids: List[str] = None,
        top_k: int = 10,
        keyword_weight: float = 0.3,
        vector_weight: float = 0.7
    ) -> List[Dict]:
        """
        Hybrid search combining vector similarity and keyword matching
        
        Args:
            query: Search query text
            query_embedding: Query embedding vector
            user_id: Filter by user ID
            document_ids: Filter by document IDs
            top_k: Number of results
            keyword_weight: Weight for keyword matching
            vector_weight: Weight for vector similarity
        
        Returns:
            List of hybrid search results
        """
        try:
            # Get vector search results
            vector_results = self.search_vectors(
                query_embedding=query_embedding,
                top_k=top_k * 2,  # Get more results for hybrid ranking
                user_id=user_id,
                document_ids=document_ids
            )
            
            if not vector_results:
                return []
            
            # Calculate keyword scores
            query_terms = set(query.lower().split())
            
            for result in vector_results:
                text = result.get("text", "").lower()
                
                # Calculate keyword match score
                text_terms = set(text.split())
                common_terms = query_terms.intersection(text_terms)
                
                keyword_score = len(common_terms) / max(len(query_terms), 1)
                
                # Combine scores
                vector_score = result.get("score", 0.0)
                hybrid_score = (vector_weight * vector_score) + (keyword_weight * keyword_score)
                
                result["hybrid_score"] = hybrid_score
                result["keyword_score"] = keyword_score
            
            # Sort by hybrid score
            vector_results.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)
            
            return vector_results[:top_k]
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return self.search_vectors(
                query_embedding=query_embedding,
                top_k=top_k,
                user_id=user_id,
                document_ids=document_ids
            )
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")


# -------------------- GLOBAL INSTANCE --------------------
# Initialize the global vector store instance
_vector_store_instance = None

def get_vector_store() -> MongoDBVectorStore:
    """Get or create global vector store instance"""
    global _vector_store_instance
    if _vector_store_instance is None:
        try:
            _vector_store_instance = MongoDBVectorStore()
            logger.info("Vector store instance created successfully")
        except Exception as e:
            logger.error(f"Failed to create vector store instance: {e}")
            _vector_store_instance = None
    return _vector_store_instance

# Create a global reference
vector_store = get_vector_store()

if __name__ == "__main__":
    # Test the vector store
    import logging
    logging.basicConfig(level=logging.INFO)
    
    store = get_vector_store()
    
    if store:
        # Test statistics
        stats = store.get_statistics()
        print(f"Vector Store Statistics: {stats}")
        
        # Test connection
        print(f"Connected to: {store.connection_string}")
        print(f"Database: {store.database_name}")
        print(f"Vector Dimension: {store.vector_dimension}")
        
        store.close()
    else:
        print("Failed to initialize vector store")