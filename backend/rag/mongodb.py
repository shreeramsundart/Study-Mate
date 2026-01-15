from pymongo import MongoClient, ASCENDING, DESCENDING
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
import uuid
import logging
from dotenv import load_dotenv
import os

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MongoDBManager:
    def __init__(self):
        self.mongo_uri = os.getenv("MONGODB_URI")
        if not self.mongo_uri:
            raise ValueError("MONGODB_URI not found in environment variables")

        self.client = MongoClient(self.mongo_uri)
        self.db = self.client["Study-Mate"]

        # Collections
        self.users = self.db["users"]
        self.chats = self.db["chats"]
        self.documents = self.db["documents"]
        self.chunks = self.db["chunks"]
        self.sessions = self.db["sessions"]

        self._create_indexes()

    # -------------------- INDEXES --------------------
    def _create_indexes(self):
        """Create necessary indexes (Atlas Free Tier safe)"""

        # Users
        self.users.create_index([("email", ASCENDING)], unique=True)
        self.users.create_index([("username", ASCENDING)], unique=True)

        # Chats
        self.chats.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
        self.chats.create_index([("session_id", ASCENDING)])

        # Documents
        self.documents.create_index([("user_id", ASCENDING), ("uploaded_at", DESCENDING)])

        # Chunks
        self.chunks.create_index([("document_id", ASCENDING)])

        # Sessions
        self.sessions.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
        self.sessions.create_index([("session_id", ASCENDING)], unique=True)

        logger.info("MongoDB indexes created successfully")

    # -------------------- USER MANAGEMENT --------------------
    def create_user(self, user_data: Dict) -> Optional[str]:
        try:
            user_data.update({
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            })
            result = self.users.insert_one(user_data)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Create user error: {e}")
            return None

    def get_user_by_email(self, email: str) -> Optional[Dict]:
        return self.users.find_one({"email": email})

    def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        from bson import ObjectId
        try:
            return self.users.find_one({"_id": ObjectId(user_id)})
        except:
            return None

    def update_user(self, user_id: str, update_data: Dict) -> bool:
        from bson import ObjectId
        try:
            update_data["updated_at"] = datetime.utcnow()
            result = self.users.update_one(
                {"_id": ObjectId(user_id)},
                {"$set": update_data}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Update user error: {e}")
            return False

    # -------------------- CHAT MANAGEMENT --------------------
    def save_chat_message(self, message_data: Dict) -> str:
        message_data.update({
            "created_at": datetime.utcnow(),
            "message_id": str(uuid.uuid4())
        })
        return str(self.chats.insert_one(message_data).inserted_id)

    def get_chat_history(self, user_id: str, session_id: str = None, limit: int = 50) -> List[Dict]:
        query = {"user_id": user_id}
        if session_id:
            query["session_id"] = session_id

        cursor = self.chats.find(query).sort("created_at", ASCENDING).limit(limit)
        return [{**doc, "_id": str(doc["_id"])} for doc in cursor]

    def clear_chat_history(self, user_id: str, session_id: str = None) -> bool:
        query = {"user_id": user_id}
        if session_id:
            query["session_id"] = session_id
        return self.chats.delete_many(query).deleted_count > 0

    # -------------------- DOCUMENT MANAGEMENT --------------------
    def save_document(self, document_data: Dict) -> str:
        document_data.update({
            "uploaded_at": datetime.utcnow(),
            "document_id": str(uuid.uuid4())
        })
        return str(self.documents.insert_one(document_data).inserted_id)

    def get_user_documents(self, user_id: str) -> List[Dict]:
        cursor = self.documents.find({"user_id": user_id}).sort("uploaded_at", DESCENDING)
        return [{**doc, "_id": str(doc["_id"])} for doc in cursor]

    def delete_document(self, document_id: str, user_id: str) -> bool:
        result = self.documents.delete_one({
            "document_id": document_id,
            "user_id": user_id
        })
        if result.deleted_count:
            self.chunks.delete_many({"document_id": document_id})
            return True
        return False

    # -------------------- VECTOR STORAGE --------------------
    def save_chunk(self, chunk_data: Dict) -> str:
        chunk_data["created_at"] = datetime.utcnow()
        if isinstance(chunk_data.get("embedding"), np.ndarray):
            chunk_data["embedding"] = chunk_data["embedding"].tolist()
        return str(self.chunks.insert_one(chunk_data).inserted_id)

    def search_similar_chunks(
        self,
        embedding: List[float],
        limit: int = 10,
        user_id: str = None,
        document_ids: List[str] = None
    ) -> List[Dict]:

        pipeline = []

        match = {}
        if user_id:
            match["user_id"] = user_id
        if document_ids:
            match["document_id"] = {"$in": document_ids}
        if match:
            pipeline.append({"$match": match})

        pipeline.append({
            "$vectorSearch": {
                "index": "vector_index",   # Atlas Search Index Name
                "path": "embedding",
                "queryVector": embedding,
                "numCandidates": limit * 10,
                "limit": limit
            }
        })

        pipeline.append({
            "$project": {
                "text": 1,
                "document_id": 1,
                "metadata": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        })

        try:
            results = list(self.chunks.aggregate(pipeline))
            for r in results:
                r["_id"] = str(r["_id"])
            return results
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
                 logger.error(f"Vector search failed: {e}")
            
            return self._fallback_similarity_search(embedding, limit, user_id, document_ids)

    def _fallback_similarity_search(
        self,
        embedding: List[float],
        limit: int,
        user_id: str = None,
        document_ids: List[str] = None
    ) -> List[Dict]:

        query = {}
        if user_id:
            query["user_id"] = user_id
        if document_ids:
            query["document_id"] = {"$in": document_ids}

        chunks = list(self.chunks.find(query))
        similarities = []

        for chunk in chunks:
            vec = chunk.get("embedding")
            if vec and len(vec) == len(embedding):
                sim = np.dot(embedding, vec) / (
                    np.linalg.norm(embedding) * np.linalg.norm(vec)
                )
                similarities.append((sim, chunk))

        similarities.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, chunk in similarities[:limit]:
            chunk["_id"] = str(chunk["_id"])
            chunk["score"] = float(score)
            results.append(chunk)

        return results

    # -------------------- SESSION MANAGEMENT --------------------
    def create_session(self, session_data: Dict) -> str:
        session_data.update({
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        })
        return str(self.sessions.insert_one(session_data).inserted_id)

    def get_user_sessions(self, user_id: str) -> List[Dict]:
        cursor = self.sessions.find({"user_id": user_id}).sort("created_at", DESCENDING)
        return [{**doc, "_id": str(doc["_id"])} for doc in cursor]

    def update_session(self, session_id: str, update_data: Dict) -> bool:
        update_data["updated_at"] = datetime.utcnow()
        return self.sessions.update_one(
            {"session_id": session_id},
            {"$set": update_data}
        ).modified_count > 0

    def delete_session(self, session_id: str, user_id: str) -> bool:
        result = self.sessions.delete_one({
            "session_id": session_id,
            "user_id": user_id
        })
        if result.deleted_count:
            self.chats.delete_many({
                "session_id": session_id,
                "user_id": user_id
            })
            return True
        return False


# -------------------- GLOBAL INSTANCE --------------------
mongodb = MongoDBManager()
