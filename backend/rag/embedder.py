from rag.logging_config import setup_logger
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()
logger = setup_logger(__name__)

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

client = None

if not GOOGLE_API_KEY or genai is None:
    if genai is None:
        logger.warning("google-genai package not available. Embedding functions will be disabled.")
    else:
        logger.warning("GOOGLE_API_KEY not found. Embedding functions will be disabled.")
else:
    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        logger.info("Google GenAI client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Google GenAI client: {e}")
        client = None

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")


def embed_chunks(texts: List[str]) -> List[List[float]]:
    if not texts:
        logger.warning("No texts provided for embedding.")
        return []

    if client is None:
        logger.error("Embedding client not configured (missing GOOGLE_API_KEY).")
        return []

    try:
        embeddings = []
        
        # Try different API methods based on SDK version
        for text in texts:
            try:
                # Method 1: New SDK format (google-genai >= 0.3.0)
                response = client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=[text]  # Note: contents should be a list
                )
                
                # Extract embedding from response
                if hasattr(response, 'embeddings') and response.embeddings:
                    embedding = response.embeddings[0].values
                elif hasattr(response, 'embedding') and response.embedding:
                    embedding = response.embedding
                else:
                    # Try to access directly
                    try:
                        embedding = response.embedding.values
                    except:
                        try:
                            embedding = response.embeddings[0].values
                        except:
                            embedding = None
                
                if embedding is None:
                    # Last resort: try to parse as dict
                    try:
                        if isinstance(response, dict):
                            embedding = response.get('embedding') or response.get('embeddings', [{}])[0].get('values', [])
                    except:
                        embedding = None
                        
            except TypeError as e:
                # Method 2: Try without task_type parameter
                if "task_type" in str(e):
                    response = client.models.embed_content(
                        model=EMBEDDING_MODEL,
                        contents=[text]
                    )
                    
                    if hasattr(response, 'embeddings') and response.embeddings:
                        embedding = response.embeddings[0].values
                    elif hasattr(response, 'embedding') and response.embedding:
                        embedding = response.embedding.values if hasattr(response.embedding, 'values') else response.embedding
                    else:
                        embedding = None
                else:
                    raise e
            
            if embedding:
                # Convert to list if needed
                if hasattr(embedding, 'tolist'):
                    embeddings.append(embedding.tolist())
                elif isinstance(embedding, list):
                    embeddings.append(embedding)
                else:
                    # Try to convert from other formats
                    try:
                        embeddings.append(list(embedding))
                    except:
                        logger.warning(f"Could not convert embedding to list: {type(embedding)}")
                        embeddings.append([0.0] * 768)  # Default dimension
            else:
                logger.warning(f"Could not extract embedding for text: {text[:50]}...")
                # Create empty embedding as fallback
                embeddings.append([0.0] * 768)
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings

    except Exception as e:
        logger.exception(f"Error generating embeddings: {e}")
        # Try batch embedding as fallback
        return embed_chunks_batch(texts)


def embed_chunks_batch(texts: List[str]) -> List[List[float]]:
    """Try batch embedding"""
    try:
        # Try batch embedding
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts
        )
        
        embeddings = []
        if hasattr(response, 'embeddings') and response.embeddings:
            for emb in response.embeddings:
                if hasattr(emb, 'values'):
                    embeddings.append(emb.values.tolist() if hasattr(emb.values, 'tolist') else list(emb.values))
                else:
                    embeddings.append([0.0] * 768)
        else:
            # Create empty embeddings
            embeddings = [[0.0] * 768 for _ in range(len(texts))]
        
        logger.info(f"Generated {len(embeddings)} embeddings using batch method")
        return embeddings
        
    except Exception as e:
        logger.error(f"Batch embedding failed: {e}")
        # Return dummy embeddings to allow processing to continue
        return [[0.0] * 768 for _ in range(len(texts))]


# Simple in-memory cache for query embeddings (Rule 5 & 6)
query_cache = {}

def embed_query(query: str) -> List[float]:
    if not query or not query.strip():
        logger.warning("Empty query provided for embedding.")
        return []

    # Check cache (Rule 6)
    if query in query_cache:
        logger.info(f"Using cached embedding for query: {query[:50]}...")
        return query_cache[query]

    if client is None:
        logger.error("Embedding client not configured (missing GOOGLE_API_KEY).")
        return []

    try:
        # Try different API methods
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[query]
        )
        
        # Extract embedding from response
        embedding = None
        if hasattr(response, 'embeddings') and response.embeddings:
            embedding = response.embeddings[0].values
        elif hasattr(response, 'embedding') and response.embedding:
            embedding = response.embedding
        else:
            try:
                embedding = response.embedding.values
            except:
                try:
                    embedding = response.embeddings[0].values
                except:
                    embedding = None
        
        if embedding is not None:
            # Convert to list if needed
            res = []
            if hasattr(embedding, 'tolist'):
                res = embedding.tolist()
            elif isinstance(embedding, list):
                res = embedding
            else:
                try:
                    res = list(embedding)
                except:
                    logger.warning(f"Could not convert query embedding to list: {type(embedding)}")
                    res = []
            
            # Cache result (Rule 6)
            if res:
                query_cache[query] = res
            return res
        else:
            logger.warning("Could not extract query embedding")
            return []
            
    except Exception as e:
        logger.exception(f"Error generating query embedding: {e}")
        return []


def embed_query_hyde(query: str) -> List[float]:
    """
    STRICT: HyDE disabled per Rule 8. Using regular cached embedding.
    """
    return embed_query(query)
