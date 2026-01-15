from langchain_text_splitters import RecursiveCharacterTextSplitter
from .logging_config import logger
from typing import List

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks using RecursiveCharacterTextSplitter
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of text chunks
    """
    try:
        logger.info(f"Starting text chunking (text length: {len(text)})")
        
        # Initialize splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )
        
        # Split text
        chunks = splitter.split_text(text)
        
        logger.info(f"Successfully chunked text into {len(chunks)} chunks")
        
        # Log chunk statistics
        if chunks:
            avg_chunk_size = sum(len(chunk) for chunk in chunks) / len(chunks)
            logger.info(f"Average chunk size: {avg_chunk_size:.0f} characters")
            logger.info(f"Chunk sizes: {[len(chunk) for chunk in chunks[:5]]}...")
        
        return chunks
        
    except Exception as e:
        logger.exception(f"Error during text chunking: {e}")
        # Fallback: simple splitting by paragraphs
        try:
            paragraphs = text.split("\n\n")
            chunks = []
            current_chunk = ""
            
            for para in paragraphs:
                if len(current_chunk) + len(para) > chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = para
                    else:
                        chunks.append(para[:chunk_size])
                        current_chunk = para[chunk_size:]
                else:
                    current_chunk += "\n\n" + para if current_chunk else para
            
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            logger.warning(f"Used fallback chunking: created {len(chunks)} chunks")
            return chunks
        except Exception as fallback_error:
            logger.error(f"Fallback chunking also failed: {fallback_error}")
            return [text] if text.strip() else []