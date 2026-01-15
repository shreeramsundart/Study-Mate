from rag.logging_config import setup_logger
from typing import Union, List, Any
import os
from dotenv import load_dotenv

load_dotenv()
logger = setup_logger(__name__)

try:
    from google import genai
except ImportError:
    genai = None

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

client = None

if not GOOGLE_API_KEY or genai is None:
    if genai is None:
        logger.warning("google-genai package not available. LLM features will be disabled.")
    else:
        logger.warning("GOOGLE_API_KEY not found. LLM features will be disabled.")
else:
    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        logger.info("Google GenAI LLM client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Google GenAI client: {e}")
        client = None


# Simple in-memory cache for responses (Answer Caching)
# key: (user_id, session_id, message_hash)
response_cache = {}

def generate_with_fallback(contents: Union[str, List[Any]], model_name: str = None) -> str:
    """Strictly use gemini-2.5-flash without fallback as per new rules"""
    if not client:
        raise ValueError("Google GenAI client not initialized")

    # STRICT: Only use gemini-2.5-flash
    target_model = "gemini-2.5-flash"
    
    try:
        logger.info(f"Generating content with strict model: {target_model}")
        response = client.models.generate_content(
            model=target_model,
            contents=contents
        )
        
        # Extract text from response
        if hasattr(response, 'text'):
            return response.text
        elif hasattr(response, 'candidates') and response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, 'content'):
                    if hasattr(candidate.content, 'parts'):
                        text_parts = []
                        for part in candidate.content.parts:
                            if hasattr(part, 'text'):
                                text_parts.append(part.text)
                        if text_parts:
                            return "".join(text_parts)
                    elif hasattr(candidate.content, 'text'):
                        return candidate.content.text
        return str(response)
        
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            return "I have reached my daily limit for AI requests. Please try again later."
        logger.error(f"Model {target_model} failed: {e}")
        return "I encountered an error while processing your request. Please try again."


def generate_answer_hyde(question: str) -> str:
    """Strictly disabled per rules"""
    return ""


def generate_response(question: str, context: str, language: str = "English") -> str:
    """Generate a response using the retrieved context - Strict RAG only"""
    if not client:
        return "AI model not available."

    if not context:
        return "I couldn’t find this information in your uploaded documents."

    try:
        lang_instruction = ""
        if language == "Tamil":
            lang_instruction = "IMPORTANT: Respond in Tamil (தமிழ்) only."
        elif language == "Hindi":
            lang_instruction = "IMPORTANT: Respond in Hindi (हिंदी) only."

        prompt = f"""
Based on the following context, answer the question. 
{lang_instruction}
Answer using ONLY the retrieved context. If the answer is not in the context, say exactly: "I couldn’t find this information in your uploaded documents."

Context:
{context}

Question:
{question}

Answer:
"""
        return generate_with_fallback(prompt)

    except Exception as e:
        logger.exception(f"Error generating response: {e}")
        return f"I encountered an error: {str(e)}"

# Title generation strictly disabled per rule 2 & 8
def generate_chat_title(message: str, response: str) -> str:
    return None
