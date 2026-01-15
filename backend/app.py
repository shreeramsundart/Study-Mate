from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import traceback
import time
import uuid
import base64
from pathlib import Path
from werkzeug.utils import secure_filename
from functools import wraps
import re
import io

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Environment setup
from dotenv import load_dotenv
# Load .env from backend directory first, then root
load_dotenv()
load_dotenv(Path(__file__).parent.parent / '.env')

# Initialize Flask app
app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app, resources={r"/api/*": {"origins": "*"}})

# JWT Configuration
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-this')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = 86400  # 24 hours in seconds

# Import RAG modules
try:
    # First initialize auth extensions
    from rag.auth import init_auth, bcrypt, jwt
    app = init_auth(app)
    
    # Now import other modules
    from rag.main import process_document, process_query, list_user_documents
    from rag.auth import AuthManager
    from rag.logging_config import setup_logger
    from rag.mongodb import mongodb
    from rag.vector_store import vector_store
    from rag.retriever import search_with_query, get_relevant_context
    from rag.llm import generate_with_fallback, generate_chat_title
    RAG_AVAILABLE = True
    logger = setup_logger(__name__)
except Exception as e:
    print(f"⚠️ RAG system import failed: {e}")
    traceback.print_exc()
    RAG_AVAILABLE = False
    # Create a simple logger if RAG fails
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Import JWT functions after initialization
from flask_jwt_extended import jwt_required, get_jwt_identity, create_access_token

# Google AI SDK
try:
    from google import genai
    from google.genai import types
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False
    print("⚠️ google-genai package not available. Install with: pip install google-genai")

# Other imports
try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False
    print("⚠️ google-api-python-client not available.")

try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    print("⚠️ duckduckgo-search not available.")

# Initialize Google AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Configure Google AI
model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
google_client = None

if GOOGLE_AI_AVAILABLE and GOOGLE_API_KEY:
    try:
        google_client = genai.Client(api_key=GOOGLE_API_KEY)
        logger.info(f"✅ Google AI configured with model: {model_name}")
    except Exception as e:
        logger.error(f"Failed to init GenAI: {e}")
        google_client = None
        GOOGLE_AI_AVAILABLE = False
else:
    logger.warning("⚠️ Google AI not configured")

# Initialize RAG system
rag_initialized = False
try:
    if RAG_AVAILABLE:
        rag_initialized = True  # MongoDB is initialized in mongodb.py
        logger.info("✅ RAG system initialized successfully")
    else:
        rag_initialized = False
        logger.warning("⚠️ RAG system not available")
except Exception as e:
    logger.error(f"Failed to initialize RAG system: {e}")
    rag_initialized = False

# ==================== AUTH DECORATORS ====================

def auth_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        # Handle malformed "null" or "undefined" tokens from frontend
        auth_header = request.headers.get('Authorization')
        if not auth_header or 'null' in auth_header or 'undefined' in auth_header:
            return jsonify({"error": "No valid token provided"}), 401
            
        @jwt_required()
        def wrapped_f():
            current_user = AuthManager.get_current_user()
            if not current_user:
                return jsonify({"error": "Unauthorized"}), 401
            kwargs['current_user'] = current_user
            return f(*args, **kwargs)
        
        return wrapped_f()
    return decorated

# ==================== AUTH ENDPOINTS ====================

@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register a new user"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        required = ['username', 'email', 'password']
        for field in required:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        result, error = AuthManager.register_user(
            username=data['username'],
            email=data['email'],
            password=data['password'],
            full_name=data.get('full_name')
        )
        
        if error:
            return jsonify({"error": error}), 400
        
        return jsonify(result), 201
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Login user"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({"error": "Email and password required"}), 400
        
        result, error = AuthManager.login_user(email, password)
        
        if error:
            return jsonify({"error": error}), 401
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/auth/me', methods=['GET'])
@auth_required
def get_current_user(current_user):
    """Get current user information"""
    return jsonify(current_user), 200

@app.route('/api/auth/update', methods=['PUT'])
@auth_required
def update_user(current_user):
    """Update user information"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Update user preferences
        if 'preferences' in data:
            AuthManager.update_user_preferences(current_user['id'], data['preferences'])
        
        return jsonify({"message": "User updated successfully"}), 200
        
    except Exception as e:
        logger.error(f"Update user error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/auth/clear-data', methods=['POST'])
@auth_required
def clear_user_data(current_user):
    """Clear all user data (chats, documents, sessions)"""
    try:
        user_id = current_user['id']
        
        # Clear chat history
        mongodb.clear_chat_history(user_id)
        
        # Clear documents and chunks
        if vector_store:
            vector_store.delete_user_data(user_id)
        
        # Clear sessions
        mongodb.sessions.delete_many({"user_id": user_id})
        
        logger.info(f"Cleared all data for user {user_id}")
        return jsonify({"message": "All data cleared successfully"}), 200
        
    except Exception as e:
        logger.error(f"Error clearing user data: {e}")
        return jsonify({"error": str(e)}), 500

# ==================== CHAT ENDPOINTS ====================

@app.route('/api/chat/session', methods=['POST'])
@auth_required
def create_session(current_user):
    """Create a new chat session"""
    try:
        session_id = str(uuid.uuid4())
        
        session_data = {
            "user_id": current_user['id'],
            "session_id": session_id,
            "title": f"Chat {time.strftime('%Y-%m-%d %H:%M')}",
            "created_at": time.time(),
            "updated_at": time.time()
        }
        
        mongodb.create_session(session_data)
        
        return jsonify({
            "session_id": session_id,
            "message": "New session created",
            "timestamp": time.time()
        }), 201
        
    except Exception as e:
        logger.error(f"Create session error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat/history', methods=['GET'])
@auth_required
def get_chat_history(current_user):
    """Get chat history for user"""
    try:
        session_id = request.args.get('session_id')
        limit = int(request.args.get('limit', 100))
        
        history = mongodb.get_chat_history(
            user_id=current_user['id'],
            session_id=session_id,
            limit=limit
        )
        
        # Get session info
        session_data = mongodb.sessions.find_one({"session_id": session_id})
        session_title = session_data.get('title', 'Chat Session') if session_data else 'Chat Session'
        
        return jsonify({
            "history": history,
            "count": len(history),
            "session_title": session_title
        }), 200
        
    except Exception as e:
        logger.error(f"Get history error: {e}")
        return jsonify({"error": str(e)}), 500

# User cooldown storage (Rule 7)
user_last_call = {}

@app.route('/api/chat/message', methods=['POST'])
@auth_required
def chat_message(current_user):
    """Handle chat messages with strict RAG constraints"""
    start_time = time.time()
    user_id = current_user['id']
    
    # 1. Rate Limiting Cooldown (Rule 7)
    now = time.time()
    if user_id in user_last_call:
        elapsed = now - user_last_call[user_id]
        if elapsed < 10:
            return jsonify({
                "response": "Please wait a moment before sending another message.",
                "cooldown": 10 - elapsed
            }), 200 # Using 200 as per requested response string
    
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        message = data.get('message', '').strip()
        session_id = data.get('session_id', str(uuid.uuid4()))
        language = data.get('language', current_user.get('preferences', {}).get('language', 'English'))
        use_specific_tool = data.get('use_specific_tool')
        
        if not message:
            return jsonify({"error": "Message is required"}), 400
            
        # 2. Answer Caching (Rule 6)
        import hashlib
        from rag.llm import response_cache
        cache_key = hashlib.md5(f"{user_id}:{session_id}:{message}".encode()).hexdigest()
        
        if cache_key in response_cache:
            logger.info(f"Returning cached answer for message: {message[:50]}...")
            cached_data = response_cache[cache_key]
            return jsonify(cached_data), 200

        logger.info(f"Processing message for user {user_id}: {message[:100]}...")
        
        # Get chat context (last 10 messages)
        chat_history = mongodb.get_chat_history(
            user_id=user_id,
            session_id=session_id,
            limit=10
        )
        
        context_messages = []
        for msg in chat_history:
            if msg.get('role') == 'user':
                context_messages.append(f"User: {msg.get('content', '')}")
            if msg.get('response'):
                context_messages.append(f"Assistant: {msg.get('response', '')}")
        
        conversation_context = "\n".join(context_messages[-6:])
        
        # Intelligent tool detection
        if use_specific_tool:
            tool_used = use_specific_tool
        else:
            tool_used = intelligent_tool_selection(
                message, user_id, language, conversation_context
            )
        
        response = ""
        sources = []
        
        # Update last call time only if we are about to make an LLM call (Rule 7)
        user_last_call[user_id] = now

        # Process based on selected tool
        if tool_used == "rag":
            # STRICT: No fallback to general chat (Rule 3)
            result = process_query(message, user_id, language)
            response = result.get("response", "")
            sources = result.get("sources", [])
        elif tool_used == "web_search":
            response = web_search_tool(message, language)
        elif tool_used == "youtube_search":
            response = youtube_search_tool(message, language)
        elif tool_used == "summarize":
            response = summarize_tool(message, language, conversation_context)
        else:
            # STRICT: Regular chat response (Rule 3)
            response = regular_chat_response(message, language, conversation_context)
        
        # Save to chat history
        chat_data = {
            "user_id": user_id,
            "session_id": session_id,
            "role": "user",
            "content": message,
            "response": response,
            "language": language,
            "tool_used": tool_used,
            "sources": sources,
            "timestamp": time.time()
        }
        
        mongodb.save_chat_message(chat_data)
        
        # Update session timestamp
        mongodb.update_session(session_id, {"updated_at": time.time()})
        
        processing_time = time.time() - start_time
        
        final_response = {
            "response": response,
            "session_id": session_id,
            "chat_title": None, # Title generation disabled (Rule 2 & 8)
            "tool_used": tool_used,
            "language": language,
            "sources": sources,
            "processing_time": processing_time,
            "timestamp": time.time()
        }
        
        # Cache the final response (Rule 6)
        response_cache[cache_key] = final_response
        
        return jsonify(final_response), 200
        
    except Exception as e:
        logger.error(f"Chat error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat/analyze-image', methods=['POST'])
@auth_required
def analyze_image(current_user):
    """Analyze uploaded image using Gemini"""
    start_time = time.time()
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        image_file = request.files['image']
        session_id = request.form.get('session_id', str(uuid.uuid4()))
        language = request.form.get('language', current_user.get('preferences', {}).get('language', 'English'))
        
        if not GOOGLE_AI_AVAILABLE or not google_client:
            return jsonify({"error": "Image analysis is not available"}), 503
        
        # Read image data
        image_data = image_file.read()
        
        # Prepare the image for Gemini
        image_part = types.Part.from_bytes(
            data=image_data,
            mime_type=image_file.mimetype or 'image/jpeg'
        )
        
        # Add language instruction
        lang_instruction = ""
        if language == "Tamil":
            lang_instruction = "IMPORTANT: Respond in Tamil (தமிழ்) only. Use Tamil script and language throughout your response."
        elif language == "Hindi":
            lang_instruction = "IMPORTANT: Respond in Hindi (हिंदी) only. Use Devanagari script and Hindi language throughout your response."
        else:
            lang_instruction = "Respond in English only."
        
        # Create prompt
        prompt = f"""Please analyze this image and provide a detailed description.
        {lang_instruction}
        
        Include:
        1. What the image shows
        2. Key elements and their significance
        3. Any text visible in the image
        4. Overall interpretation or context
        
        Be detailed and descriptive."""
        
        # Generate response using Gemini with fallback
        analysis_result = generate_with_fallback([prompt, image_part])
        
        # Save to chat history
        chat_data = {
            "user_id": current_user['id'],
            "session_id": session_id,
            "role": "user",
            "content": f"[Image uploaded: {image_file.filename}]",
            "response": analysis_result,
            "language": language,
            "tool_used": "image",
            "timestamp": time.time()
        }
        
        mongodb.save_chat_message(chat_data)
        
        # Update session timestamp
        mongodb.update_session(session_id, {"updated_at": time.time()})
        
        processing_time = time.time() - start_time
        
        return jsonify({
            "response": analysis_result,
            "session_id": session_id,
            "tool_used": "image",
            "language": language,
            "processing_time": processing_time,
            "timestamp": time.time()
        }), 200
        
    except Exception as e:
        logger.error(f"Image analysis error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

def intelligent_tool_selection(query: str, user_id: str, language: str, context: str = "") -> str:
    """Intelligently select the best tool based on query and context with regex word boundaries"""
    import re
    query_lower = query.lower()
    
    def contains_word(keywords, text):
        for k in keywords:
            # Check for word boundary
            if re.search(r'\b' + re.escape(k.lower()) + r'\b', text):
                return True
        return False

    # 1. Check for summarize requests (Highest priority)
    if contains_word(['summarize', 'summary', 'sum up', 'shorten'], query_lower):
        return "summarize"

    # 2. Check for explicit tool requests in context
    if context:
        context_lower = context.lower()
        if "youtube" in context_lower or "video" in context_lower:
            if contains_word(['more', 'another', 'other', 'find', 'next'], query_lower):
                return "youtube_search"
        elif "search" in context_lower or "web" in context_lower:
            if contains_word(['more', 'another', 'other', 'find', 'next'], query_lower):
                return "web_search"
    
    # 3. Check for document-specific queries (RAG)
    doc_keywords = [
        'document', 'pdf', 'file', 'upload', 'my doc', 'my file',
        'what does it say', 'tell me about', 'according to', 'based on',
        'in the document', 'from the pdf', 'in my file',
        'resume', 'cv', 'portfolio', 'biodata', 'projects', 'skills', 'experience'
    ]
    if contains_word(doc_keywords, query_lower):
        return "rag"
    
    # 4. Check for YouTube requests
    youtube_keywords = ['youtube', 'video', 'watch', 'tutorial', 'how to video', 'demo']
    if contains_word(youtube_keywords, query_lower):
        return "youtube_search"
    
    # 5. Check for web search requests
    web_keywords = [
        'search for', 'find info', 'look up', 'latest news', 
        'what is happening', 'who is', 'news about', 'google it'
    ]
    if contains_word(web_keywords, query_lower):
        return "web_search"
    
    # 6. For general questions, always use RAG to ensure document grounding (Rule 3)
    greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'who are you', 'what can you do']
    is_greeting = any(contains_word([g], query_lower) for g in greetings)
    
    if not is_greeting:
        return "rag"
    
    return "chat"

# ==================== TOOL IMPLEMENTATIONS ====================

def web_search_tool(query: str, language: str) -> str:
    """Enhanced web search"""
    if not DDGS_AVAILABLE:
        return "Web search is currently unavailable. Please install duckduckgo-search package."
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            
            if not results:
                return f"No web results found for '{query}'"
            
            if language == "Tamil":
                formatted = f"**🌐 '{query}' க்கான வலைத் தேடல் முடிவுகள்:**\n\n"
            elif language == "Hindi":
                formatted = f"**🌐 '{query}' के लिए वेब खोज परिणाम:**\n\n"
            else:
                formatted = f"**🌐 Web Search Results for '{query}':**\n\n"
                
            for i, result in enumerate(results, 1):
                title = result.get('title', 'No title')
                href = result.get('href', 'No URL')
                body = result.get('body', 'No description')[:200]
                formatted += f"{i}. **{title}**\n"
                formatted += f"   🔗 {href}\n"
                formatted += f"   📝 {body}...\n\n"
            
            return formatted
            
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return f"Error performing web search: {str(e)}"

def youtube_search_tool(query: str, language: str) -> str:
    """Search YouTube videos"""
    if not YOUTUBE_API_KEY:
        return "YouTube search is not configured. Please add YOUTUBE_API_KEY to .env file."
    
    if not GOOGLE_API_AVAILABLE:
        return "YouTube search requires google-api-python-client. Please install it."
    
    try:
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        
        search_response = youtube.search().list(
            q=query,
            part='id,snippet',
            maxResults=5,
            type='video',
            order='relevance'
        ).execute()
        
        videos = []
        for item in search_response['items']:
            video_id = item['id']['videoId']
            videos.append({
                'title': item['snippet']['title'],
                'description': item['snippet']['description'][:150] + "..." if len(item['snippet']['description']) > 150 else item['snippet']['description'],
                'channel': item['snippet']['channelTitle'],
                'url': f"https://www.youtube.com/watch?v={video_id}",
                'published': item['snippet']['publishedAt'][:10]
            })
        
        if not videos:
            return f"No YouTube videos found for '{query}'"
        
        if language == "Tamil":
            formatted = f"**🎥 '{query}' க்கான YouTube வீடியோக்கள்:**\n\n"
        elif language == "Hindi":
            formatted = f"**🎥 '{query}' के लिए YouTube वीडियो:**\n\n"
        else:
            formatted = f"**🎥 YouTube videos for '{query}':**\n\n"
            
        for i, video in enumerate(videos, 1):
            formatted += f"{i}. **{video['title']}**\n"
            formatted += f"   📺 {video['channel']}\n"
            formatted += f"   📅 {video['published']}\n"
            formatted += f"   🔗 {video['url']}\n"
            formatted += f"   📝 {video['description']}\n\n"
        
        return formatted
        
    except Exception as e:
        logger.error(f"YouTube search error: {e}")
        return f"Error searching YouTube: {str(e)}"

def summarize_tool(query: str, language: str, context: str = "") -> str:
    """Summarize text or documents"""
    if not GOOGLE_AI_AVAILABLE or not google_client:
        return "Summarization is currently unavailable."
    
    try:
        # Check if user wants to summarize uploaded documents
        # If so, they should be routed to RAG, but we can handle it here if needed
        # For now, let's just make sure we don't block it with a generic message if it got here
        pass
        
        # Summarize the current query if it contains substantial text (more than just "summarize this")
        # Otherwise fall back to summarizing the conversation context
        if len(query.split()) > 10:
            text_to_summarize = query
        else:
            text_to_summarize = context if context else query
        
        lang_instruction = ""
        if language == "Tamil":
            lang_instruction = "IMPORTANT: Summarize in Tamil (தமிழ்) only."
        elif language == "Hindi":
            lang_instruction = "IMPORTANT: Summarize in Hindi (हिंदी) only."
        
        prompt = f"""Please provide a concise summary of the following:
{lang_instruction}

Text to summarize:
{text_to_summarize}

Concise Summary:"""
        
        # Use fallback generation
        return generate_with_fallback(prompt)
        
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        return f"Error creating summary: {str(e)}"

def regular_chat_response(message: str, language: str, context: str = "") -> str:
    """Regular AI chat response using new Google SDK with context awareness"""
    if not GOOGLE_AI_AVAILABLE or not google_client:
        return "AI model not available. Please check configuration."
    
    try:
        # Add language instruction
        lang_instruction = ""
        if language == "Tamil":
            lang_instruction = "IMPORTANT: Respond in Tamil (தமிழ்) only. Use Tamil script and language throughout your response."
        elif language == "Hindi":
            lang_instruction = "IMPORTANT: Respond in Hindi (हिंदी) only. Use Devanagari script and Hindi language throughout your response."
        
        # Include conversation context if available
        context_section = f"\n\nPrevious Conversation:\n{context}" if context else ""
        
        # Rule 3 Strict Enforcement: Only answer if information is present
        # Since this is a general chat fallback, and Rule 3 forbids world knowledge:
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
        if any(g in message.lower() for g in greetings):
             # Ground the assistant's persona
             prompt = f"""You are a RAG AI assistant. You can only answer questions based on uploaded documents.
             {lang_instruction}
             Message: {message}
             Answer:"""
             return generate_with_fallback(prompt)
        
        return "I couldn’t find this information in your uploaded documents."
        
    except Exception as e:
        logger.error(f"Chat response error: {e}")
        return f"I encountered an error: {str(e)}"

# ==================== DOCUMENT ENDPOINTS ====================

@app.route('/api/documents/upload', methods=['POST'])
@auth_required
def upload_document(current_user):
    """Upload and process a PDF document"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"error": "Only PDF files are supported"}), 400

        # Save file temporarily
        filename = secure_filename(file.filename)
        temp_dir = Path(__file__).parent.parent / 'temp'
        temp_dir.mkdir(exist_ok=True)
        file_path = temp_dir / filename
        file.save(str(file_path))

        # Process document through RAG system
        if RAG_AVAILABLE:
            result = process_document(file, current_user['id'])
            
            # Clean up temp file
            if file_path.exists():
                file_path.unlink()
            
            if result.get('success'):
                return jsonify({
                    "success": True,
                    "message": result['message'],
                    "document_id": result['document_id'],
                    "filename": filename
                }), 201
            else:
                return jsonify({
                    "success": False,
                    "error": result.get('error', 'Unknown error')
                }), 400
        else:
            return jsonify({"error": "RAG system not available"}), 500

    except Exception as e:
        logger.error(f"Upload error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/documents/list', methods=['GET'])
@auth_required
def list_documents(current_user):
    """List user's documents"""
    try:
        if RAG_AVAILABLE:
            documents = list_user_documents(current_user['id'])
            return jsonify({
                "documents": documents,
                "count": len(documents)
            }), 200
        else:
            return jsonify({"error": "RAG system not available"}), 500
        
    except Exception as e:
        logger.error(f"List documents error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/documents/<document_id>', methods=['DELETE'])
@auth_required
def delete_document(current_user, document_id):
    """Delete a document"""
    try:
        if RAG_AVAILABLE and hasattr(vector_store, 'delete_document'):
            success = vector_store.delete_document(document_id, current_user['id'])
            if success:
                return jsonify({"message": "Document deleted successfully"}), 200
            else:
                return jsonify({"error": "Document not found"}), 404
        else:
            return jsonify({"error": "RAG system not available"}), 500
        
    except Exception as e:
        logger.error(f"Delete document error: {e}")
        return jsonify({"error": str(e)}), 500

# ==================== SESSION ENDPOINTS ====================

@app.route('/api/sessions', methods=['GET'])
@auth_required
def list_sessions(current_user):
    """List user's chat sessions"""
    try:
        sessions = mongodb.get_user_sessions(current_user['id'])
        return jsonify({
            "sessions": sessions,
            "count": len(sessions)
        }), 200
        
    except Exception as e:
        logger.error(f"List sessions error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/sessions/<session_id>', methods=['DELETE'])
@auth_required
def delete_session(current_user, session_id):
    """Delete a chat session"""
    try:
        success = mongodb.delete_session(session_id, current_user['id'])
        if success:
            return jsonify({"message": "Session deleted successfully"}), 200
        else:
            return jsonify({"error": "Session not found"}), 404
        
    except Exception as e:
        logger.error(f"Delete session error: {e}")
        return jsonify({"error": str(e)}), 500

# ==================== HEALTH & STATIC ====================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy" if RAG_AVAILABLE else "degraded",
        "google_ai_configured": GOOGLE_AI_AVAILABLE and GOOGLE_API_KEY is not None,
        "rag_initialized": rag_initialized,
        "rag_available": RAG_AVAILABLE,
        "web_search_available": DDGS_AVAILABLE,
        "youtube_search_available": GOOGLE_API_AVAILABLE and YOUTUBE_API_KEY is not None,
        "image_analysis_available": GOOGLE_AI_AVAILABLE and GOOGLE_API_KEY is not None,
        "timestamp": time.time()
    })

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    """Serve the frontend application"""
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

# ==================== STARTUP ====================

def initialize_services():
    """Initialize services on startup"""
    logger.info("🚀 Starting RAG AI Assistant with Intelligent Tool Selection...")
    
    # Create necessary directories
    temp_dir = Path(__file__).parent.parent / 'temp'
    temp_dir.mkdir(exist_ok=True)
    
    logger.info("✅ Directories created")
    logger.info(f"📚 RAG system: {'✅ Ready' if rag_initialized else '❌ Failed'}")
    logger.info(f"🔧 Google AI: {'✅ Configured' if GOOGLE_AI_AVAILABLE and GOOGLE_API_KEY else '❌ Not configured'}")
    logger.info(f"🌐 Web Search: {'✅ Available' if DDGS_AVAILABLE else '❌ Not available'}")
    logger.info(f"🎥 YouTube Search: {'✅ Available' if GOOGLE_API_AVAILABLE and YOUTUBE_API_KEY else '❌ Not available'}")
    logger.info(f"🖼️ Image Analysis: {'✅ Available' if GOOGLE_AI_AVAILABLE and GOOGLE_API_KEY else '❌ Not available'}")

# ==================== MAIN ====================

if __name__ == '__main__':
    PORT = int(os.getenv('PORT', 5000))
    
    # Initialize on startup
    initialize_services()
    
    print(f"\n{'='*60}")
    print("🤖 RAG AI ASSISTANT WITH INTELLIGENT TOOL SELECTION")
    print(f"{'='*60}")
    print("🔧 Features:")
    print("  • Automatic tool selection based on query")
    print("  • Image upload and analysis with Gemini")
    print("  • PDF upload and RAG processing")
    print("  • Context-aware responses")
    print("  • RAG first, LLM fallback strategy")
    print(f"🌐 Web Interface: http://localhost:{PORT}")
    print(f"📁 API Base: http://localhost:{PORT}/api")
    print(f"{'='*60}\n")
    
    app.run(host='0.0.0.0', port=PORT, debug=os.getenv('DEBUG', 'False').lower() == 'true')