from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_bcrypt import Bcrypt
from datetime import timedelta
import os
from dotenv import load_dotenv
import logging
from .mongodb import mongodb
from bson import ObjectId

load_dotenv()
logger = logging.getLogger(__name__)

# Initialize extensions
bcrypt = Bcrypt()
jwt = JWTManager()

def init_auth(app):
    """Initialize authentication extensions"""
    app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-this')
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
    
    bcrypt.init_app(app)
    jwt.init_app(app)
    
    return app

class AuthManager:
    @staticmethod
    def register_user(username: str, email: str, password: str, full_name: str = None):
        """Register a new user"""
        # Check if user exists
        existing_user = mongodb.get_user_by_email(email)
        if existing_user:
            return None, "User already exists"
        
        # Hash password
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        
        # Create user data
        user_data = {
            "username": username,
            "email": email,
            "password": hashed_password,
            "full_name": full_name or username,
            "role": "user",
            "is_active": True,
            "preferences": {
                "language": "English",
                "use_tools": True,
                "theme": "light"
            }
        }
        
        # Save to database
        user_id = mongodb.create_user(user_data)
        if not user_id:
            return None, "Failed to create user"
        
        # Ensure id is a string and create access token
        user_id_str = str(user_id)
        access_token = create_access_token(identity=user_id_str)
        
        user_data["id"] = user_id_str
        # Remove any leftover ObjectId copies to make JSON serializable
        if "_id" in user_data:
            try:
                # convert if it's an ObjectId
                user_data["_id"] = str(user_data["_id"])
            except Exception:
                del user_data["_id"]
        del user_data["password"]  # Don't return password
        
        return {"user": user_data, "token": access_token}, None
    
    @staticmethod
    def login_user(email: str, password: str):
        """Login user"""
        # Find user
        user = mongodb.get_user_by_email(email)
        if not user:
            return None, "Invalid credentials"
        
        # Check password
        if not bcrypt.check_password_hash(user['password'], password):
            return None, "Invalid credentials"
        
        # Check if user is active
        if not user.get('is_active', True):
            return None, "Account is disabled"
        
        # Create access token
        user_id = str(user['_id'])
        access_token = create_access_token(identity=user_id)
        
        # Prepare user data
        user_data = {
            "id": user_id,
            "username": user['username'],
            "email": user['email'],
            "full_name": user.get('full_name', user['username']),
            "role": user.get('role', 'user'),
            "preferences": user.get('preferences', {})
        }
        
        return {"user": user_data, "token": access_token}, None
    
    @staticmethod
    def get_current_user():
        """Get current user from JWT"""
        user_id = get_jwt_identity()
        if not user_id:
            return None
        
        user = mongodb.get_user_by_id(user_id)
        if not user:
            return None
        
        user_data = {
            "id": str(user['_id']),
            "username": user['username'],
            "email": user['email'],
            "full_name": user.get('full_name', user['username']),
            "role": user.get('role', 'user'),
            "preferences": user.get('preferences', {})
        }
        
        return user_data
    
    @staticmethod
    def update_user_preferences(user_id: str, preferences: dict):
        """Update user preferences"""
        user = mongodb.get_user_by_id(user_id)
        if not user:
            return False
        
        current_prefs = user.get('preferences', {})
        current_prefs.update(preferences)
        
        return mongodb.update_user(user_id, {"preferences": current_prefs})