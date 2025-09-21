# main.py - FastAPI Backend for SIH Farmer Assistant
from dotenv import load_dotenv

load_dotenv()
import os
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import mysql.connector
from mysql.connector.pooling import MySQLConnectionPool
from datetime import datetime, timedelta
from jose import jwt
import bcrypt
from typing import Optional, List
import pickle
import numpy as np
import pandas as pd
from pydantic import BaseModel, EmailStr
import json
import logging
import joblib
from datetime import datetime
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', 'password'),
    'database': os.getenv('DB_NAME', 'farmer_assistant'),
    'pool_size': 10,
    'pool_reset_session': True
}

# Initialize MySQL connection pool

db_pool = None
try:
    db_pool = MySQLConnectionPool(**DB_CONFIG)
    logger.info("✅ MySQL connection pool created successfully.")
except Exception as e:
    logger.error(f"❌ Error creating MySQL connection pool: {e}")

def get_db_connection():
    if db_pool is None:
        raise Exception("Database connection pool is not initialized.")
    return db_pool.get_connection()

# JWT configuration
JWT_SECRET = os.getenv('JWT_SECRET', 'your-secret-key-change-in-production')
JWT_ALGORITHM = 'HS256'
JWT_EXPIRATION_HOURS = 24 * 7  # 1 week

# Global variables
ml_models = {}

# Pydantic models for API
class UserRegister(BaseModel):
    name: str
    email: EmailStr
    password: str
    phone: Optional[str] = None
    location: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class YieldPredictionRequest(BaseModel):
    crop: str
    season: str
    state: str
    area: float
    fertilizer: float
    pesticide: float
    rainfall: float 

class CropRecommendationRequest(BaseModel):
    season: str
    state: str
    area: float
    fertilizer: float
    pesticide: float
    rainfall:float

class PostCreate(BaseModel):
    title: str
    body: str
    category: Optional[str] = 'general'
    image_url: Optional[str] = None

class SoilCropRecommendationRequest(BaseModel):
    temperature: float
    humidity: float
    moisture: float
    soil_type: str
    nitrogen: float
    phosphorous: float
    potassium: float
    fertilizer: str

class CommentCreate(BaseModel):
    body: str

class LikeToggle(BaseModel):
    post_id: int

# ML Model classes
class YieldPredictor:
    """Wrapper around trained ML model for yield prediction"""

    def __init__(self, model_path="Model\yield_prediction_model.pkl"):
        try:
            self.model = joblib.load(model_path)
            logger.info("Yield prediction model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None

    def predict(self, features: dict):
        """Predict crop yield given farmer input features"""
        try:
            if self.model is None:
                raise ValueError("Model not loaded")

            # Convert dict to DataFrame (columns must match training features)
            input_df = pd.DataFrame([{
                "Crop": features.get("crop", "").capitalize(),
                "Season": features.get("season", "").capitalize(),
                "State": features.get("state", "").title(),
                "Crop_Year": datetime.now().year,
                "Area": features.get("area", 1.0),
                "Annual_Rainfall": features.get("rainfall", 100.0),
                "Fertilizer": features.get("fertilizer", 0.0),
                "Pesticide": features.get("pesticide", 0.0),
                "Production":16435941   
            }])

            # Run prediction
            prediction = self.model.predict(input_df)

            return round(float(prediction[0]), 2)

        except Exception as e:
            logger.error(f"Error in yield prediction: {e}")
            return None

class CropRecommendationModel:
    """Use trained ML model for crop recommendation"""
    
    def __init__(self, model_path="Model\crop_recommendation_model.pkl"):
        try:
            self.model = joblib.load(model_path)
            logger.info("Crop recommendation model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
    
    def predict(self, features: dict):
        """Recommend crops based on input features"""
        try:
            if self.model is None:
                # fallback: just return a list of strings
                return ["wheat"]

            # Convert features into DataFrame (columns must match training)
            input_df = pd.DataFrame([{
    "Season": features.get("season", "").capitalize(),
    "State": features.get("state", "").title(),
    "Crop_Year": datetime.now().year,
    "Area": features.get("area", 1.0),
    "Annual_Rainfall": features.get("rainfall", 100.0),
    "Fertilizer": features.get("fertilizer", 0.0),
    "Pesticide": features.get("pesticide", 0.0),
    "Production": 16435941,  # default value
    "Yield": 79             # placeholder, model predicts this
}])
            
            # Make prediction
            pred_class = self.model.predict(input_df)[0]

            # Try getting probabilities if supported
            recommended = [pred_class]  # always include predicted class
            try:
                crop_proba = self.model.predict_proba(input_df)[0]
                classes = self.model.classes_
                top_indices = crop_proba.argsort()[-3:][::-1]  # top 3
                recommended = [classes[i] for i in top_indices]
            except AttributeError:
                logger.warning("Model does not support predict_proba; returning only predicted class.")

            return recommended

        except Exception as e:
            logger.error(f"Error in crop recommendation: {e}")
            return ["wheat"]  # fallback


class SoilCropRecommendationModel:
    def __init__(self, model_path="Model/crop_recommendation_soil_model.pkl"):
        self.model = joblib.load(model_path)
        # Save the columns the model expects
        self.columns = getattr(self.model, "feature_names_in_", None)  

    def predict(self, features: dict):
        # Create DataFrame with raw features
        input_df = pd.DataFrame([{
            "Temperature": features["temperature"],
            "Humidity": features["humidity"],
            "Moisture": features["moisture"],
            "Soil Type": features["soil_type"].capitalize(),
            "Nitrogen": features["nitrogen"],
            "Phosphorous": features["phosphorous"],
            "Potassium": features["potassium"],
            "Fertilizer Name": features["fertilizer"]
        }])

        # One-hot encode categorical features
        input_df = pd.get_dummies(input_df)

        # Align input with model columns
        input_df = input_df.reindex(columns=self.model.feature_names_in_, fill_value=0)

        # Predict top 3 crops
        probs = self.model.predict_proba(input_df)[0]
        classes = self.model.classes_
        top_indices = probs.argsort()[-3:][::-1]
        return [classes[i] for i in top_indices]


    

# Database functions
def init_db():
    """Initialize database connection pool"""
    global db_pool
    try:
        db_pool = MySQLConnectionPool(**DB_CONFIG)
        logger.info("Database connection pool initialized")
        
        # Test connection
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            logger.info("Database connection test successful")
            
    except mysql.connector.Error as e:
        logger.error(f"Database initialization error: {e}")
        raise

# Authentication functions
def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_jwt_token(user_id: int, email: str) -> str:
    """Create JWT token"""
    payload = {
        'user_id': user_id,
        'email': email,
        'exp': datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_jwt_token(token: str) -> dict:
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Security dependency
security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from JWT token"""
    payload = verify_jwt_token(credentials.credentials)
    
    with get_db_connection() as conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, name, email, location FROM users WHERE id = %s", (payload['user_id'],))
        user = cursor.fetchone()
        
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        
        return user

# Weather and soil data functions
def get_weather_data(state: str, season: str):
    """Get weather data for state and season"""
    with get_db_connection() as conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT rainfall, temperature, humidity FROM weather_data WHERE state = %s AND season = %s",
            (state, season)
        )
        data = cursor.fetchone()
        
        if not data:
            # Return default values
            return {
                'rainfall': 100.0,
                'temperature': 25.0,
                'humidity': 65.0
            }
        
        return data

def get_soil_data(state: str):
    """Get soil data for state"""
    with get_db_connection() as conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT ph_level, nitrogen, phosphorus, potassium, organic_matter FROM soil_data WHERE state = %s",
            (state,)
        )
        data = cursor.fetchone()
        
        if not data:
            # Return default values
            return {
                'ph_level': 7.0,
                'nitrogen': 240.0,
                'phosphorus': 30.0,
                'potassium': 180.0,
                'organic_matter': 3.0
            }
        
        return data

# App lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    yield
    # Shutdown
    if db_pool:
        db_pool.disconnect()

# Create FastAPI app
app = FastAPI(
    title="SIH Farmer Assistant API",
    description="Smart farming platform with ML-powered predictions and community features",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

# Authentication endpoints
@app.post("/auth/register")
async def register(user: UserRegister):
    """Register new user"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check if user exists
            cursor.execute("SELECT id FROM users WHERE email = %s", (user.email,))
            if cursor.fetchone():
                raise HTTPException(status_code=400, detail="Email already registered")
            
            # Create user
            hashed_password = hash_password(user.password)
            cursor.execute(
                """INSERT INTO users (name, email, password_hash, phone, location) 
                   VALUES (%s, %s, %s, %s, %s)""",
                (user.name, user.email, hashed_password, user.phone, user.location)
            )
            conn.commit()
            
            user_id = cursor.lastrowid
            token = create_jwt_token(user_id, user.email)
            
            return {
                "message": "User registered successfully",
                "token": token,
                "user": {
                    "id": user_id,
                    "name": user.name,
                    "email": user.email,
                    "location": user.location
                }
            }
            
    except mysql.connector.Error as e:
        logger.error(f"Database error in register: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/auth/login")
async def login(user: UserLogin):
    """Login user"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT id, name, email, password_hash, location FROM users WHERE email = %s",
                (user.email,)
            )
            db_user = cursor.fetchone()
            
            if not db_user or not verify_password(user.password, db_user['password_hash']):
                raise HTTPException(status_code=401, detail="Invalid credentials")
            
            token = create_jwt_token(db_user['id'], db_user['email'])
            
            return {
                "message": "Login successful",
                "token": token,
                "user": {
                    "id": db_user['id'],
                    "name": db_user['name'],
                    "email": db_user['email'],
                    "location": db_user['location']
                }
            }
            
    except mysql.connector.Error as e:
        logger.error(f"Database error in login: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

# ML Prediction endpoints
@app.post("/predict/yield")
async def predict_yield(
    request: YieldPredictionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Predict crop yield"""
    try:
        
        # Prepare features for ML model
        features = {
            'crop': request.crop.lower(),
            'season': request.season.lower(),
            'state': request.state.lower(),
            'area': request.area,
            'fertilizer': request.fertilizer,
            'pesticide': request.pesticide,
            'rainfall': request.rainfall,
        }
        yield_predictor = YieldPredictor()
        pred = yield_predictor.predict(features)
        
        
        return {
            "predicted_yield": pred,
        }
        
    except Exception as e:
        logger.error(f"Error in yield prediction: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.post("/predict/crop")
async def recommend_crop(
    request: CropRecommendationRequest,
):
    """Recommend suitable crops"""
    try:
        
        
        # Prepare features for ML model
        features = {
            'season': request.season.lower(),
            'state': request.state.lower(),
            'area': request.area,
            'fertilizer': request.fertilizer,
            'pesticide': request.pesticide,
            'rainfall': request.rainfall,
        }
        crop_predictor = CropRecommendationModel()
        pred = crop_predictor.predict(features)
        # Get crop recommendations
        
        return {
            "recommended_crops": pred,
            "recommendations": [
                "Consider crop rotation for better soil health",
                "Plan irrigation based on weather forecast",
                "Use organic fertilizers when possible"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error in crop recommendation: {e}")
        raise HTTPException(status_code=500, detail="Recommendation failed")

@app.post("/predict/soil-crop")
async def recommend_soil_crop(request: SoilCropRecommendationRequest):
    """Recommend suitable crops based on soil and environmental conditions"""
    try:
        # Prepare features for ML model
        features = {
            "temperature": request.temperature,
            "humidity": request.humidity,
            "moisture": request.moisture,
            "soil_type": request.soil_type,
            "nitrogen": request.nitrogen,
            "phosphorous": request.phosphorous,
            "potassium": request.potassium,
            "fertilizer": request.fertilizer
        }

        soil_crop_predictor = SoilCropRecommendationModel()
        pred = soil_crop_predictor.predict(features)

        # Return top 3 recommended crops
        return {
            "recommended_crops": pred,
            "recommendations": [
                "Maintain proper soil moisture levels",
                "Use balanced fertilizers based on soil test",
                "Consider crop rotation for better soil health"
            ]
        }

    except Exception as e:
        logger.error(f"Error in soil crop recommendation: {e}")
        raise HTTPException(status_code=500, detail="Soil crop recommendation failed")
# Community endpoints
@app.get("/community/posts")
async def get_posts(
    limit: int = 20,
    offset: int = 0,
    category: Optional[str] = None
):
    """Get community posts"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            
            query = """
                SELECT p.*, u.name as author_name, u.location as author_location
                FROM posts p 
                JOIN users u ON p.user_id = u.id
            """
            params = []
            
            if category:
                query += " WHERE p.category = %s"
                params.append(category)
            
            query += " ORDER BY p.created_at DESC LIMIT %s OFFSET %s"
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            posts = cursor.fetchall()
            
            # Get comments for each post
            for post in posts:
                cursor.execute(
                    """SELECT c.*, u.name as commenter_name 
                       FROM comments c 
                       JOIN users u ON c.user_id = u.id 
                       WHERE c.post_id = %s 
                       ORDER BY c.created_at ASC LIMIT 3""",
                    (post['id'],)
                )
                post['recent_comments'] = cursor.fetchall()
            
            return {"posts": posts}
            
    except Exception as e:
        logger.error(f"Error fetching posts: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch posts")

@app.post("/community/posts")
async def create_post(
    post: PostCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create new community post"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO posts (user_id, title, body, category, image_url) 
                   VALUES (%s, %s, %s, %s, %s)""",
                (current_user['id'], post.title, post.body, post.category, post.image_url)
            )
            conn.commit()
            
            post_id = cursor.lastrowid
            
            return {
                "message": "Post created successfully",
                "post_id": post_id
            }
            
    except Exception as e:
        logger.error(f"Error creating post: {e}")
        raise HTTPException(status_code=500, detail="Failed to create post")

@app.post("/community/posts/{post_id}/comments")
async def add_comment(
    post_id: int,
    comment: CommentCreate,
    current_user: dict = Depends(get_current_user)
):
    """Add comment to post"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check if post exists
            cursor.execute("SELECT id FROM posts WHERE id = %s", (post_id,))
            if not cursor.fetchone():
                raise HTTPException(status_code=404, detail="Post not found")
            
            # Add comment
            cursor.execute(
                """INSERT INTO comments (post_id, user_id, body) 
                   VALUES (%s, %s, %s)""",
                (post_id, current_user['id'], comment.body)
            )
            conn.commit()
            
            return {"message": "Comment added successfully"}
            
    except Exception as e:
        logger.error(f"Error adding comment: {e}")
        raise HTTPException(status_code=500, detail="Failed to add comment")

@app.post("/community/posts/{post_id}/like")
async def toggle_like(
    post_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Toggle like on post"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check if post exists
            cursor.execute("SELECT id FROM posts WHERE id = %s", (post_id,))
            if not cursor.fetchone():
                raise HTTPException(status_code=404, detail="Post not found")
            
            # Check if user already liked
            cursor.execute(
                "SELECT id FROM likes WHERE post_id = %s AND user_id = %s",
                (post_id, current_user['id'])
            )
            existing_like = cursor.fetchone()
            
            if existing_like:
                # Unlike
                cursor.execute(
                    "DELETE FROM likes WHERE post_id = %s AND user_id = %s",
                    (post_id, current_user['id'])
                )
                action = "unliked"
            else:
                # Like
                cursor.execute(
                    "INSERT INTO likes (post_id, user_id) VALUES (%s, %s)",
                    (post_id, current_user['id'])
                )
                action = "liked"
            
            conn.commit()
            
            # Get updated like count
            cursor.execute("SELECT likes_count FROM posts WHERE id = %s", (post_id,))
            likes_count = cursor.fetchone()[0]
            
            return {
                "message": f"Post {action} successfully",
                "likes_count": likes_count,
                "action": action
            }
            
    except Exception as e:
        logger.error(f"Error toggling like: {e}")
        raise HTTPException(status_code=500, detail="Failed to toggle like")

@app.get("/user/predictions")
async def get_user_predictions(
    current_user: dict = Depends(get_current_user),
    limit: int = 10
):
    """Get user's prediction history"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                """SELECT * FROM predictions 
                   WHERE user_id = %s 
                   ORDER BY created_at DESC 
                   LIMIT %s""",
                (current_user['id'], limit)
            )
            predictions = cursor.fetchall()
            
            return {"predictions": predictions}
            
    except Exception as e:
        logger.error(f"Error fetching predictions: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch predictions")


@app.get("/test-db")
def test_db():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT DATABASE();")
        db_name = cursor.fetchone()
        cursor.close()
        conn.close()
        return {"message": "✅ Database connected successfully!", "database": db_name}
    except Exception as e:
        logger.error(f"❌ Database test failed: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)