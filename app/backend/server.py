"from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException, Request, Response, Query
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone, timedelta
import base64
from emergentintegrations.llm.chat import LlmChat, UserMessage, ImageContent
from pypdf import PdfReader
from pdf2image import convert_from_bytes
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import tempfile
import httpx

# NEW: Import for text extraction and analysis
import pytesseract
from PIL import Image
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter
import string

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get('DB_NAME', 'test_database')]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix=\"/api\")


# Define Models
class StatusCheck(BaseModel):
    model_config = ConfigDict(extra=\"ignore\")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StatusCheckCreate(BaseModel):
    client_name: str

class ImageAnalysisResponse(BaseModel):
    analysis: str
    analysis_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class TextExtractionResponse(BaseModel):
    text: str
    analysis_id: str
    word_count: int
    char_count: int

class SummaryResponse(BaseModel):
    summary: str
    original_sentences: int
    summary_sentences: int

class ImportantPointsResponse(BaseModel):
    points: List[str]
    total_points: int

class QAResponse(BaseModel):
    question: str
    answer: str
    confidence: float

class User(BaseModel):
    user_id: str
    email: str
    name: str
    picture: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class UserSession(BaseModel):
    user_id: str
    session_token: str
    expires_at: datetime
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class AnalysisHistory(BaseModel):
    analysis_id: str
    user_id: str
    filename: str
    file_type: str
    analysis: str
    timestamp: datetime


# Helper function to get session token from cookies or headers
def get_session_token(request: Request) -> Optional[str]:
    \"\"\"Extract session token from cookies or Authorization header\"\"\"
    # First check cookies
    token = request.cookies.get(\"session_token\")
    if token:
        return token
    
    # Fallback to Authorization header
    auth_header = request.headers.get(\"Authorization\")
    if auth_header and auth_header.startswith(\"Bearer \"):
        return auth_header.split(\" \")[1]
    
    return None


# Helper function to get current user
async def get_current_user(request: Request) -> Optional[User]:
    \"\"\"Get current authenticated user from session token\"\"\"
    session_token = get_session_token(request)
    if not session_token:
        return None
    
    # Find session in database
    session_doc = await db.user_sessions.find_one(
        {\"session_token\": session_token},
        {\"_id\": 0}
    )
    
    if not session_doc:
        return None
    
    # Check if session expired
    expires_at = session_doc[\"expires_at\"]
    if isinstance(expires_at, str):
        expires_at = datetime.fromisoformat(expires_at)
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    
    if expires_at < datetime.now(timezone.utc):
        return None
    
    # Get user from database
    user_doc = await db.users.find_one(
        {\"user_id\": session_doc[\"user_id\"]},
        {\"_id\": 0}
    )
    
    if not user_doc:
        return None
    
    # Handle datetime fields
    if isinstance(user_doc.get('created_at'), str):
        user_doc['created_at'] = datetime.fromisoformat(user_doc['created_at'])
    
    return User(**user_doc)


# NEW: Text processing utilities
def clean_text(text: str) -> str:
    \"\"\"Clean and normalize text\"\"\"
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation for sentence detection
    text = text.strip()
    return text

def extract_text_from_image(image_bytes: bytes) -> str:
    \"\"\"Extract text from image using Tesseract OCR\"\"\"
    try:
        image = Image.open(BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
        return clean_text(text)
    except Exception as e:
        logging.error(f\"OCR extraction error: {str(e)}\")
        return \"\"

def extract_text_from_pdf_pages(pdf_bytes: bytes) -> str:
    \"\"\"Extract text from PDF\"\"\"
    try:
        pdf_reader = PdfReader(BytesIO(pdf_bytes))
        text = \"\"
        for page in pdf_reader.pages:
            text += page.extract_text() + \"
\"
        return clean_text(text)
    except Exception as e:
        logging.error(f\"PDF text extraction error: {str(e)}\")
        return \"\"

def summarize_text(text: str, num_sentences: int = 5) -> str:
    \"\"\"Summarize text using TF-IDF and sentence scoring\"\"\"
    if not text or len(text.strip()) < 50:
        return text
    
    try:
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        if len(sentences) <= num_sentences:
            return text
        
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
        
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
        except ValueError:
            # If TF-IDF fails, return first N sentences
            return ' '.join(sentences[:num_sentences])
        
        # Calculate sentence scores based on TF-IDF
        sentence_scores = tfidf_matrix.sum(axis=1).A1
        
        # Get top N sentences
        top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
        top_indices = sorted(top_indices)  # Keep original order
        
        summary_sentences = [sentences[i] for i in top_indices]
        return ' '.join(summary_sentences)
        
    except Exception as e:
        logging.error(f\"Summarization error: {str(e)}\")
        # Fallback to first N sentences
        sentences = sent_tokenize(text)
        return ' '.join(sentences[:num_sentences])

def extract_important_points(text: str, num_points: int = 5) -> List[str]:
    \"\"\"Extract important points using keyword frequency and sentence scoring\"\"\"
    if not text or len(text.strip()) < 50:
        return [text] if text else []
    
    try:
        sentences = sent_tokenize(text)
        
        if len(sentences) <= num_points:
            return sentences
        
        # Tokenize and remove stopwords
        stop_words = set(stopwords.words('english'))
        
        # Calculate word frequencies
        words = word_tokenize(text.lower())
        words = [w for w in words if w.isalnum() and w not in stop_words and len(w) > 3]
        word_freq = Counter(words)
        
        # Score sentences based on keyword frequency
        sentence_scores = []
        for sentence in sentences:
            sentence_words = word_tokenize(sentence.lower())
            score = sum(word_freq.get(w, 0) for w in sentence_words if w in word_freq)
            
            # Bonus for sentences with numbers (often important facts)
            if any(char.isdigit() for char in sentence):
                score *= 1.2
            
            # Bonus for sentences that are not too short or too long
            word_count = len(sentence_words)
            if 5 <= word_count <= 25:
                score *= 1.1
                
            sentence_scores.append((sentence, score))
        
        # Sort by score and get top N
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        important_sentences = [s[0] for s in sentence_scores[:num_points]]
        
        # Return in original order
        ordered_points = []
        for sentence in sentences:
            if sentence in important_sentences:
                ordered_points.append(sentence)
        
        return ordered_points
        
    except Exception as e:
        logging.error(f\"Important points extraction error: {str(e)}\")
        sentences = sent_tokenize(text)
        return sentences[:num_points]

def answer_question(text: str, question: str) -> tuple[str, float]:
    \"\"\"Answer question based on text content using similarity matching\"\"\"
    if not text or not question:
        return \"No text or question provided.\", 0.0
    
    try:
        sentences = sent_tokenize(text)
        
        if not sentences:
            return \"No content found in the document.\", 0.0
        
        # Add question to the mix for vectorization
        all_texts = sentences + [question]
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(stop_words='english')
        
        try:
            tfidf_matrix = vectorizer.fit_transform(all_texts)
        except ValueError:
            return \"Unable to process the text for question answering.\", 0.0
        
        # Question vector is the last one
        question_vector = tfidf_matrix[-1]
        sentence_vectors = tfidf_matrix[:-1]
        
        # Calculate cosine similarity
        similarities = cosine_similarity(question_vector, sentence_vectors).flatten()
        
        # Get best matching sentence
        best_idx = similarities.argmax()
        best_score = similarities[best_idx]
        
        if best_score < 0.1:
            return \"I couldn't find a relevant answer in the document.\", best_score
        
        # Return the best matching sentence and surrounding context
        answer_sentences = []
        
        # Add previous sentence if available
        if best_idx > 0:
            answer_sentences.append(sentences[best_idx - 1])
        
        # Add best sentence
        answer_sentences.append(sentences[best_idx])
        
        # Add next sentence if available
        if best_idx < len(sentences) - 1:
            answer_sentences.append(sentences[best_idx + 1])
        
        answer = ' '.join(answer_sentences)
        
        return answer, float(best_score)
        
    except Exception as e:
        logging.error(f\"Question answering error: {str(e)}\")
        return f\"Error processing question: {str(e)}\", 0.0


# Routes
@api_router.get(\"/\")
async def root():
    return {\"message\": \"LifeLens AI Backend\"}

@api_router.post(\"/status\", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.model_dump()
    status_obj = StatusCheck(**status_dict)
    
    doc = status_obj.model_dump()
    doc['timestamp'] = doc['timestamp'].isoformat()
    
    _ = await db.status_checks.insert_one(doc)
    return status_obj

@api_router.get(\"/status\", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find({}, {\"_id\": 0}).to_list(1000)
    
    for check in status_checks:
        if isinstance(check['timestamp'], str):
            check['timestamp'] = datetime.fromisoformat(check['timestamp'])
    
    return status_checks


# Auth Routes
@api_router.post(\"/auth/session\")
async def create_session(request: Request, response: Response):
    \"\"\"Exchange session_id for session_token and user data\"\"\"
    # Get session_id from header
    session_id = request.headers.get(\"X-Session-ID\")
    if not session_id:
        raise HTTPException(status_code=400, detail=\"X-Session-ID header required\")
    
    try:
        # Call Emergent auth API to get user data
        async with httpx.AsyncClient() as http_client:
            auth_response = await http_client.get(
                \"https://demobackend.emergentagent.com/auth/v1/env/oauth/session-data\",
                headers={\"X-Session-ID\": session_id},
                timeout=10.0
            )
            
            if auth_response.status_code != 200:
                raise HTTPException(status_code=401, detail=\"Invalid session_id\")
            
            auth_data = auth_response.json()
        
        # Extract user data
        user_email = auth_data.get(\"email\")
        user_name = auth_data.get(\"name\")
        user_picture = auth_data.get(\"picture\")
        session_token = auth_data.get(\"session_token\")
        
        if not user_email or not session_token:
            raise HTTPException(status_code=400, detail=\"Invalid auth response\")
        
        # Check if user exists
        existing_user = await db.users.find_one(
            {\"email\": user_email},
            {\"_id\": 0}
        )
        
        if existing_user:
            user_id = existing_user[\"user_id\"]
            # Update user data if needed
            await db.users.update_one(
                {\"user_id\": user_id},
                {\"$set\": {
                    \"name\": user_name,
                    \"picture\": user_picture
                }}
            )
        else:
            # Create new user
            user_id = f\"user_{uuid.uuid4().hex[:12]}\"
            user_doc = {
                \"user_id\": user_id,
                \"email\": user_email,
                \"name\": user_name,
                \"picture\": user_picture,
                \"created_at\": datetime.now(timezone.utc).isoformat()
            }
            await db.users.insert_one(user_doc)
        
        # Create session
        expires_at = datetime.now(timezone.utc) + timedelta(days=7)
        session_doc = {
            \"user_id\": user_id,
            \"session_token\": session_token,
            \"expires_at\": expires_at.isoformat(),
            \"created_at\": datetime.now(timezone.utc).isoformat()
        }
        
        # Delete old sessions for this user
        await db.user_sessions.delete_many({\"user_id\": user_id})
        
        # Insert new session
        await db.user_sessions.insert_one(session_doc)
        
        # Set httpOnly cookie
        response.set_cookie(
            key=\"session_token\",
            value=session_token,
            httponly=True,
            secure=True,
            samesite=\"none\",
            max_age=7*24*60*60,  # 7 days
            path=\"/\"
        )
        
        # Get user data
        user_doc = await db.users.find_one({\"user_id\": user_id}, {\"_id\": 0})
        if isinstance(user_doc.get('created_at'), str):
            user_doc['created_at'] = datetime.fromisoformat(user_doc['created_at'])
        
        return User(**user_doc)
        
    except httpx.RequestError as e:
        logging.error(f\"Error calling auth API: {str(e)}\")
        raise HTTPException(status_code=500, detail=\"Authentication service unavailable\")
    except Exception as e:
        logging.error(f\"Error creating session: {str(e)}\")
        raise HTTPException(status_code=500, detail=f\"Failed to create session: {str(e)}\")


@api_router.get(\"/auth/me\")
async def get_me(request: Request):
    \"\"\"Get current user data from session\"\"\"
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail=\"Not authenticated\")
    return user


@api_router.post(\"/auth/logout\")
async def logout(request: Request, response: Response):
    \"\"\"Logout and clear session\"\"\"
    session_token = get_session_token(request)
    if session_token:
        # Delete session from database
        await db.user_sessions.delete_one({\"session_token\": session_token})
    
    # Clear cookie
    response.delete_cookie(
        key=\"session_token\",
        path=\"/\",
        secure=True,
        samesite=\"none\"
    )
    
    return {\"message\": \"Logged out successfully\"}


# NEW: Text Extraction Routes
@api_router.post(\"/extract-text\", response_model=TextExtractionResponse)
async def extract_text(request: Request, file: UploadFile = File(...)):
    \"\"\"Extract text from uploaded image or PDF\"\"\"
    try:
        contents = await file.read()
        
        extracted_text = \"\"
        
        # Check file type and extract accordingly
        if file.content_type in ['image/jpeg', 'image/png', 'image/webp']:
            extracted_text = extract_text_from_image(contents)
        elif file.content_type == 'application/pdf':
            extracted_text = extract_text_from_pdf_pages(contents)
        else:
            raise HTTPException(status_code=400, detail=\"Unsupported file type\")
        
        if not extracted_text:
            raise HTTPException(status_code=400, detail=\"No text could be extracted from the file\")
        
        analysis_id = str(uuid.uuid4())
        
        # Store extracted text in database
        current_user = await get_current_user(request)
        
        text_doc = {
            \"analysis_id\": analysis_id,
            \"user_id\": current_user.user_id if current_user else None,
            \"filename\": file.filename,
            \"file_type\": \"text_extraction\",
            \"extracted_text\": extracted_text,
            \"word_count\": len(extracted_text.split()),
            \"char_count\": len(extracted_text),
            \"timestamp\": datetime.now(timezone.utc).isoformat()
        }
        await db.text_extractions.insert_one(text_doc)
        
        return TextExtractionResponse(
            text=extracted_text,
            analysis_id=analysis_id,
            word_count=len(extracted_text.split()),
            char_count=len(extracted_text)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f\"Text extraction error: {str(e)}\")
        raise HTTPException(status_code=500, detail=f\"Failed to extract text: {str(e)}\")


@api_router.post(\"/summarize/{analysis_id}\", response_model=SummaryResponse)
async def summarize_extracted_text(analysis_id: str, num_sentences: int = Query(default=5, ge=1, le=10)):
    \"\"\"Summarize previously extracted text\"\"\"
    try:
        # Get extracted text from database
        text_doc = await db.text_extractions.find_one(
            {\"analysis_id\": analysis_id},
            {\"_id\": 0}
        )
        
        if not text_doc:
            raise HTTPException(status_code=404, detail=\"Text extraction not found\")
        
        text = text_doc['extracted_text']
        original_sentences = len(sent_tokenize(text))
        
        summary = summarize_text(text, num_sentences)
        summary_sentence_count = len(sent_tokenize(summary))
        
        # Store summary
        summary_doc = {
            \"analysis_id\": analysis_id,
            \"summary\": summary,
            \"original_sentences\": original_sentences,
            \"summary_sentences\": summary_sentence_count,
            \"timestamp\": datetime.now(timezone.utc).isoformat()
        }
        await db.summaries.update_one(
            {\"analysis_id\": analysis_id},
            {\"$set\": summary_doc},
            upsert=True
        )
        
        return SummaryResponse(
            summary=summary,
            original_sentences=original_sentences,
            summary_sentences=summary_sentence_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f\"Summarization error: {str(e)}\")
        raise HTTPException(status_code=500, detail=f\"Failed to summarize: {str(e)}\")


@api_router.post(\"/important-points/{analysis_id}\", response_model=ImportantPointsResponse)
async def get_important_points(analysis_id: str, num_points: int = Query(default=5, ge=1, le=10)):
    \"\"\"Extract important points from previously extracted text\"\"\"
    try:
        # Get extracted text from database
        text_doc = await db.text_extractions.find_one(
            {\"analysis_id\": analysis_id},
            {\"_id\": 0}
        )
        
        if not text_doc:
            raise HTTPException(status_code=404, detail=\"Text extraction not found\")
        
        text = text_doc['extracted_text']
        points = extract_important_points(text, num_points)
        
        # Store important points
        points_doc = {
            \"analysis_id\": analysis_id,
            \"points\": points,
            \"total_points\": len(points),
            \"timestamp\": datetime.now(timezone.utc).isoformat()
        }
        await db.important_points.update_one(
            {\"analysis_id\": analysis_id},
            {\"$set\": points_doc},
            upsert=True
        )
        
        return ImportantPointsResponse(
            points=points,
            total_points=len(points)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f\"Important points extraction error: {str(e)}\")
        raise HTTPException(status_code=500, detail=f\"Failed to extract important points: {str(e)}\")


@api_router.post(\"/answer-question/{analysis_id}\", response_model=QAResponse)
async def answer_question_from_text(analysis_id: str, question: str = Query(..., min_length=3)):
    \"\"\"Answer a question based on previously extracted text\"\"\"
    try:
        # Get extracted text from database
        text_doc = await db.text_extractions.find_one(
            {\"analysis_id\": analysis_id},
            {\"_id\": 0}
        )
        
        if not text_doc:
            raise HTTPException(status_code=404, detail=\"Text extraction not found\")
        
        text = text_doc['extracted_text']
        answer, confidence = answer_question(text, question)
        
        # Store Q&A
        qa_doc = {
            \"analysis_id\": analysis_id,
            \"question\": question,
            \"answer\": answer,
            \"confidence\": confidence,
            \"timestamp\": datetime.now(timezone.utc).isoformat()
        }
        await db.qa_history.insert_one(qa_doc)
        
        return QAResponse(
            question=question,
            answer=answer,
            confidence=confidence
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f\"Question answering error: {str(e)}\")
        raise HTTPException(status_code=500, detail=f\"Failed to answer question: {str(e)}\")


# Original Analysis Routes (kept intact)
@api_router.post(\"/analyze-image\", response_model=ImageAnalysisResponse)
async def analyze_image(request: Request, file: UploadFile = File(...)):
    \"\"\"Analyze an uploaded image using OpenAI vision API\"\"\"
    try:
        # Read the uploaded file
        contents = await file.read()
        
        # Validate file type
        allowed_types = ['image/jpeg', 'image/png', 'image/webp']
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f\"Invalid file type. Only JPEG, PNG, and WEBP images are supported.\"
            )
        
        # Convert to base64
        base64_image = base64.b64encode(contents).decode('utf-8')
        
        # Get API key from environment
        api_key = os.environ.get('EMERGENT_LLM_KEY')
        if not api_key:
            raise HTTPException(status_code=500, detail=\"API key not configured\")
        
        # Create LLM chat instance
        chat = LlmChat(
            api_key=api_key,
            session_id=f\"analysis-{uuid.uuid4()}\",
            system_message=\"You are an expert AI image analyst. Provide detailed, insightful, and accurate analysis of images. Focus on describing what you see, identifying key elements, colors, composition, mood, and any notable details. Be professional yet engaging.\"
        )
        
        # Configure to use OpenAI GPT-5.2
        chat.with_model(\"openai\", \"gpt-5.2\")
        
        # Create image content
        image_content = ImageContent(image_base64=base64_image)
        
        # Create user message with image
        user_message = UserMessage(
            text=\"Please analyze this image in detail. Describe what you see, identify key elements, colors, composition, mood, and provide meaningful insights about the image.\",
            file_contents=[image_content]
        )
        
        # Send message and get response
        analysis_result = await chat.send_message(user_message)
        
        # Generate analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Check if user is logged in
        current_user = await get_current_user(request)
        
        # Store analysis in database
        analysis_doc = {
            \"analysis_id\": analysis_id,
            \"user_id\": current_user.user_id if current_user else None,
            \"filename\": file.filename,
            \"file_type\": \"image\",
            \"content_type\": file.content_type,
            \"analysis\": analysis_result,
            \"timestamp\": datetime.now(timezone.utc).isoformat()
        }
        await db.analyses.insert_one(analysis_doc)
        
        return ImageAnalysisResponse(
            analysis=analysis_result,
            analysis_id=analysis_id,
            timestamp=datetime.now(timezone.utc)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f\"Error analyzing image: {str(e)}\")
        raise HTTPException(status_code=500, detail=f\"Failed to analyze image: {str(e)}\")


@api_router.post(\"/analyze-pdf\", response_model=ImageAnalysisResponse)
async def analyze_pdf(request: Request, file: UploadFile = File(...), mode: str = \"text\"):
    \"\"\"Analyze an uploaded PDF - either text extraction or image analysis\"\"\"
    try:
        # Read the uploaded file
        contents = await file.read()
        
        # Validate file type
        if file.content_type != 'application/pdf':
            raise HTTPException(
                status_code=400, 
                detail=\"Invalid file type. Only PDF files are supported.\"
            )
        
        # Get API key from environment
        api_key = os.environ.get('EMERGENT_LLM_KEY')
        if not api_key:
            raise HTTPException(status_code=500, detail=\"API key not configured\")
        
        analysis_result = \"\"
        
        if mode == \"text\":
            # Extract text from PDF
            pdf_reader = PdfReader(BytesIO(contents))
            extracted_text = \"\"
            for page in pdf_reader.pages:
                extracted_text += page.extract_text() + \"
\"
            
            if not extracted_text.strip():
                raise HTTPException(status_code=400, detail=\"No text found in PDF. Try image mode.\")
            
            # Analyze text using LLM
            chat = LlmChat(
                api_key=api_key,
                session_id=f\"pdf-text-analysis-{uuid.uuid4()}\",
                system_message=\"You are an expert document analyst. Analyze the provided text and give comprehensive insights about its content, structure, key points, and overall message.\"
            )
            chat.with_model(\"openai\", \"gpt-5.2\")
            
            user_message = UserMessage(
                text=f\"Please analyze this document text and provide detailed insights:

{extracted_text[:4000]}\"  # Limit to avoid token limits
            )
            
            analysis_result = await chat.send_message(user_message)
            
        else:  # image mode
            # Convert PDF pages to images and analyze
            images = convert_from_bytes(contents, dpi=150, fmt='jpeg')
            
            # Analyze first page as image (can be extended to all pages)
            if not images:
                raise HTTPException(status_code=400, detail=\"Could not convert PDF to images\")
            
            # Convert first page to base64
            img_buffer = BytesIO()
            images[0].save(img_buffer, format='JPEG')
            img_buffer.seek(0)
            base64_image = base64.b64encode(img_buffer.read()).decode('utf-8')
            
            # Analyze using vision API
            chat = LlmChat(
                api_key=api_key,
                session_id=f\"pdf-image-analysis-{uuid.uuid4()}\",
                system_message=\"You are an expert document analyst. Analyze the visual content of this PDF page and provide detailed insights about its layout, design, content, and key information.\"
            )
            chat.with_model(\"openai\", \"gpt-5.2\")
            
            image_content = ImageContent(image_base64=base64_image)
            user_message = UserMessage(
                text=f\"Please analyze this PDF page (page 1 of {len(images)}). Describe the layout, content, key information, and visual elements.\",
                file_contents=[image_content]
            )
            
            analysis_result = await chat.send_message(user_message)
        
        # Generate analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Check if user is logged in
        current_user = await get_current_user(request)
        
        # Store analysis in database
        analysis_doc = {
            \"analysis_id\": analysis_id,
            \"user_id\": current_user.user_id if current_user else None,
            \"filename\": file.filename,
            \"file_type\": \"pdf\",
            \"content_type\": file.content_type,
            \"analysis\": analysis_result,
            \"timestamp\": datetime.now(timezone.utc).isoformat()
        }
        await db.analyses.insert_one(analysis_doc)
        
        return ImageAnalysisResponse(
            analysis=analysis_result,
            analysis_id=analysis_id,
            timestamp=datetime.now(timezone.utc)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f\"Error analyzing PDF: {str(e)}\")
        raise HTTPException(status_code=500, detail=f\"Failed to analyze PDF: {str(e)}\")


@api_router.get(\"/download-text/{analysis_id}\")
async def download_text(analysis_id: str):
    \"\"\"Download analysis as text file\"\"\"
    try:
        # Get analysis from database
        analysis_doc = await db.analyses.find_one(
            {\"analysis_id\": analysis_id},
            {\"_id\": 0}
        )
        
        if not analysis_doc:
            raise HTTPException(status_code=404, detail=\"Analysis not found\")
        
        # Create text file
        text_content = f\"\"\"LifeLens AI Analysis Report
{'='*50}

Filename: {analysis_doc['filename']}
File Type: {analysis_doc['file_type']}
Analysis Date: {analysis_doc['timestamp']}

{'='*50}

ANALYSIS:

{analysis_doc['analysis']}

{'='*50}
Generated by LifeLens AI
\"\"\"
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
            tmp.write(text_content)
            tmp_path = tmp.name
        
        return FileResponse(
            tmp_path,
            media_type='text/plain',
            filename=f\"analysis_{analysis_id}.txt\",
            headers={\"Content-Disposition\": f\"attachment; filename=analysis_{analysis_id}.txt\"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f\"Error downloading text: {str(e)}\")
        raise HTTPException(status_code=500, detail=f\"Failed to download text: {str(e)}\")


@api_router.get(\"/download-pdf/{analysis_id}\")
async def download_pdf(analysis_id: str):
    \"\"\"Download analysis as PDF file\"\"\"
    try:
        # Get analysis from database
        analysis_doc = await db.analyses.find_one(
            {\"analysis_id\": analysis_id},
            {\"_id\": 0}
        )
        
        if not analysis_doc:
            raise HTTPException(status_code=404, detail=\"Analysis not found\")
        
        # Create PDF
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp_path = tmp.name
        
        doc = SimpleDocTemplate(tmp_path, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor='#6D28D9',
            spaceAfter=30,
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor='#5B21B6',
            spaceAfter=12,
        )
        
        # Add content
        story.append(Paragraph(\"LifeLens AI Analysis Report\", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        story.append(Paragraph(f\"<b>Filename:</b> {analysis_doc['filename']}\", styles['Normal']))
        story.append(Paragraph(f\"<b>File Type:</b> {analysis_doc['file_type'].upper()}\", styles['Normal']))
        story.append(Paragraph(f\"<b>Analysis Date:</b> {analysis_doc['timestamp']}\", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        story.append(Paragraph(\"Analysis Results\", heading_style))
        
        # Split analysis into paragraphs
        for paragraph in analysis_doc['analysis'].split('
'):
            if paragraph.strip():
                story.append(Paragraph(paragraph, styles['Normal']))
                story.append(Spacer(1, 0.1*inch))
        
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph(\"<i>Generated by LifeLens AI</i>\", styles['Normal']))
        
        # Build PDF
        doc.build(story)
        
        return FileResponse(
            tmp_path,
            media_type='application/pdf',
            filename=f\"analysis_{analysis_id}.pdf\",
            headers={\"Content-Disposition\": f\"attachment; filename=analysis_{analysis_id}.pdf\"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f\"Error downloading PDF: {str(e)}\")
        raise HTTPException(status_code=500, detail=f\"Failed to download PDF: {str(e)}\")


@api_router.get(\"/history\", response_model=List[AnalysisHistory])
async def get_history(request: Request):
    \"\"\"Get analysis history for logged-in user\"\"\"
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail=\"Authentication required\")
    
    # Get user's analyses
    analyses = await db.analyses.find(
        {\"user_id\": user.user_id},
        {\"_id\": 0}
    ).sort(\"timestamp\", -1).to_list(100)
    
    # Convert datetime strings
    for analysis in analyses:
        if isinstance(analysis.get('timestamp'), str):
            analysis['timestamp'] = datetime.fromisoformat(analysis['timestamp'])
    
    return [AnalysisHistory(**a) for a in analyses]


# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=[\"*\"],
    allow_headers=[\"*\"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event(\"shutdown\")
async def shutdown_db_client():
    client.close()
"
