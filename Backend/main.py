import os
import uuid
import shutil
from datetime import datetime, timedelta
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.pool import QueuePool
import fitz  # PyMuPDF
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from transformers import pipeline
import mimetypes
from functools import lru_cache
import gc
import threading
from typing import Dict, List, Optional
import hashlib
from concurrent.futures import ThreadPoolExecutor
import torch
from passlib.context import CryptContext
from jose import JWTError, jwt

# === Configuration ===
UPLOAD_DIR = "uploads"
INDEX_DIR = "faiss_indexes"
TEMP_DIR = "temp"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_CONVERSATION_HISTORY = 5
CACHE_TTL = 3600  # 1 hour cache TTL
MAX_CACHE_SIZE = 1000  # Maximum number of cached items
MAX_MEMORY_USAGE = 0.8  # Maximum memory usage threshold (80%)
DB_FILE = "test.db"

# === Authentication Configuration ===
SECRET_KEY = "your-secret-key-here"  # Change this to a secure secret key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# === Database Setup with Connection Pooling ===
SQLALCHEMY_DATABASE_URL = f"sqlite:///./{DB_FILE}"

# Create engine
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,  # Recycle connections after 30 minutes
    connect_args={"check_same_thread": False}
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

# === Cache Implementation ===
class Cache:
    def __init__(self, ttl: int = CACHE_TTL, max_size: int = MAX_CACHE_SIZE):
        self.cache: Dict[str, tuple] = {}
        self.ttl = ttl
        self.max_size = max_size
        self.lock = threading.Lock()

    def get(self, key: str) -> Optional[any]:
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                if datetime.now().timestamp() - timestamp < self.ttl:
                    return value
                del self.cache[key]
        return None

    def set(self, key: str, value: any):
        with self.lock:
            if len(self.cache) >= self.max_size:
                # Remove oldest entries
                sorted_items = sorted(self.cache.items(), key=lambda x: x[1][1])
                for key_to_remove, _ in sorted_items[:len(sorted_items)//4]:
                    del self.cache[key_to_remove]
            self.cache[key] = (value, datetime.now().timestamp())

    def clear(self):
        with self.lock:
            self.cache.clear()

# Initialize caches
answer_cache = Cache()
index_cache = Cache()

# === Memory Management ===
def check_memory_usage():
    """Check if memory usage is above threshold and trigger garbage collection if needed."""
    import psutil
    process = psutil.Process()
    memory_percent = process.memory_percent()
    
    if memory_percent > MAX_MEMORY_USAGE * 100:
        gc.collect()
        # Clear some caches if memory is still high
        if process.memory_percent() > MAX_MEMORY_USAGE * 100:
            answer_cache.clear()
            gc.collect()

# === Models ===
class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class Document(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True, index=True)
    filename = Column(String)
    upload_date = Column(DateTime)
    is_processed = Column(Integer, default=0)  # 0 = not processed, 1 = processed
    processing_status = Column(String, default="pending")  # pending, processing, completed, failed
    processing_error = Column(String, nullable=True)
    last_updated = Column(DateTime, default=datetime.utcnow)
    conversations = relationship("Conversation", back_populates="document")

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(String, primary_key=True, index=True)
    document_id = Column(String, ForeignKey("documents.id"))
    question = Column(Text)
    answer = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    document = relationship("Document", back_populates="conversations")

# Create all tables
Base.metadata.create_all(bind=engine)

# === App Setup ===
app = FastAPI()

# Add CORS middleware with more permissive settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# === Global Stores ===
index_store = {}
qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
    device=0 if torch.cuda.is_available() else -1
)
processing_documents = set()  # Track documents currently being processed

# Define the embedding model name as a constant
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# === Schemas ===
class QuestionRequest(BaseModel):
    document_id: str
    question: str
    conversation_id: str | None = None  # Optional conversation ID for follow-up questions

class QuestionResponse(BaseModel):
    answer: str
    conversation_id: str

# === Authentication Models ===
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str

class UserResponse(BaseModel):
    id: str
    email: str
    username: str
    created_at: datetime

# === Authentication Functions ===
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == token_data.username).first()
        if user is None:
            raise credentials_exception
        return user
    finally:
        db.close()

# === Authentication Endpoints ===
@app.post("/register", response_model=UserResponse)
async def register(user_data: UserCreate):
    db = SessionLocal()
    try:
        # Check if user already exists
        if db.query(User).filter(User.email == user_data.email).first():
            raise HTTPException(status_code=400, detail="Email already registered")
        if db.query(User).filter(User.username == user_data.username).first():
            raise HTTPException(status_code=400, detail="Username already taken")
        
        # Create new user
        user = User(
            id=str(uuid.uuid4()),
            email=user_data.email,
            username=user_data.username,
            hashed_password=get_password_hash(user_data.password)
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == form_data.username).first()
        if not user or not verify_password(form_data.password, user.hashed_password):
            raise HTTPException(
                status_code=401,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}
    finally:
        db.close()

# === Batch Processing ===
class BatchProcessor:
    def __init__(self, batch_size: int = 5):
        self.batch_size = batch_size
        self.processing_queue: List[str] = []
        self.lock = threading.Lock()

    def add_to_queue(self, document_id: str):
        with self.lock:
            if document_id not in self.processing_queue:
                self.processing_queue.append(document_id)
                if len(self.processing_queue) >= self.batch_size:
                    self.process_batch()

    def process_batch(self):
        with self.lock:
            batch = self.processing_queue[:self.batch_size]
            self.processing_queue = self.processing_queue[self.batch_size:]
            
        for doc_id in batch:
            process_document(doc_id)

batch_processor = BatchProcessor()

# === Optimized Question Answering ===
def get_cache_key(document_id: str, question: str, conversation_id: Optional[str] = None) -> str:
    """Generate a cache key for a question."""
    key = f"{document_id}:{question}"
    if conversation_id:
        key = f"{key}:{conversation_id}"
    return hashlib.md5(key.encode()).hexdigest()

# Initialize embeddings with error handling
def get_embeddings():
    try:
        # Try to use GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Using device: {device}")
        
        print(f"[INFO] Loading embedding model: {EMBEDDING_MODEL}")
        
        try:
            # First try to load the model with default settings
            embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={'device': device}
            )
            print("[INFO] Successfully loaded embeddings model with default settings")
            return embeddings
        except Exception as e:
            print(f"[WARN] Failed to load model with default settings: {str(e)}")
            print("[INFO] Trying with normalized embeddings...")
            
            # If that fails, try with normalized embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )
            print("[INFO] Successfully loaded embeddings model with normalized embeddings")
            return embeddings
            
    except Exception as e:
        error_msg = f"Failed to load embedding model: {str(e)}"
        print(f"[ERROR] {error_msg}")
        import traceback
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        raise Exception(error_msg)

@app.post("/ask")
async def ask_question(
    req: QuestionRequest,
    current_user: User = Depends(get_current_user)
):
    print(f"[INFO] Processing question for document {req.document_id}: {req.question}")
    
    # Check memory usage
    check_memory_usage()
    
    # Check cache first
    cache_key = get_cache_key(req.document_id, req.question, req.conversation_id)
    cached_answer = answer_cache.get(cache_key)
    if cached_answer:
        print(f"[INFO] Returning cached answer for document {req.document_id}")
        return cached_answer

    db = SessionLocal()
    try:
        # Check if document exists and is processed
        doc = db.query(Document).filter(Document.id == req.document_id).first()
        if not doc:
            print(f"[ERROR] Document not found: {req.document_id}")
            raise HTTPException(status_code=404, detail="Document not found")
        
        if not doc.is_processed:
            print(f"[ERROR] Document not processed: {req.document_id}, status: {doc.processing_status}")
            raise HTTPException(
                status_code=400, 
                detail=f"Document is not processed yet. Current status: {doc.processing_status}"
            )
        
        if doc.processing_status == "failed":
            print(f"[ERROR] Document processing failed: {req.document_id}, error: {doc.processing_error}")
            raise HTTPException(
                status_code=400,
                detail=f"Document processing failed: {doc.processing_error}"
            )

        print(f"[INFO] Getting conversation context for document {req.document_id}")
        # Get conversation context
        conversation_context = get_conversation_context(db, req.document_id, req.conversation_id)

        # Load or get cached index
        print(f"[INFO] Loading index for document {req.document_id}")
        index = index_cache.get(req.document_id)
        if not index:
            index_path = os.path.join(INDEX_DIR, req.document_id)
            if not os.path.exists(index_path):
                print(f"[ERROR] Index not found at path: {index_path}")
                raise HTTPException(status_code=404, detail="Document index not found")
            
            print(f"[INFO] Loading embeddings model")
            try:
                embeddings = get_embeddings()
                
                print(f"[INFO] Loading FAISS index from {index_path}")
                try:
                    # Check if index files exist
                    index_files = os.listdir(index_path)
                    print(f"[INFO] Found index files: {index_files}")
                    
                    # Check for required FAISS files
                    if not any(f.endswith('.faiss') for f in index_files):
                        raise ValueError("No .faiss file found in the index directory")
                    if not any(f.endswith('.pkl') for f in index_files):
                        raise ValueError("No .pkl file found in the index directory")
                    
                    # Load the index with explicit parameters
                    print("[INFO] Attempting to load FAISS index...")
                    try:
                        # First try loading without any extra parameters
                        index = FAISS.load_local(
                            index_path,
                            embeddings
                        )
                        print("[INFO] Successfully loaded FAISS index")
                    except Exception as e:
                        print(f"[ERROR] Failed to load FAISS index: {str(e)}")
                        # Try to get more information about the error
                        import traceback
                        print(f"[ERROR] Traceback: {traceback.format_exc()}")
                        raise
                    
                    # Validate the index
                    if not hasattr(index, 'similarity_search'):
                        raise ValueError("Invalid FAISS index: missing similarity_search method")
                    
                    # Test the index with a simple query
                    print("[INFO] Testing index with sample query")
                    try:
                        test_docs = index.similarity_search("test", k=1)
                        print("[INFO] Successfully performed test query")
                    except Exception as e:
                        print(f"[ERROR] Test query failed: {str(e)}")
                        raise
                    
                    if not isinstance(test_docs, list):
                        raise ValueError("Invalid FAISS index: similarity_search did not return a list")
                    
                    if len(test_docs) == 0:
                        raise ValueError("Invalid FAISS index: test query returned no results")
                    
                    print(f"[INFO] Successfully validated FAISS index")
                    index_cache.set(req.document_id, index)
                    
                except Exception as e:
                    error_msg = f"Failed to load or validate FAISS index: {str(e)}"
                    print(f"[ERROR] {error_msg}")
                    import traceback
                    print(f"[ERROR] Traceback: {traceback.format_exc()}")
                    raise HTTPException(status_code=500, detail=error_msg)
                    
            except Exception as e:
                error_msg = f"Failed to initialize embeddings: {str(e)}"
                print(f"[ERROR] {error_msg}")
                import traceback
                print(f"[ERROR] Traceback: {traceback.format_exc()}")
                raise HTTPException(status_code=500, detail=error_msg)

        # Get relevant context and generate answer
        print(f"[INFO] Performing similarity search for question")
        try:
            if not hasattr(index, 'similarity_search'):
                raise ValueError("Invalid index object: missing similarity_search method")
            
            # Ensure the question is not empty and is a string
            if not isinstance(req.question, str) or not req.question.strip():
                raise ValueError("Invalid question: must be a non-empty string")
            
            # Log the question and index type for debugging
            print(f"[INFO] Question type: {type(req.question)}, Index type: {type(index)}")
            print(f"[INFO] Question content: {req.question}")
            
            # Perform the search with error handling
            try:
                docs = index.similarity_search(req.question, k=3)
            except Exception as search_error:
                print(f"[ERROR] Raw similarity search error: {str(search_error)}")
                raise ValueError(f"Similarity search operation failed: {str(search_error)}")
            
            # Validate the search results
            if not isinstance(docs, list):
                raise ValueError(f"Invalid search results: expected list, got {type(docs)}")
            
            print(f"[INFO] Successfully performed similarity search, found {len(docs)} results")
            
        except Exception as e:
            error_msg = f"Similarity search failed: {str(e)}"
            print(f"[ERROR] {error_msg}")
            import traceback
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=error_msg)
            
        if not docs:
            print(f"[INFO] No relevant context found for question")
            return {"answer": "I couldn't find any relevant information to answer your question."}
            
        context = "\n".join([doc.page_content for doc in docs])
        print(f"[INFO] Found {len(docs)} relevant chunks")
        
        # Add conversation context if available
        if conversation_context:
            context = f"Previous conversation:\n{conversation_context}\n\nDocument context:\n{context}"
            print(f"[INFO] Added conversation context")
        
        print(f"[INFO] Generating answer using QA pipeline")
        try:
            answer = qa_pipeline(question=req.question, context=context)
            print(f"[INFO] Successfully generated answer")
        except Exception as e:
            print(f"[ERROR] QA pipeline failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")
        
        # Store the conversation
        conversation_id = str(uuid.uuid4())
        print(f"[INFO] Storing conversation with ID: {conversation_id}")
        try:
            conversation = Conversation(
                id=conversation_id,
                document_id=req.document_id,
                question=req.question,
                answer=answer['answer']
            )
            db.add(conversation)
            db.commit()
            print(f"[INFO] Successfully stored conversation")
        except Exception as e:
            print(f"[ERROR] Failed to store conversation: {str(e)}")
            # Don't raise here, we still want to return the answer
        
        response = {
            "answer": answer['answer'],
            "conversation_id": conversation_id
        }
        
        # Cache the answer
        answer_cache.set(cache_key, response)
        print(f"[INFO] Successfully cached answer")
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error answering question: {str(e)}"
        print(f"[ERROR] {error_msg}")
        import traceback
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)
    finally:
        db.close()

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File size exceeds {MAX_FILE_SIZE/1024/1024}MB limit")

    try:
        # Generate a unique ID for the document
        document_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{document_id}.pdf")
        
        # Save the file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Create database entry
        db = SessionLocal()
        try:
            doc = Document(
                id=document_id,
                filename=file.filename,
                upload_date=datetime.utcnow(),
                is_processed=0,
                processing_status="processing",
                last_updated=datetime.utcnow()
            )
            db.add(doc)
            db.commit()
            print(f"[INFO] Created database entry for document: {document_id}")
        except Exception as e:
            print(f"[ERROR] Database error: {str(e)}")
            raise HTTPException(status_code=500, detail="Database error")
        finally:
            db.close()

        # Start processing in a separate thread
        print(f"[INFO] Starting processing for document: {document_id}")
        import threading
        thread = threading.Thread(target=process_document, args=(document_id,))
        thread.daemon = True  # Make thread daemon so it exits when main program exits
        thread.start()
        print(f"[INFO] Processing thread started for document: {document_id}")

        return {"document_id": document_id, "message": "File uploaded successfully"}
    except Exception as e:
        print(f"[ERROR] Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add cleanup endpoint
@app.post("/cleanup")
async def cleanup_resources():
    """Cleanup resources and clear caches."""
    answer_cache.clear()
    index_cache.clear()
    gc.collect()
    return {"status": "success", "message": "Resources cleaned up successfully"}

@app.get("/documents")
async def get_documents(current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        docs = db.query(Document).all()
        current_time = datetime.utcnow()
        
        # Update documents that have been processing for too long
        for doc in docs:
            if (doc.processing_status in ["processing", "extracting_text", "creating_index"] and 
                doc.id in processing_documents and 
                (current_time - doc.last_updated).total_seconds() > 300):  # 5 minutes timeout
                doc.processing_status = "failed"
                doc.processing_error = "Processing timeout"
                processing_documents.remove(doc.id)
                db.commit()

        # Only return documents that have been updated in the last 5 minutes
        # or are in a final state (completed/failed)
        recent_docs = [
            doc for doc in docs
            if (current_time - doc.last_updated).total_seconds() <= 300 or
               doc.processing_status in ["completed", "failed"]
        ]

        return [
            {
                "id": doc.id,
                "filename": doc.filename,
                "upload_date": doc.upload_date.isoformat() if doc.upload_date else None,
                "is_processed": bool(doc.is_processed),
                "processing_status": doc.processing_status,
                "processing_error": doc.processing_error,
                "last_updated": doc.last_updated.isoformat() if doc.last_updated else None
            }
            for doc in recent_docs
        ]
    except Exception as e:
        print(f"Error fetching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

# === PDF Processing ===
def validate_pdf(file_path: str) -> bool:
    """Validate if the file is a valid PDF using mimetypes."""
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type == 'application/pdf'

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF with optimized processing."""
    try:
        doc = fitz.open(file_path)
        text_chunks = []
        
        # Process pages in parallel using a thread pool
        def process_page(page):
            return page.get_text()
        
        with ThreadPoolExecutor() as executor:
            text_chunks = list(executor.map(process_page, doc))
        
        # Join all text chunks
        text = "\n".join(text_chunks)
        doc.close()
        return text
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")

def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks for better processing."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    return text_splitter.split_text(text)

def save_index_atomically(index: FAISS, document_id: str):
    """Save FAISS index atomically using temporary directory."""
    temp_index_path = os.path.join(TEMP_DIR, f"{document_id}_temp")
    final_index_path = os.path.join(INDEX_DIR, document_id)
    
    # Save to temporary location first
    index.save_local(temp_index_path)
    
    # Atomic rename
    if os.path.exists(final_index_path):
        shutil.rmtree(final_index_path)
    os.rename(temp_index_path, final_index_path)

def process_document(document_id: str):
    """Process a document in a separate thread."""
    if document_id in processing_documents:
        print(f"[WARN] Document {document_id} is already being processed")
        return

    processing_documents.add(document_id)
    print(f"[INFO] Processing started for document: {document_id}")
    
    db = SessionLocal()
    try:
        # Update status to processing
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            print(f"[ERROR] Document {document_id} not found in database")
            return
        
        doc.processing_status = "extracting_text"
        doc.last_updated = datetime.utcnow()
        db.commit()
        print(f"[INFO] Updated status to extracting_text for document: {document_id}")

        # Process the document
        file_path = os.path.join(UPLOAD_DIR, f"{document_id}.pdf")
        if not os.path.exists(file_path):
            print(f"[ERROR] PDF file not found at {file_path}")
            raise Exception("PDF file not found")

        # Extract text with progress tracking
        print(f"[INFO] Starting text extraction for document: {document_id}")
        text = extract_text_from_pdf(file_path)
        if not text.strip():
            print(f"[ERROR] No text extracted from document: {document_id}")
            raise Exception("No text could be extracted from the PDF")
        print(f"[INFO] Text extraction completed for document: {document_id}")
        
        # Update status to creating index
        doc.processing_status = "creating_index"
        doc.last_updated = datetime.utcnow()
        db.commit()
        print(f"[INFO] Updated status to creating_index for document: {document_id}")

        # Create and save the index with progress tracking
        print(f"[INFO] Starting index creation for document: {document_id}")
        try:
            embeddings = get_embeddings()
        except Exception as e:
            print(f"[ERROR] Failed to initialize embeddings: {str(e)}")
            raise Exception(f"Failed to initialize embeddings: {str(e)}")
        
        # Split text into smaller chunks for better processing
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)
        print(f"[INFO] Split text into {len(chunks)} chunks for document: {document_id}")
        
        # Create index in batches
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            try:
                if i == 0:
                    print("[INFO] Creating initial FAISS index...")
                    index = FAISS.from_texts(batch, embeddings)
                    print("[INFO] Successfully created initial FAISS index")
                else:
                    print(f"[INFO] Adding batch {i//batch_size + 1} to index...")
                    index.add_texts(batch)
                    print(f"[INFO] Successfully added batch {i//batch_size + 1}")
                print(f"[INFO] Processed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks for document: {document_id}")
            except Exception as e:
                print(f"[ERROR] Failed to process batch {i//batch_size + 1}: {str(e)}")
                raise Exception(f"Failed to create index: {str(e)}")
        
        # Save index
        index_path = os.path.join(INDEX_DIR, document_id)
        print(f"[INFO] Saving index to {index_path}")
        try:
            index.save_local(index_path)
            print("[INFO] Successfully saved index")
        except Exception as e:
            print(f"[ERROR] Failed to save index: {str(e)}")
            raise Exception(f"Failed to save index: {str(e)}")
            
        index_store[document_id] = index
        print(f"[INFO] Index saved for document: {document_id}")

        # Update status to completed
        doc.is_processed = 1
        doc.processing_status = "completed"
        doc.last_updated = datetime.utcnow()
        db.commit()
        print(f"[INFO] Successfully processed document: {document_id}")
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] Failed to process document {document_id}: {error_msg}")
        # Update status to failed
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.processing_status = "failed"
            doc.processing_error = error_msg
            doc.last_updated = datetime.utcnow()
            db.commit()
    finally:
        processing_documents.remove(document_id)
        db.close()
        print(f"[INFO] Processing completed for document: {document_id}")

def get_conversation_context(db: Session, document_id: str, conversation_id: str | None = None) -> str:
    """Get relevant conversation history for context."""
    if not conversation_id:
        return ""
    
    # Get the last N conversations for context
    conversations = db.query(Conversation)\
        .filter(Conversation.document_id == document_id)\
        .order_by(Conversation.timestamp.desc())\
        .limit(MAX_CONVERSATION_HISTORY)\
        .all()
    
    # Format conversation history
    context = []
    for conv in reversed(conversations):
        context.append(f"Previous Q: {conv.question}")
        context.append(f"Previous A: {conv.answer}")
    
    return "\n".join(context)

# Add new endpoint to get conversation history
@app.get("/conversations/{document_id}")
async def get_conversations(
    document_id: str,
    current_user: User = Depends(get_current_user)
):
    db = SessionLocal()
    try:
        conversations = db.query(Conversation)\
            .filter(Conversation.document_id == document_id)\
            .order_by(Conversation.timestamp.asc())\
            .all()
        
        return [
            {
                "id": conv.id,
                "question": conv.question,
                "answer": conv.answer,
                "timestamp": conv.timestamp.isoformat()
            }
            for conv in conversations
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_user)
):
    db = SessionLocal()
    try:
        # Get the document
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        # Delete the PDF file
        pdf_path = os.path.join(UPLOAD_DIR, f"{document_id}.pdf")
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

        # Delete the index directory
        index_path = os.path.join(INDEX_DIR, document_id)
        if os.path.exists(index_path):
            shutil.rmtree(index_path)

        # Delete conversations
        db.query(Conversation).filter(Conversation.document_id == document_id).delete()

        # Delete the document from database
        db.delete(doc)
        db.commit()

        return {"message": "Document deleted successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
