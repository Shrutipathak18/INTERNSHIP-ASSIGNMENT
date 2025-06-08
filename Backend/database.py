# database.py

from sqlalchemy import create_engine, Column, String, DateTime, Integer, ForeignKey, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
import datetime
import os
import hashlib
from fastapi import Depends

@app.post("/some-route")
def some_route(db: Session = Depends(get_db)):
# Get database URL from environment variable or use default
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./documents.db")

# Create SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}  # Only needed for SQLite
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class
Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    documents = relationship("Document", back_populates="owner")

class Document(Base):
    __tablename__ = "documents"

    document_id = Column(String, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)  # Original filename before processing
    file_hash = Column(String, unique=True, index=True)  # For deduplication
    file_size = Column(Integer)  # File size in bytes
    upload_date = Column(DateTime, default=func.now())
    file_path = Column(String, nullable=False)
    text_path = Column(String, nullable=False)
    is_processed = Column(Boolean, default=False)
    processing_error = Column(Text, nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("User", back_populates="documents")
    is_deleted = Column(Boolean, default=False)
    deleted_at = Column(DateTime, nullable=True)

# Create all tables
def init_db():
    Base.metadata.drop_all(bind=engine)  # Drop all tables
    Base.metadata.create_all(bind=engine)  # Create all tables

# Initialize the database
init_db()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Helper function to calculate file hash
def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()
