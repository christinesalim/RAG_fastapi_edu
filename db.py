import os
import urllib

from dotenv import load_dotenv
from pgvector.sqlalchemy import Vector
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import text
from sqlalchemy_utils import database_exists, create_database

# Get DB credentials and connection settings
load_dotenv()

# Database credentials
POSTGRES_USERNAME = os.getenv('POSTGRES_USERNAME')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
DATABASE_NAME = os.getenv('DATABASE_NAME', 'fastapi_rag_db')

# Flag to check if database is available
DATABASE_AVAILABLE = False
engine = None
SessionLocal = None

try:
    if POSTGRES_USERNAME and POSTGRES_PASSWORD:
        encoded_password = urllib.parse.quote_plus(POSTGRES_PASSWORD)

        # Create the engine for the specific database so we have a connection pool
        database_url = f"postgresql://{POSTGRES_USERNAME}:{encoded_password}@{POSTGRES_HOST}:{POSTGRES_PORT}/{DATABASE_NAME}"  # noqa
        engine = create_engine(database_url)

        # Check and create the datbase
        if not database_exists(engine.url):
            create_database(engine.url)

        # Session local factory
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        DATABASE_AVAILABLE = True
        print("✅ Database connection established")
    else:
        print("⚠️  Database credentials not found. Running without database.")
except Exception as e:
    print(f"⚠️  Could not connect to database: {e}")
    print("Running without database support.")

def get_db():
    """Get database session. Returns None if database is not available."""
    if not DATABASE_AVAILABLE or SessionLocal is None:
        print("⚠️  Database not available")
        return None
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Define models
Base = declarative_base()

class File(Base):
    __tablename__ = 'files'
    file_id = Column(Integer, primary_key=True)
    file_name = Column(String(255))
    file_content = Column(Text)
    
    
class FileChunk(Base):
      __tablename__ = 'file_chunks'
      chunk_id = Column(Integer, primary_key=True)
      file_id = Column(Integer, ForeignKey('files.file_id'))
      chunk_text = Column(Text)
      embedding_vector = Column(Vector(1536))
      
      
# Ensure the vector extension is enabled and create tables
if DATABASE_AVAILABLE and engine is not None:
    try:
        with engine.begin() as connection:
            connection.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))

        # Create tables
        Base.metadata.create_all(engine)
        print("✅ Database tables created/verified")
    except Exception as e:
        print(f"⚠️  Error creating tables: {e}")