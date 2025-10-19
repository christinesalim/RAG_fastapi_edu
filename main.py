"""
FastAPI application for RAG (Retrieval-Augmented Generation) system.
Handles file uploads and question answering using OpenAI.
"""

import os
from fastapi import FastAPI, UploadFile, HTTPException, Depends, BackgroundTasks
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
import shutil
import io
from db import get_db, File, FileChunk, SessionLocal
from sqlalchemy.orm import Session
from file_parser import FileParser
from background_tasks import TextProcessor, client
from sqlalchemy import select
import openai


# Load environment variables from .env file
load_dotenv()

app = FastAPI()

client = OpenAI()

# Get OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

#Pydantic models
class QuestionModel(BaseModel):
    """Request model for question answering endpoint."""
    question: str
    
class AskModel(BaseModel):
    """Request model for asking a question using a specific file as a reference"""
    document_id: int
    question: str

@app.get("/")
async def root(db: Session = Depends(get_db)):
    """List all uploaded files."""
    if db is None:
        return {"message": "Database not available. Please configure PostgreSQL connection."}

    # Query the database for all files
    files_query = select(File)
    files = db.scalars(files_query).all()

    # Format and return the list of files
    files_list = [{"file_id": file.file_id, "file_name": file.file_name} for file in files]
    return files_list


@app.post("/uploadfile/")
async def upload_file(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Upload a file to the sources directory for processing."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    allowed_extensions = ['txt', 'pdf']
    
    #Check if the file extension is allowed
    file_extension = file.filename.split('.')[-1]
    if file_extension not in allowed_extensions:
        raise HTTPException (status_code=400, detail="File type not allowed")

    try: 
        # Make sure the directory exists
        folder = "sources"
        os.makedirs(folder, exist_ok=True)
        
        # Secure way to save the file
        file_location = os.path.join(folder, file.filename)
        
        # Read the content asynch
        file_content = await file.read()
        
        with open(file_location, "wb") as file_object:
            # Convert bytes to file like object
            file_like_obj = io.BytesIO(file_content)
            
            # Use shutil.copyfileobj for secure file writing
            shutil.copyfileobj(file_like_obj, file_object)

        content_parser = FileParser(file_location)
        file_text_content = content_parser.parse()
        
        # Save the file details in the database
        new_file = File(file_name=file.filename,
                        file_content=file_text_content)
        db.add(new_file)
        db.commit()
        db.refresh(new_file)
        
        # Add background job for processing the file
        background_tasks.add_task(process_file_in_background,
                                  new_file.file_id,
                                  file_text_content)
        return {"info": "File saved", "file_name": file.filename}
        
    except Exception as e:
        print(f"Error saving file: {e}")
        raise HTTPException(status_code=500, detail="Error saving file") from e

# Function to get similar chunks
async def get_similar_chunks(file_id: int, question: str, db: Session):
    """Find the most relevant chunks from a file similar to the user's question.

    Uses vector similarity search with OpenAI embeddings and pgvector's L2 distance
    to find the top 10 most semantically similar chunks.

    Args:
        file_id (int): ID of the file to search chunks from
        question (str): User's question to find similar content for
        db (Session): SQLAlchemy database session

    Raises:
        HTTPException: 500 error if embedding creation or database query fails

    Returns:
        list[FileChunk]: List of up to 10 FileChunk objects ordered by similarity
    """
    try: 
        # Create embeddings for the question by converting the question into a vector
        response = client.embeddings.create(input=question, model="text-embedding-ada-002")
        
        # Extract the question embedding from the API response
        question_embedding = response.data[0].embedding
        
        # Filter chunks and select top 10 chunks with smaller Euclidean distance (more similar)
        similar_chunks_query = select(FileChunk).where(FileChunk.file_id == file_id)\
            .order_by(FileChunk.embedding_vector.l2_distance(question_embedding)).limit(10)
        similar_chunks = db.scalars(similar_chunks_query).all()
        
        return similar_chunks
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    
    
@app.post("/find-similar-chunks/{file_id}")
async def find_similar_chunks_endpoint(file_id: int, question_data: QuestionModel, db: Session = Depends(get_db)):
    """Find semantically similar chunks from a file."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        similar_chunks = await get_similar_chunks(file_id, question_data.question, db)

        formatted_response = [
            {"chunk_id": chunk.chunk_id, "chunk_text": chunk.chunk_text}
            for chunk in similar_chunks
        ]
        return formatted_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def process_file_in_background(file_id: int, text: str):
    """Background task to chunk and embed file content."""
    db = SessionLocal()
    try:
        processor = TextProcessor(db, file_id)
        processor.chunk_and_embed(text)
    finally:
        db.close()  
  

@app.post("/ask/")
async def ask_question(request: AskModel, db: Session = Depends(get_db)):
    """Answer a question using OpenAI's GPT model with RAG."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        similar_chunks = await get_similar_chunks(request.document_id, request.question, db)
        
        #Construct context from similar chunks
        context_texts = [chunk.chunk_text for chunk in similar_chunks]
        context = " ".join(context_texts)
        
        # Update the system with the context
        system_message = f"You are a helpful assistant. Here is the context to use to reply to questions: {context}"
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": request.question}
            ]
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        return { "error": str(e) }
    
    