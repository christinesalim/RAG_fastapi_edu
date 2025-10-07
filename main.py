"""
FastAPI application for RAG (Retrieval-Augmented Generation) system.
Handles file uploads and question answering using OpenAI.
"""

import os
from fastapi import FastAPI, UploadFile, HTTPException
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
import shutil
import io


# Load environment variables from .env file
load_dotenv()

app = FastAPI()

client = OpenAI()

# Get OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

class Question(BaseModel):
    """Request model for question answering endpoint."""
    question: str

@app.get("/")
def read_root():
    """Health check endpoint to verify service status."""
    return {"status": "Service is running", "message": "FastAPI server is healthy"}

@app.post("/uploadfile/")
async def upload_file(file: UploadFile):
    """Upload a file to the sources directory for processing."""
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

        # Sanitize filename to prevent path traversal
        #safe_filename = os.path.basename(file.filename)
        #file_location = os.path.join(folder, safe_filename)
        
        # Secure way to save the file
        file_location = os.path.join(folder, file.filename)
        
        # Read the content asynch
        file_content = await file.read()
        
        with open(file_location, "wb") as file_object:
            # Convert bytes to file like object
            file_like_obj = io.BytesIO(file_content)
            
            # Use shutil.copyfileobj for secure file writing
            shutil.copyfileobj(file_like_obj, file_object)

            return {"info": "File saved", "filename": file.filename}
  
    except Exception as e:
        print(f"Error saving fiel: {e}")
        raise HTTPException(status_code=500, detail="Error saving file") from e
        
  
@app.post("/ask/")
async def ask_question(question: Question):
    """Answer a question using OpenAI's GPT model."""
    if not question.question:
        return {"error": "No question provided"}

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question.question}
            ]
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        return { "error": str(e) }
    
    