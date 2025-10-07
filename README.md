# RAG FastAPI Project

## Architecture

![RAG Architecture Diagram](images/rag-architecture.png)

The system uses a Retrieval-Augmented Generation (RAG) pattern with:

- **Embedding Creator**: Converts user questions into embeddings
- **PostgreSQL + pgvector**: Stores and retrieves document chunks using vector similarity
- **LLM**: Generates answers by combining the question with relevant chunks

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the server:

```bash
uvicorn main:app --reload
```

## Endpoints

- `GET /` - Returns service status and health check
- `POST /uploadfile/` - Upload a file to the `sources/` directory

## Testing with curl

```bash
# Test root endpoint
curl http://localhost:8000/

# Test file upload endpoint
curl -X 'POST' 'http://localhost:8000/uploadfile/' -F 'file=@filename.txt'

# Test the open API ask endpoint
curl -X POST "http://127.0.0.1:8000/ask/" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the capital of France?"
  }'
```

## Environment Setup

1. Copy your OpenAI API key to the `.env` file:

```bash
# Edit .env file and replace with your actual API key
OPENAI_API_KEY=your_openai_api_key_here
```

**Important:** Never commit your actual API key to version control. The `.env` file is already in `.gitignore`.

## API Documentation

FastAPI automatically generates interactive API documentation:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
