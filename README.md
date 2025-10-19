# RAG FastAPI Project

## Architecture

![RAG Architecture Diagram](images/rag-architecture.png)

The system uses a Retrieval-Augmented Generation (RAG) pattern with:

- **Embedding Creator**: Converts user questions into embeddings
- **PostgreSQL + pgvector**: Stores and retrieves document chunks using vector similarity
- **LLM**: Generates answers by combining the question with relevant chunks

## The Complete Flow

```
User uploads file
       ↓
FastAPI receives it
       ↓
Parse file (text/PDF)
       ↓
Save to database (files table)
       ↓
Background task: chunk_and_embed (async, doesn't block)
       ↓
User gets immediate response
       ↓
Later: Background task finishes
       ↓
Chunks saved to database (file_chunks table)
       ↓
User can now search/ask questions
```

## Setup

### Prerequisites

- Python 3.8+
- PostgreSQL with pgvector extension (for full functionality)
- OpenAI API key

### Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Configure environment variables in `.env`:

```bash
# Required for all features
OPENAI_API_KEY=your_openai_api_key_here

# Required for database features (file upload, RAG)
POSTGRES_USERNAME=your_username
POSTGRES_PASSWORD=your_password
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
DATABASE_NAME=fastapi_rag_db
```

3. Run the server:

```bash
uvicorn main:app --reload
```

### Running Without PostgreSQL

The application can run in **demo mode** without PostgreSQL. If database credentials are not provided:

- ✅ Server starts successfully
- ✅ Health check endpoint (`GET /`) works
- ⚠️  Database features return `503` error with helpful message
- ⚠️  File upload, RAG, and similarity search require PostgreSQL

This is useful for:
- Testing the application structure
- Running in Educative or similar environments
- Demonstrating the API without full database setup

**Example output without database:**
```bash
⚠️  Database credentials not found. Running without database.
INFO:     Started server process
INFO:     Uvicorn running on http://127.0.0.1:8000
```

**API response without database:**
```bash
curl http://localhost:8000/
# Response: {"message": "Database not available. Please configure PostgreSQL connection."}
```

## API Endpoints

### `GET /`

**List all uploaded files**

Returns a list of all files stored in the database.

**Response:**

```json
[
  {
    "file_id": 1,
    "file_name": "document.pdf"
  },
  {
    "file_id": 2,
    "file_name": "notes.txt"
  }
]
```

**Example:**

```bash
curl http://localhost:8000/
```

---

### `POST /uploadfile/`

**Upload and process a document**

Uploads a file (.txt or .pdf), parses its content, saves it to the database, and schedules background processing to chunk and embed the content.

**Request:**

- Form data with `file` field
- Supported formats: `.txt`, `.pdf`

**Response:**

```json
{
  "info": "File saved",
  "file_name": "document.pdf"
}
```

**Example:**

```bash
curl -X POST 'http://localhost:8000/uploadfile/' \
  -F 'file=@document.pdf'
```

**Errors:**

- `400` - No file provided or file type not allowed
- `500` - Error saving or processing file

---

### `POST /find-similar-chunks/{file_id}`

**Find semantically similar chunks**

Retrieves the top 10 most relevant text chunks from a specific file based on semantic similarity to the question.

**Request:**

- Path parameter: `file_id` (integer)
- JSON body:

```json
{
  "question": "What is the main topic?"
}
```

**Response:**

```json
[
  {
    "chunk_id": 15,
    "chunk_text": "Barack Obama was born in Hawaii in 1961..."
  },
  {
    "chunk_id": 16,
    "chunk_text": "He served as the 44th President..."
  }
]
```

**Example:**

```bash
curl -X POST 'http://localhost:8000/find-similar-chunks/1' \
  -H 'Content-Type: application/json' \
  -d '{"question": "Where was Obama born?"}'
```

---

### `POST /ask/`

**Ask a question with RAG (Retrieval-Augmented Generation)**

Answers questions about a document by combining vector similarity search with GPT-3.5-turbo. Retrieves relevant chunks and uses them as context for the LLM.

**Request:**

```json
{
  "document_id": 1,
  "question": "Where was Obama born?"
}
```

**Response:**

```json
{
  "response": "Barack Obama was born in Hawaii in 1961."
}
```

**Example:**

```bash
curl -X POST 'http://localhost:8000/ask/' \
  -H 'Content-Type: application/json' \
  -d '{
    "document_id": 1,
    "question": "Where was Obama born?"
  }'
```

**How it works:**

1. Converts question to embedding vector
2. Finds top 10 similar chunks using L2 distance
3. Combines chunks into context
4. Sends question + context to GPT-3.5-turbo
5. Returns AI-generated answer

## Environment Setup

1. Copy your OpenAI API key to the `.env` file:

```bash
# Edit .env file and replace with your actual API key
OPENAI_API_KEY=your_openai_api_key_here
```

## API Documentation

FastAPI automatically generates interactive API documentation:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Key Takeaways

Run the parser test: /opt/anaconda3/bin/python test_parser.py

1. Factory Pattern - One interface to get the right parser: ParserFactory.get_parser('pdf')
2. Dual approach for PDFs:

- First tries direct text extraction (faster)
- Falls back to OCR for scanned PDFs (slower but works on images)

3. PyMuPDF vs PyPDF2:

- PyPDF2: Text extraction from digital PDFs
- PyMuPDF (fitz): Converts PDF pages to images for OCR

4. OCR pipeline: PDF → Image (pixmap) → PIL Image → Tesseract → Text

The OCR result shows some character recognition quirks (like "Il" instead of "II"), which is typical for OCR - it's not perfect but gets most of the text!

# Database Schema

![Database Schema](images/schema.png)

files table: holds data for each file

file_chunks table: breaks file contents into chunks and embeddings for 1536-dimensional vector (OpenAI embedding size)

## Why break files into chunks?

- Large files are split into smaller chunks
- Each chunk gets an embedding
- Enables semantic search on smaller pieces

## File Upload and Chunking Flow

```
User uploads file (obama.txt)
       ↓
File saved to database
  • file_name: "obama.txt"
  • file_content: "Barack Obama was..." (Full text)
       ↓
TextProcessor reads file.file_content
       ↓
Splits content into sentences
       ↓
Groups sentences into chunks
       ↓
Each chunk saved to file_chunks table
  • chunk_text: "Barack Obama was..." (Piece of content)
  • embedding_vector: [0.023, -0.009, ...]
  • file_id: Links back to parent file
```

## Pydantic Models

Pydantic models are used to define the shape and type of data the endpoints expect. They provide automatic validation, serialization, and documentation for API requests.

### QuestionModel

Used by the `/find-similar-chunks/{file_id}` endpoint to accept a question for similarity search.

```python
class QuestionModel(BaseModel):
    question: str
```

**Fields:**
- `question` (str, required) - The question text to search for similar chunks

**Example Request:**
```json
{
  "question": "Where was Obama born?"
}
```

**Usage:**
```bash
curl -X POST 'http://localhost:8000/find-similar-chunks/1' \
  -H 'Content-Type: application/json' \
  -d '{"question": "Where was Obama born?"}'
```

---

### AskModel

Used by the `/ask/` endpoint for RAG-based question answering.

```python
class AskModel(BaseModel):
    document_id: int
    question: str
```

**Fields:**
- `document_id` (int, required) - The ID of the uploaded file to search in
- `question` (str, required) - The question to answer using the document context

**Example Request:**
```json
{
  "document_id": 1,
  "question": "Where was Obama born?"
}
```

**Usage:**
```bash
curl -X POST 'http://localhost:8000/ask/' \
  -H 'Content-Type: application/json' \
  -d '{
    "document_id": 1,
    "question": "Where was Obama born?"
  }'
```

---

### Benefits of Pydantic Models

1. **Automatic Validation** - FastAPI automatically validates incoming data against the model schema
2. **Type Safety** - Ensures data types are correct (e.g., `document_id` must be an integer)
3. **Auto-generated Documentation** - Models appear in Swagger UI at http://localhost:8000/docs
4. **Clear Error Messages** - Returns helpful 422 validation errors if data is invalid
5. **IDE Support** - Provides autocomplete and type hints in your code editor

**Example Validation Error:**
```bash
# Missing required field
curl -X POST 'http://localhost:8000/ask/' \
  -H 'Content-Type: application/json' \
  -d '{"document_id": 1}'

# Response: 422 Unprocessable Entity
{
  "detail": [
    {
      "loc": ["body", "question"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

## Testing

The project includes comprehensive unit tests for all components.

### Run All Tests

```bash
pytest -v
```

or with Anaconda Python:

```bash
/opt/anaconda3/bin/python -m pytest -v
```

### Test Files

- **[test_background_tasks.py](test_background_tasks.py)** - 17 tests for `TextProcessor` (chunking and embedding)
- **[test_main.py](test_main.py)** - 14 passing tests + 4 skipped tests for FastAPI endpoints

### Test Coverage

**Background Tasks:**

- Text chunking with NLTK
- OpenAI embedding creation
- Database storage of chunks
- Edge cases (empty text, special characters)

**API Endpoints:**

- File listing (`GET /`)
- File upload (`POST /uploadfile/`)
- Question answering (`POST /ask/`)
- Integration tests (full workflow)
- Error handling and validation

### Run Specific Tests

```bash
# Run only endpoint tests
pytest test_main.py -v

# Run only background task tests
pytest test_background_tasks.py -v

# Run tests matching a pattern
pytest -v -k "upload"

# Show detailed output
pytest -v -s
```

### Test Summary

```
14 passed, 4 skipped (test_main.py)
17 passed (test_background_tasks.py)
─────────────────────────────
31 total passing tests
```

**Note:** 4 tests are skipped in `test_main.py` because they require PostgreSQL's pgvector extension. The functionality is fully tested through the endpoint tests instead.
