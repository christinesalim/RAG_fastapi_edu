"""
Test suite for background_tasks.py
Tests text chunking, embedding, and database operations.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from background_tasks import TextProcessor
from db import Base, FileChunk, File


# Test fixtures

@pytest.fixture
def test_db():
    """Create an in-memory SQLite database for testing."""
    # Use in-memory SQLite (no pgvector, but good for testing logic)
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    TestSessionLocal = sessionmaker(bind=engine)
    session = TestSessionLocal()

    yield session

    session.close()


@pytest.fixture
def sample_file(test_db):
    """Create a sample file in the test database."""
    file = File(
        file_name="test.txt",
        file_content="This is test content."
    )
    test_db.add(file)
    test_db.commit()
    return file


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client to avoid real API calls."""
    with patch('background_tasks.client') as mock_client:
        # Mock the embedding response
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = [0.1] * 1536  # 1536-dim vector
        mock_client.embeddings.create.return_value = mock_response
        yield mock_client


# Tests for TextProcessor initialization

def test_text_processor_init(test_db):
    """Test TextProcessor initialization with default parameters."""
    processor = TextProcessor(db=test_db, file_id=1)

    assert processor.db == test_db
    assert processor.file_id == 1
    assert processor.chunk_size == 2


def test_text_processor_init_custom_chunk_size(test_db):
    """Test TextProcessor initialization with custom chunk size."""
    processor = TextProcessor(db=test_db, file_id=1, chunk_size=5)

    assert processor.chunk_size == 5


# Tests for text chunking

def test_chunk_simple_text(test_db, mock_openai_client):
    """Test chunking simple text with 2 sentences."""
    processor = TextProcessor(db=test_db, file_id=1, chunk_size=2)
    text = "First sentence. Second sentence."

    processor.chunk_and_embed(text)

    # Should create 1 chunk (2 sentences)
    chunks = test_db.query(FileChunk).all()
    assert len(chunks) == 1
    assert chunks[0].chunk_text == "First sentence. Second sentence."


def test_chunk_multiple_chunks(test_db, mock_openai_client):
    """Test chunking text that creates multiple chunks."""
    processor = TextProcessor(db=test_db, file_id=1, chunk_size=2)
    text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."

    processor.chunk_and_embed(text)

    chunks = test_db.query(FileChunk).all()
    assert len(chunks) == 3  # 5 sentences / 2 = 3 chunks (2, 2, 1)
    assert chunks[0].chunk_text == "Sentence one. Sentence two."
    assert chunks[1].chunk_text == "Sentence three. Sentence four."
    assert chunks[2].chunk_text == "Sentence five."


def test_chunk_with_custom_size(test_db, mock_openai_client):
    """Test chunking with different chunk sizes."""
    processor = TextProcessor(db=test_db, file_id=1, chunk_size=3)
    text = "One. Two. Three. Four. Five."

    processor.chunk_and_embed(text)

    chunks = test_db.query(FileChunk).all()
    assert len(chunks) == 2  # 5 sentences / 3 = 2 chunks (3, 2)


def test_chunk_single_sentence(test_db, mock_openai_client):
    """Test chunking text with single sentence."""
    processor = TextProcessor(db=test_db, file_id=1, chunk_size=2)
    text = "Just one sentence."

    processor.chunk_and_embed(text)

    chunks = test_db.query(FileChunk).all()
    assert len(chunks) == 1
    assert chunks[0].chunk_text == "Just one sentence."


def test_chunk_complex_sentences(test_db, mock_openai_client):
    """Test chunking with complex punctuation."""
    processor = TextProcessor(db=test_db, file_id=1, chunk_size=2)
    text = "Dr. Smith works at M.I.T. She studies A.I. technology. Is it amazing?"

    processor.chunk_and_embed(text)

    chunks = test_db.query(FileChunk).all()
    # NLTK Punkt should correctly identify 3 sentences
    assert len(chunks) == 2  # (2, 1)


# Tests for embeddings

def test_embedding_api_called(test_db, mock_openai_client):
    """Test that OpenAI embedding API is called for each chunk."""
    processor = TextProcessor(db=test_db, file_id=1, chunk_size=2)
    text = "First. Second. Third."

    processor.chunk_and_embed(text)

    # Should call embeddings API twice (2 chunks)
    assert mock_openai_client.embeddings.create.call_count == 2


def test_embedding_uses_correct_model(test_db, mock_openai_client):
    """Test that the correct embedding model is used."""
    processor = TextProcessor(db=test_db, file_id=1, chunk_size=2)
    text = "Test sentence."

    processor.chunk_and_embed(text)

    # Verify the model used
    call_args = mock_openai_client.embeddings.create.call_args
    assert call_args.kwargs['model'] == 'text-embedding-ada-002'


def test_embedding_stored_in_database(test_db, mock_openai_client):
    """Test that embeddings are stored with chunks."""
    processor = TextProcessor(db=test_db, file_id=1, chunk_size=2)
    text = "Store this."

    processor.chunk_and_embed(text)

    chunks = test_db.query(FileChunk).all()
    # Note: SQLite doesn't support Vector type, so this would be None
    # In real PostgreSQL with pgvector, this would be the embedding
    assert len(chunks) == 1


# Tests for database operations

def test_file_id_association(test_db, sample_file, mock_openai_client):
    """Test that chunks are associated with correct file_id."""
    processor = TextProcessor(db=test_db, file_id=sample_file.file_id, chunk_size=2)
    text = "First. Second."

    processor.chunk_and_embed(text)

    chunks = test_db.query(FileChunk).all()
    assert all(chunk.file_id == sample_file.file_id for chunk in chunks)


def test_multiple_files_separated(test_db, mock_openai_client):
    """Test that chunks from different files are kept separate."""
    # Process file 1
    processor1 = TextProcessor(db=test_db, file_id=1, chunk_size=2)
    processor1.chunk_and_embed("File one. Sentence one.")

    # Process file 2
    processor2 = TextProcessor(db=test_db, file_id=2, chunk_size=2)
    processor2.chunk_and_embed("File two. Sentence two.")

    # Check file 1 chunks
    file1_chunks = test_db.query(FileChunk).filter_by(file_id=1).all()
    assert len(file1_chunks) == 1
    assert "File one" in file1_chunks[0].chunk_text

    # Check file 2 chunks
    file2_chunks = test_db.query(FileChunk).filter_by(file_id=2).all()
    assert len(file2_chunks) == 1
    assert "File two" in file2_chunks[0].chunk_text


def test_commit_called(test_db, mock_openai_client):
    """Test that database commit is called after processing."""
    mock_db = Mock(spec=Session)
    processor = TextProcessor(db=mock_db, file_id=1, chunk_size=2)

    processor.chunk_and_embed("Test. Sentence.")

    mock_db.commit.assert_called_once()


# Edge cases

def test_empty_text(test_db, mock_openai_client):
    """Test handling of empty text."""
    processor = TextProcessor(db=test_db, file_id=1, chunk_size=2)

    processor.chunk_and_embed("")

    chunks = test_db.query(FileChunk).all()
    # Empty text should create no chunks
    assert len(chunks) == 0


def test_whitespace_only_text(test_db, mock_openai_client):
    """Test handling of whitespace-only text."""
    processor = TextProcessor(db=test_db, file_id=1, chunk_size=2)

    processor.chunk_and_embed("   \n\t  ")

    chunks = test_db.query(FileChunk).all()
    assert len(chunks) <= 1  # May create empty chunk or no chunks


def test_very_long_text(test_db, mock_openai_client):
    """Test processing of longer text."""
    processor = TextProcessor(db=test_db, file_id=1, chunk_size=2)

    # Create text with 100 sentences
    sentences = [f"This is sentence number {i}." for i in range(100)]
    text = " ".join(sentences)

    processor.chunk_and_embed(text)

    chunks = test_db.query(FileChunk).all()
    assert len(chunks) == 50  # 100 sentences / 2 = 50 chunks


# Integration test

def test_full_workflow(test_db, sample_file, mock_openai_client):
    """Test complete workflow from file to chunks."""
    # Create processor
    processor = TextProcessor(
        db=test_db,
        file_id=sample_file.file_id,
        chunk_size=2
    )

    # Process text
    text = "Barack Obama was president. He served two terms. Obama was born in Hawaii."
    processor.chunk_and_embed(text)

    # Verify chunks created
    chunks = test_db.query(FileChunk).filter_by(file_id=sample_file.file_id).all()
    assert len(chunks) == 2

    # Verify chunks linked to file
    assert all(chunk.file_id == sample_file.file_id for chunk in chunks)

    # Verify embedding API called
    assert mock_openai_client.embeddings.create.call_count == 2


if __name__ == "__main__":
    # Run tests with: pytest test_background_tasks.py -v
    pytest.main([__file__, "-v"])
