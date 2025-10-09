"""
File parser module for RAG application.
Contains parsers for different file types (text, PDF, etc.).
"""

import logging
from abc import ABC, abstractmethod

class BaseParser(ABC):
  """Abstract base class for file parsers"""
  
  @abstractmethod
  def parse(self, filepath: str) -> str:
    """
    Abstract method to parse a file

    Args:
        filepath (str): path to file
        
    Returns:
        Extracted text content as string
    """
    pass
  
  
class TextParser(BaseParser):
  """Parser for plain text files (.txt)."""
  
  def parse(self, filepath: str) -> str:
    """
    Parse a text file and return its content.

    Args:
        filepath: Path to the text file

    Returns:
        Content of the text file

    Raises:
        FileNotFoundError: If the file doesn't exist
        PermissionError: If the file cannot be read due to permissions
        UnicodeDecodeError: If the file cannot be decoded as UTF-8
        IOError: For other I/O related errors
    """
    try:
      with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()
    except FileNotFoundError as e:
      logging.error(f"File not found: {filepath}")
      raise FileNotFoundError(f"File not found: {filepath}") from e
    except PermissionError as e:
      logging.error(f"Permission denied reading file: {filepath}")
      raise PermissionError(f"Permission denied reading file: {filepath}") from e
    except UnicodeDecodeError as e:
      logging.error(f"File is not valid UTF-8 text: {filepath}")
      raise UnicodeDecodeError(
        e.encoding, e.object, e.start, e.end,
        f"File is not valid UTF-8 text: {filepath}"
      ) from e
    except IOError as e:
      logging.error(f"Error reading file {filepath}: {str(e)}")
      raise IOError(f"Error reading file {filepath}: {str(e)}") from e

      