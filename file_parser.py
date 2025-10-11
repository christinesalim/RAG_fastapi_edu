"""
File parser module for RAG application.
Contains parsers for different file types (text, PDF, etc.).
"""

import io
import logging
from abc import ABC, abstractmethod
from typing import Type, Dict

import fitz  # PyMuPDF
import PyPDF2
import pytesseract
from PIL import Image


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


class PdfParser(BaseParser):
    """Parser for PDF files with OCR fallback."""

    def parse(self, filepath: str) -> str:
        """
        Parse a PDF file and extract text content.

        Args:
            filepath: Path to the PDF file

        Returns:
            Extracted text content from all pages

        Raises:
            IOError: If PDF cannot be read or decrypted
        """
        try:
            content: str = ""
            with open(filepath, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                if reader.is_encrypted:
                    try:
                        # Decrypt with empty password
                        reader.decrypt('')
                    except Exception as e:
                        logging.error(f"Failed to decrypt PDF file: {e}")
                        raise IOError(f"Failed to decrypt PDF: {e}") from e

                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    page_content = page.extract_text()
                    if not page_content:  # If text extraction fails use OCR
                        page_content = self._ocr_page(filepath, page_num)
                    content += page_content

            return content
        except IOError:
            raise
        except Exception as e:
            logging.error(f"Error processing PDF: {e}")
            raise IOError(f"Error processing PDF file: {e}") from e

    def _ocr_page(self, filepath: str, page_num: int) -> str:
        """
        Extract text from a PDF page using OCR.

        Args:
            filepath: Path to the PDF file
            page_num: Page number to process (0-indexed)

        Returns:
            Extracted text from the page via OCR
        """
        try:
            document = fitz.open(filepath)
            page = document.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            ocr_text = pytesseract.image_to_string(img)
            document.close()
            return ocr_text
        except Exception as e:
            logging.error(f"OCR processing error: {e}")
            return ""


class ParserFactory:
    """Factory class for creating file parsers based on extension."""

    _parsers: Dict[str, Type[BaseParser]] = {}

    @classmethod
    def register_parser(cls, extension: str, parser: Type[BaseParser]) -> None:
        """
        Register a parser for a file extension.

        Args:
            extension: File extension (without dot)
            parser: Parser class to handle this extension
        """
        cls._parsers[extension] = parser

    @classmethod
    def get_parser(cls, extension: str) -> BaseParser:
        """
        Get a parser instance for the given extension.

        Args:
            extension: File extension (without dot)

        Returns:
            Parser instance for the extension

        Raises:
            ValueError: If no parser is registered for the extension
        """
        parser = cls._parsers.get(extension)
        if not parser:
            raise ValueError(f"No parser found for extension {extension}")
        return parser()


ParserFactory.register_parser('txt', TextParser)
ParserFactory.register_parser('pdf', PdfParser)

    