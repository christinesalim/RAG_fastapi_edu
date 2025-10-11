"""
Test script for file parsers.
"""

from file_parser import ParserFactory

def test_text_parser():
    """Test parsing a text file."""
    print("Testing TextParser...")
    parser = ParserFactory.get_parser('txt')
    content = parser.parse('obama.txt')
    print(f"✓ Text file parsed successfully")
    print(f"  Content length: {len(content)} characters")
    print(f"  First 100 chars: {content[:100]}...")
    print()

def test_pdf_parser():
    """Test parsing a PDF file."""
    print("Testing PdfParser...")
    parser = ParserFactory.get_parser('pdf')
    content = parser.parse('obama.pdf')
    print(f"✓ PDF file parsed successfully")
    print(f"  Content length: {len(content)} characters")
    print(f"  First 100 chars: {content[:100]}...")
    print()

def test_ocr_pdf_parser():
    """Test parsing a scanned PDF with OCR."""
    print("Testing PdfParser with OCR (obama-ocr.pdf)...")
    parser = ParserFactory.get_parser('pdf')
    content = parser.parse('obama-ocr.pdf')
    print(f"✓ OCR PDF file parsed successfully")
    print(f"  Content length: {len(content)} characters")
    print(f"  First 100 chars: {content[:100]}...")
    print()

def test_factory_error():
    """Test factory with unsupported extension."""
    print("Testing ParserFactory error handling...")
    try:
        parser = ParserFactory.get_parser('docx')
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
    print()

if __name__ == "__main__":
    print("=" * 60)
    print("File Parser Test Suite")
    print("=" * 60)
    print()

    test_text_parser()
    test_pdf_parser()
    test_ocr_pdf_parser()
    test_factory_error()

    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
