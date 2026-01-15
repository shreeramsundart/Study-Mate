import os
import io
import tempfile
from .logging_config import logger
from typing import Optional, Tuple

def _extract_pdf_text(content: bytes) -> str:
    """
    Extract text from PDF content using multiple methods
    
    Args:
        content: PDF file content as bytes
    
    Returns:
        Extracted text as string
    """
    logger.info("Starting PDF text extraction")
    
    extracted_text = ""
    
    # Method 1: Try PyPDF2 first (fastest)
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(content))
        
        text_parts = []
        total_pages = len(reader.pages)
        
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_parts.append(page_text)
                logger.debug(f"Extracted text from page {i+1}/{total_pages}")
            except Exception as page_error:
                logger.warning(f"Failed to extract text from page {i+1}: {page_error}")
                text_parts.append("")
        
        extracted_text = "\n".join(text_parts).strip()
        
        if extracted_text:
            logger.info(f"Successfully extracted {len(extracted_text)} characters using PyPDF2")
            return extracted_text
        else:
            logger.warning("PyPDF2 extracted empty text, trying OCR...")
    
    except ImportError:
        logger.error("PyPDF2 not installed. Please install: pip install PyPDF2")
    except Exception as e:
        logger.warning(f"PyPDF2 extraction failed: {e}. Trying fallback...")
    
    # Method 2: Try pdfplumber (better for complex layouts)
    try:
        import pdfplumber
        
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            text_parts = []
            for i, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        text_parts.append(page_text)
                except Exception as page_error:
                    logger.warning(f"pdfplumber failed on page {i+1}: {page_error}")
            
            extracted_text = "\n".join(text_parts).strip()
            
            if extracted_text:
                logger.info(f"Successfully extracted {len(extracted_text)} characters using pdfplumber")
                return extracted_text
    
    except ImportError:
        logger.debug("pdfplumber not installed. Skipping...")
    except Exception as e:
        logger.warning(f"pdfplumber extraction failed: {e}")
    
    # Method 3: OCR fallback for scanned PDFs
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
        
        logger.info("Converting PDF to images for OCR...")
        
        # Convert PDF to images
        images = convert_from_bytes(content, dpi=300)
        logger.info(f"Converted {len(images)} pages to images")
        
        ocr_text_parts = []
        
        for i, image in enumerate(images):
            try:
                # Perform OCR
                page_text = pytesseract.image_to_string(image)
                if page_text.strip():
                    ocr_text_parts.append(page_text)
                logger.info(f"OCR completed for page {i+1}/{len(images)}")
            except Exception as ocr_error:
                logger.warning(f"OCR failed for page {i+1}: {ocr_error}")
        
        extracted_text = "\n".join(ocr_text_parts).strip()
        
        if extracted_text:
            logger.info(f"Successfully extracted {len(extracted_text)} characters using OCR")
            return extracted_text
        else:
            logger.error("OCR extraction failed to produce text")
    
    except ImportError as import_error:
        logger.error(f"OCR dependencies missing: {import_error}")
        logger.error("Install with: pip install pdf2image pytesseract pillow")
        logger.error("Also install Tesseract OCR: https://github.com/tesseract-ocr/tesseract")
    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
    
    # Method 4: Try pymupdf (fitz) as last resort
    try:
        import fitz  # PyMuPDF
        
        doc = fitz.open(stream=content, filetype="pdf")
        text_parts = []
        
        for i in range(len(doc)):
            try:
                page = doc.load_page(i)
                page_text = page.get_text()
                if page_text.strip():
                    text_parts.append(page_text)
            except Exception as page_error:
                logger.warning(f"PyMuPDF failed on page {i+1}: {page_error}")
        
        extracted_text = "\n".join(text_parts).strip()
        
        if extracted_text:
            logger.info(f"Successfully extracted {len(extracted_text)} characters using PyMuPDF")
            return extracted_text
    
    except ImportError:
        logger.debug("PyMuPDF not installed. Skipping...")
    except Exception as e:
        logger.error(f"PyMuPDF extraction failed: {e}")
    
    # Final fallback
    if not extracted_text.strip():
        logger.error("All PDF text extraction methods failed")
        return ""
    
    return extracted_text

def extract_text_from_file(file_path: str) -> Tuple[Optional[str], str]:
    """
    Extract text from various file types
    
    Args:
        file_path: Path to the file
    
    Returns:
        Tuple of (extracted_text, error_message)
    """
    try:
        if not os.path.exists(file_path):
            return None, "File does not exist"
        
        # Check file extension
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            with open(file_path, 'rb') as f:
                content = f.read()
            text = _extract_pdf_text(content)
            return text, ""
        
        elif ext in ['.txt', '.md', '.csv', '.json', '.xml']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read(), ""
        
        else:
            return None, f"Unsupported file type: {ext}"
    
    except Exception as e:
        logger.exception(f"Error extracting text from file {file_path}: {e}")
        return None, str(e)