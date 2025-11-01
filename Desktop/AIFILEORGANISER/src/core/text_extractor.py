"""
Text Extraction Module

Copyright (c) 2025 Alexandru Emanuel Vasile. All rights reserved.
Proprietary Software - 200-Key Limited Release License

Shared text extraction functionality without circular dependencies.
This module provides text extraction from various file types (PDF, DOCX, plain text)
without depending on the classifier module, breaking the circular import.

NOTICE: This software is proprietary and confidential.
See LICENSE.txt for full terms and conditions.

Author: Alexandru Emanuel Vasile
License: Proprietary (200-key limited release)
"""

from pathlib import Path
from typing import Dict, Any, Optional
import mimetypes

try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from docx import Document
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False


class TextExtractor:
    """Shared text extraction without classifier dependency."""

    def __init__(self, config):
        """
        Initialize text extractor.

        Args:
            config: Configuration object with text_extract_limit
        """
        self.config = config
        self.text_extract_limit = config.text_extract_limit

    def extract_file_info(self, path: Path) -> Dict[str, Any]:
        """
        Extract comprehensive file information.

        Args:
            path (Path): Path to file

        Returns:
            Dict: File information including:
                - path (str): File path
                - filename (str): File name
                - stem (str): File name without extension
                - extension (str): File extension
                - size (int): File size in bytes
                - mime_type (str): MIME type
                - text_snippet (str): Extracted text content (if available)
                - modified_time (float): Last modified timestamp
        """
        stat = path.stat()
        extension = path.suffix.lower().lstrip('.')
        mime_type, _ = mimetypes.guess_type(str(path))
        text_snippet = self._extract_text(path, extension)

        return {
            'path': str(path),
            'filename': path.name,
            'stem': path.stem,
            'extension': extension,
            'size': stat.st_size,
            'mime_type': mime_type,
            'text_snippet': text_snippet,
            'modified_time': stat.st_mtime
        }

    def _extract_text(self, path: Path, extension: str) -> Optional[str]:
        """
        Extract text content from file for AI analysis.

        Args:
            path (Path): Path to file
            extension (str): File extension

        Returns:
            str or None: Extracted text or None if extraction fails
        """
        try:
            if extension in ['txt', 'md', 'log', 'csv', 'json', 'xml', 'html']:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read(self.text_extract_limit)

            elif extension == 'pdf' and PDF_SUPPORT:
                return self._extract_pdf_text(path)

            elif extension == 'docx' and DOCX_SUPPORT:
                return self._extract_docx_text(path)

        except Exception:
            pass

        return None

    def _extract_pdf_text(self, path: Path) -> Optional[str]:
        """
        Extract text from PDF file.

        Args:
            path (Path): Path to PDF file

        Returns:
            str or None: Extracted text or None if extraction fails
        """
        try:
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                if len(reader.pages) > 0:
                    text = reader.pages[0].extract_text()
                    return text[:self.text_extract_limit]
        except Exception:
            pass
        return None

    def _extract_docx_text(self, path: Path) -> Optional[str]:
        """
        Extract text from DOCX file.

        Args:
            path (Path): Path to DOCX file

        Returns:
            str or None: Extracted text or None if extraction fails
        """
        try:
            doc = Document(path)
            text = '\n'.join([para.text for para in doc.paragraphs[:5]])
            return text[:self.text_extract_limit]
        except Exception:
            pass
        return None
