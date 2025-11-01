"""
File Classifier Module

Copyright (c) 2025 Alexandru Emanuel Vasile. All rights reserved.
Proprietary Software - 200-Key Limited Release License

This module handles file classification using a hybrid approach:
1. Rule-based classification (fast, deterministic)
2. AI-powered classification (semantic, context-aware)

The classifier determines the appropriate category and destination path
for files based on their type, name, and content.

NOTICE: This software is proprietary and confidential.
See LICENSE.txt for full terms and conditions.

Author: Alexandru Emanuel Vasile
License: Proprietary (200-key limited release)
"""

import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import mimetypes

# Try to import text extraction libraries
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


class FileClassifier:
    """
    Hybrid file classifier combining rule-based and AI-powered classification.

    Attributes:
        config: Configuration object
        ollama_client: Optional Ollama client for AI classification
        destination_rules: Extension-to-path mapping
        enable_ai: Whether to use AI classification
    """

    def __init__(self, config, ollama_client=None):
        """
        Initialize file classifier.

        Args:
            config: Configuration object
            ollama_client: Optional Ollama client for AI classification
        """
        self.config = config
        self.ollama_client = ollama_client
        self.destination_rules = config.destination_rules
        self.enable_ai = config.enable_ai and ollama_client is not None
        self.text_extract_limit = config.text_extract_limit

    def classify(self, file_path: str, deep_analysis: bool = False) -> Dict[str, Any]:
        """
        Classify a file and suggest organization action.

        This is the main entry point that combines rule-based, AI, and agent classification.

        Args:
            file_path (str): Path to the file to classify
            deep_analysis (bool): If True, use agent analyzer for deep multi-step analysis

        Returns:
            Dict: Classification result containing:
                - category (str): File category
                - suggested_path (str): Destination path
                - rename (str or None): Suggested new filename
                - reason (str): Explanation
                - confidence (str): 'high', 'medium', or 'low'
                - method (str): 'rule-based', 'ai', or 'agent'
                - evidence (list, optional): Evidence strings (if agent used)
                - action (str, optional): Suggested action (if agent used)
                - block_reason (str, optional): Reason for blocking (if agent used)
        """
        path = Path(file_path)

        # Basic file information
        file_info = self._extract_file_info(path)

        # Stage 1: Rule-based classification
        rule_result = self._classify_by_rules(file_info)

        # Stage 2: Check if we should use agent deep analysis
        if deep_analysis or rule_result['confidence'] == 'low':
            # Try agent analysis if available
            agent_result = self._classify_by_agent(file_path)
            if agent_result and agent_result.get('success') and agent_result.get('confidence') in ['high', 'medium']:
                return agent_result

        # If rule-based gives high confidence and no deep analysis requested, use it
        if rule_result['confidence'] == 'high' and not deep_analysis:
            return rule_result

        # Stage 3: Try standard AI classification if enabled
        if self.enable_ai and self.ollama_client:
            ai_result = self._classify_by_ai(file_info)
            if ai_result.get('success'):
                return {
                    'category': ai_result.get('category', 'Unsorted'),
                    'suggested_path': ai_result.get('suggested_path'),
                    'rename': ai_result.get('rename'),
                    'reason': ai_result.get('reason', 'AI classification'),
                    'confidence': 'high',
                    'method': 'ai'
                }

        # Fallback to rule-based result
        return rule_result

    def _extract_file_info(self, path: Path) -> Dict[str, Any]:
        """
        Extract comprehensive file information.

        Args:
            path (Path): Path to the file

        Returns:
            Dict: File information including name, extension, size, mime type, etc.
        """
        stat = path.stat()
        extension = path.suffix.lower().lstrip('.')

        # Determine MIME type
        mime_type, _ = mimetypes.guess_type(str(path))

        # Extract text snippet if possible
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
            path (Path): File path
            extension (str): File extension

        Returns:
            str or None: Extracted text snippet (up to configured limit)
        """
        try:
            # Plain text files
            if extension in ['txt', 'md', 'log', 'csv', 'json', 'xml', 'html', 'py', 'js', 'java', 'cpp', 'h']:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read(self.text_extract_limit)

            # PDF files
            elif extension == 'pdf' and PDF_SUPPORT:
                return self._extract_pdf_text(path)

            # DOCX files
            elif extension == 'docx' and DOCX_SUPPORT:
                return self._extract_docx_text(path)

        except Exception as e:
            # If extraction fails, just return None
            pass

        return None

    def _extract_pdf_text(self, path: Path) -> Optional[str]:
        """Extract text from PDF file (first page only)."""
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
        """Extract text from DOCX file."""
        try:
            doc = Document(path)
            text = '\n'.join([para.text for para in doc.paragraphs[:5]])  # First 5 paragraphs
            return text[:self.text_extract_limit]
        except Exception:
            pass
        return None

    def _classify_by_rules(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify file using rule-based approach.

        Args:
            file_info (Dict): File information

        Returns:
            Dict: Classification result
        """
        extension = file_info['extension']
        filename = file_info['filename'].lower()
        stem = file_info['stem'].lower()

        # Check destination rules
        if extension in self.destination_rules:
            suggested_path = self.destination_rules[extension]
            category = self._path_to_category(suggested_path)

            # Try to detect specific subcategories
            refined_path = self._refine_path_by_patterns(filename, stem, suggested_path)

            return {
                'category': category,
                'suggested_path': refined_path,
                'rename': self._suggest_rename_by_patterns(filename, stem),
                'reason': f'Classified by file extension (.{extension})',
                'confidence': 'high',
                'method': 'rule-based'
            }

        # Pattern-based classification for unknown extensions
        pattern_result = self._classify_by_patterns(filename, stem)
        if pattern_result:
            return pattern_result

        # Default: Unsorted
        return {
            'category': 'Unsorted',
            'suggested_path': 'Unsorted/',
            'rename': None,
            'reason': 'No matching classification rules',
            'confidence': 'low',
            'method': 'rule-based'
        }

    def _classify_by_patterns(self, filename: str, stem: str) -> Optional[Dict[str, Any]]:
        """
        Classify files based on filename patterns.

        Args:
            filename (str): Lowercase filename
            stem (str): Lowercase filename without extension

        Returns:
            Dict or None: Classification result if pattern matches
        """
        # Invoice patterns
        if re.search(r'(invoice|receipt|bill)', stem):
            return {
                'category': 'Finance',
                'suggested_path': 'Documents/Finance/Invoices/',
                'rename': None,
                'reason': 'Detected invoice-related keywords',
                'confidence': 'medium',
                'method': 'rule-based'
            }

        # Resume/CV patterns
        if re.search(r'(resume|cv|curriculum)', stem):
            return {
                'category': 'Documents',
                'suggested_path': 'Documents/Personal/Resume/',
                'rename': None,
                'reason': 'Detected resume/CV keywords',
                'confidence': 'medium',
                'method': 'rule-based'
            }

        # Screenshot patterns
        if re.search(r'(screenshot|screen shot|capture)', stem):
            return {
                'category': 'Pictures',
                'suggested_path': 'Pictures/Screenshots/',
                'rename': None,
                'reason': 'Detected screenshot pattern',
                'confidence': 'high',
                'method': 'rule-based'
            }

        # Project/code patterns
        if re.search(r'(project|code|src|source)', stem):
            return {
                'category': 'Projects',
                'suggested_path': 'Projects/',
                'rename': None,
                'reason': 'Detected project-related keywords',
                'confidence': 'medium',
                'method': 'rule-based'
            }

        return None

    def _refine_path_by_patterns(self, filename: str, stem: str, base_path: str) -> str:
        """
        Refine destination path based on filename patterns.

        Args:
            filename (str): Lowercase filename
            stem (str): Lowercase filename without extension
            base_path (str): Base destination path

        Returns:
            str: Refined path with potential subdirectories
        """
        # Extract year if present (YYYY format)
        year_match = re.search(r'(20\d{2})', stem)
        if year_match:
            year = year_match.group(1)
            return f"{base_path}{year}/"

        return base_path

    def _suggest_rename_by_patterns(self, filename: str, stem: str) -> Optional[str]:
        """
        Suggest better filename based on patterns.

        Args:
            filename (str): Current filename
            stem (str): Filename without extension

        Returns:
            str or None: Suggested new name, or None if current name is good
        """
        # Check for unclear names
        unclear_patterns = [
            r'^(untitled|document|file|download|image|photo|scan)[\d\-_]*$',
            r'^[a-z0-9]{8,}$',  # Random hash-like names
            r'^\d+$'  # Just numbers
        ]

        for pattern in unclear_patterns:
            if re.match(pattern, stem.lower()):
                # Current name is unclear, but we need more context to suggest better name
                # This would be better handled by AI
                return None

        # Current name seems reasonable
        return None

    def _classify_by_ai(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify file using AI (Ollama).

        Args:
            file_info (Dict): File information

        Returns:
            Dict: AI classification result
        """
        if not self.ollama_client:
            return {'success': False, 'error': 'No AI client available'}

        return self.ollama_client.classify_file(
            filename=file_info['filename'],
            extension=file_info['extension'],
            text_snippet=file_info.get('text_snippet'),
            file_size=file_info['size']
        )

    def _classify_by_agent(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Classify file using deep agent analysis.

        This method uses the AgentAnalyzer for multi-step reasoning
        and policy-aware classification.

        Args:
            file_path (str): Path to the file

        Returns:
            Dict or None: Agent classification result or None if agent unavailable
        """
        try:
            # Lazy import to avoid circular dependencies
            from agent.agent_analyzer import AgentAnalyzer

            # Only use agent if Ollama is available
            if not self.ollama_client or not self.ollama_client.is_available():
                return None

            # Get folder policy for this file
            policy = self.config.get_folder_policy(file_path)

            # Create analyzer (we don't have db_manager in classifier, pass None)
            analyzer = AgentAnalyzer(self.config, self.ollama_client, db_manager=None)

            # Perform analysis
            result = analyzer.analyze_file(file_path, policy=policy)

            return result

        except ImportError:
            # Agent module not available
            return None
        except Exception as e:
            # Agent analysis failed, return None to fallback
            return None

    def _path_to_category(self, path: str) -> str:
        """
        Extract category name from path.

        Args:
            path (str): Destination path

        Returns:
            str: Category name
        """
        parts = path.strip('/').split('/')
        return parts[0] if parts else 'Unsorted'


def classify_file(file_path: str, config, ollama_client=None) -> Dict[str, Any]:
    """
    Convenience function to classify a single file.

    Args:
        file_path (str): Path to file
        config: Configuration object
        ollama_client: Optional Ollama client

    Returns:
        Dict: Classification result
    """
    classifier = FileClassifier(config, ollama_client)
    return classifier.classify(file_path)


if __name__ == "__main__":
    # Test classifier
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from config import get_config
    from ai.ollama_client import OllamaClient

    config = get_config()
    ollama = OllamaClient(config.ollama_base_url, config.ollama_model)

    classifier = FileClassifier(config, ollama if ollama.is_available() else None)

    # Test with a dummy file
    test_file = "test_invoice_2025.pdf"
    print(f"Classifying: {test_file}")

    result = classifier._classify_by_rules({
        'path': test_file,
        'filename': test_file,
        'stem': 'test_invoice_2025',
        'extension': 'pdf',
        'size': 12345,
        'mime_type': 'application/pdf',
        'text_snippet': None,
        'modified_time': 0
    })

    print(f"Result: {result}")
