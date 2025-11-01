"""
Ollama Client Module

Copyright (c) 2025 Alexandru Emanuel Vasile. All rights reserved.
Proprietary Software - 200-Key Limited Release License

This module provides integration with local Ollama instance for AI-powered
file classification. It handles prompt construction, API communication,
and response parsing for semantic file understanding.

NOTICE: This software is proprietary and confidential.
See LICENSE.txt for full terms and conditions.

Author: Alexandru Emanuel Vasile
License: Proprietary (200-key limited release)
"""

import json
import requests
from typing import Dict, Any, Optional
from pathlib import Path


class OllamaClient:
    """
    Client for communicating with local Ollama instance.

    This client sends file metadata and text snippets to Ollama for intelligent
    classification and organization suggestions.

    Attributes:
        base_url (str): Ollama API base URL (default: http://localhost:11434)
        model (str): Ollama model to use (default: llama3)
        timeout (int): Request timeout in seconds
    """

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3", timeout: int = 30):
        """
        Initialize Ollama client.

        Args:
            base_url (str): Ollama API endpoint
            model (str): Model name to use for inference
            timeout (int): Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout

    def is_available(self) -> bool:
        """
        Check if Ollama service is available and running.

        Returns:
            bool: True if Ollama is accessible, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def list_models(self) -> list:
        """
        List available Ollama models.

        Returns:
            list: List of available model names
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
        except requests.exceptions.RequestException as e:
            print(f"Error listing models: {e}")

        return []

    def _construct_classification_prompt(self, filename: str, extension: str,
                                        text_snippet: Optional[str] = None,
                                        file_size: Optional[int] = None) -> str:
        """
        Construct prompt for file classification.

        Args:
            filename (str): Name of the file
            extension (str): File extension
            text_snippet (str, optional): Extracted text from file
            file_size (int, optional): File size in bytes

        Returns:
            str: Formatted prompt for Ollama
        """
        size_info = f"\nSize: {file_size} bytes" if file_size else ""
        snippet_info = f"\nContent preview:\n{text_snippet[:500]}" if text_snippet else ""

        prompt = f"""You are a file classification AI assistant. Your task is to analyze file information and suggest an organized storage location.

File Information:
- Filename: {filename}
- Type: {extension}{size_info}{snippet_info}

Based on this information, provide a classification suggestion in the following JSON format:
{{
  "category": "The main category (e.g., Documents, Finance, Projects, Media)",
  "suggested_path": "Relative path for organization (e.g., Documents/Invoices/2025/)",
  "rename": "Suggested filename if renaming would improve clarity (or null if current name is good)",
  "reason": "Brief explanation (1-2 sentences) for your suggestion"
}}

Important guidelines:
1. Choose clear, intuitive categories
2. Use date-based subfolders (YYYY/MM) when appropriate for time-sensitive documents
3. Only suggest renaming if the current filename is unclear or could be improved
4. Keep paths concise but descriptive
5. Return ONLY the JSON object, no additional text

Provide your classification:"""

        return prompt

    def classify_file(self, filename: str, extension: str,
                     text_snippet: Optional[str] = None,
                     file_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Classify a file using Ollama AI.

        Args:
            filename (str): Name of the file
            extension (str): File extension
            text_snippet (str, optional): Extracted text content
            file_size (int, optional): File size in bytes

        Returns:
            Dict: Classification result with keys:
                - category (str): File category
                - suggested_path (str): Suggested destination path
                - rename (str or None): Suggested new filename
                - reason (str): Explanation for classification
                - success (bool): Whether classification succeeded
                - error (str, optional): Error message if failed
        """
        # Default fallback response
        fallback = {
            "category": "Unsorted",
            "suggested_path": None,
            "rename": None,
            "reason": "AI classification unavailable",
            "success": False
        }

        # Check if Ollama is available
        if not self.is_available():
            fallback["error"] = "Ollama service not available"
            return fallback

        # Construct prompt
        prompt = self._construct_classification_prompt(filename, extension, text_snippet, file_size)

        try:
            # Call Ollama API
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"
                },
                timeout=self.timeout
            )

            if response.status_code != 200:
                fallback["error"] = f"API returned status {response.status_code}"
                return fallback

            # Parse response
            result = response.json()
            response_text = result.get("response", "")

            # Try to parse JSON from response
            try:
                # Sometimes the model returns JSON wrapped in markdown code blocks
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end].strip()
                elif "```" in response_text:
                    json_start = response_text.find("```") + 3
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end].strip()

                classification = json.loads(response_text)
                classification["success"] = True
                return classification

            except json.JSONDecodeError as e:
                fallback["error"] = f"Failed to parse JSON response: {e}"
                fallback["raw_response"] = response_text[:200]  # Include snippet for debugging
                return fallback

        except requests.exceptions.Timeout:
            fallback["error"] = "Request timed out"
            return fallback
        except requests.exceptions.RequestException as e:
            fallback["error"] = f"Request failed: {str(e)}"
            return fallback

    def chat(self, message: str, context: Optional[list] = None) -> str:
        """
        General chat interface with Ollama (for future chat-with-files feature).

        Args:
            message (str): User message
            context (list, optional): Previous conversation context

        Returns:
            str: AI response
        """
        try:
            payload = {
                "model": self.model,
                "prompt": message,
                "stream": False
            }

            if context:
                payload["context"] = context

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 200:
                return response.json().get("response", "No response received")
            else:
                return f"Error: API returned status {response.status_code}"

        except requests.exceptions.RequestException as e:
            return f"Error communicating with Ollama: {str(e)}"

    def pull_model(self, model_name: str) -> bool:
        """
        Pull/download a model from Ollama registry.

        Args:
            model_name (str): Name of the model to pull

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                timeout=300  # Longer timeout for model downloads
            )
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"Error pulling model: {e}")
            return False


# Module-level convenience functions

def create_client(base_url: str = "http://localhost:11434", model: str = "llama3") -> OllamaClient:
    """
    Create an Ollama client instance.

    Args:
        base_url (str): Ollama API endpoint
        model (str): Model name

    Returns:
        OllamaClient: Configured client instance
    """
    return OllamaClient(base_url=base_url, model=model)


def quick_classify(filename: str, extension: str, snippet: str = None) -> Dict[str, Any]:
    """
    Quick classification using default Ollama settings.

    Args:
        filename (str): File name
        extension (str): File extension
        snippet (str, optional): Text content snippet

    Returns:
        Dict: Classification result
    """
    client = create_client()
    return client.classify_file(filename, extension, snippet)


if __name__ == "__main__":
    # Test Ollama client
    client = OllamaClient()

    print("Testing Ollama connection...")
    if client.is_available():
        print("✓ Ollama is available")

        print("\nAvailable models:")
        models = client.list_models()
        for model in models:
            print(f"  - {model}")

        print("\nTesting file classification...")
        result = client.classify_file(
            filename="invoice_march_2025.pdf",
            extension="pdf",
            text_snippet="Invoice #12345\nDate: 2025-03-15\nAmount: $250.00\nClient: Acme Corp"
        )

        print(f"\nClassification result:")
        print(json.dumps(result, indent=2))
    else:
        print("✗ Ollama is not available. Please ensure it's running on http://localhost:11434")
