"""
Agent Analyzer Module

Copyright (c) 2025 Alexandru Emanuel Vasile. All rights reserved.
Proprietary Software - 200-Key Limited Release License

This module implements a safe, local AI agent that performs deep multi-step
file analysis using Ollama. The agent returns structured JSON plans for
file organization while enforcing all safety policies and blacklist rules.

NOTICE: This software is proprietary and confidential.
See LICENSE.txt for full terms and conditions.

Author: Alexandru Emanuel Vasile
License: Proprietary (200-key limited release)
"""

import json
import os
import re
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
import jsonschema
from jsonschema import validate, ValidationError

# Import existing components for reuse (HIGH #6 FIX - use TextExtractor instead of FileClassifier)
try:
    from ..core.text_extractor import TextExtractor
    from ..utils.logger import get_logger
except ImportError:
    from core.text_extractor import TextExtractor
    from utils.logger import get_logger


# JSON schema for agent response validation
AGENT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string"},
        "suggested_path": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "rename": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "method": {"type": "string", "enum": ["agent", "rule-based", "ai"]},
        "reason": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "action": {"type": "string", "enum": ["move", "rename", "archive", "delete", "none"]},
        "block_reason": {"anyOf": [{"type": "string"}, {"type": "null"}]}
    },
    "required": ["category", "confidence", "method", "reason", "evidence", "action"]
}


class AgentAnalyzer:
    """
    Local AI agent for deep file analysis using Ollama.

    This agent uses multi-step reasoning to analyze files and produce
    safe, validated JSON plans for organization. All operations respect
    folder policies and path blacklists.

    Attributes:
        config: Configuration object
        ollama_client: Ollama client for local LLM inference
        db_manager: Database manager (optional, for recent paths lookup)
        file_classifier: FileClassifier instance for text extraction
        few_shot_examples: List of few-shot examples for prompting
    """

    def __init__(self, config, ollama_client, db_manager=None):
        """
        Initialize agent analyzer.

        Args:
            config: Configuration object with base_destination, path_blacklist, etc.
            ollama_client: OllamaClient instance for local inference
            db_manager: Optional DatabaseManager for context lookups
        """
        self.config = config
        self.ollama_client = ollama_client
        self.db_manager = db_manager
        # Create a text extractor (HIGH #6 FIX - breaks circular import)
        self.text_extractor = TextExtractor(config)
        # Load few-shot examples
        self.few_shot_examples = self._load_few_shot_examples()
        # Initialize logger
        self.logger = get_logger()

    def _load_few_shot_examples(self, max_examples: int = 3) -> list:
        """
        Load few-shot examples from docs/agent_examples.json.

        Args:
            max_examples: Maximum number of examples to load

        Returns:
            list: List of example dicts (input, output)
        """
        try:
            examples_path = Path(__file__).parent.parent.parent / "docs" / "agent_examples.json"
            if not examples_path.exists():
                return []

            with open(examples_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                examples = data.get('examples', [])
                # Return up to max_examples
                return examples[:max_examples]
        except Exception:
            # If loading fails, continue without examples
            return []

    def analyze_file(self, file_path: str, policy: dict = None,
                     max_snippet_chars: int = 1000) -> Dict[str, Any]:
        """
        Perform deep analysis of a file using the agent.

        This method:
        1. Extracts file metadata and text snippets
        2. Gathers relevant context (policies, recent paths, etc.)
        3. Calls Ollama with a structured prompt
        4. Validates and sanitizes the response
        5. Applies safety checks (blacklist, policy enforcement)

        Args:
            file_path (str): Path to the file to analyze
            policy (dict, optional): Folder policy override
            max_snippet_chars (int): Maximum characters to extract for analysis

        Returns:
            Dict: Classification-like result with agent plan:
                {
                    "category": str,
                    "suggested_path": str | None,
                    "rename": str | None,
                    "confidence": "high"|"medium"|"low",
                    "method": "agent",
                    "reason": str,
                    "evidence": [str, ...],
                    "action": "move"|"rename"|"archive"|"delete"|"none",
                    "block_reason": str | None,
                    "success": bool
                }
        """
        # Check if Ollama is available
        if not self.ollama_client or not self.ollama_client.is_available():
            return {
                'success': False,
                'error': 'Ollama unavailable',
                'category': 'Unsorted',
                'suggested_path': None,
                'rename': None,
                'confidence': 'low',
                'method': 'agent',
                'reason': 'Agent analysis requires Ollama to be running',
                'evidence': [],
                'action': 'none',
                'block_reason': 'Ollama service not available'
            }

        try:
            path = Path(file_path)

            # Validate file exists
            if not path.exists():
                return self._error_response('File not found')

            # Extract file information using text extractor (HIGH #6 FIX)
            file_info = self.text_extractor.extract_file_info(path)

            # Limit text snippet to max_snippet_chars
            text_snippet = file_info.get('text_snippet', '')
            if text_snippet and len(text_snippet) > max_snippet_chars:
                text_snippet = text_snippet[:max_snippet_chars]

            # Get folder policy if not provided
            if policy is None:
                policy = self.config.get_folder_policy(str(path.parent))

            # Get recent paths for context (if db available)
            recent_paths = []
            if self.db_manager:
                try:
                    recent_logs = self.db_manager.get_recent_logs(limit=10)
                    recent_paths = [log.get('new_path', '') for log in recent_logs if log.get('new_path')]
                except Exception:
                    pass  # Not critical

            # Construct agent prompt
            prompt = self._construct_agent_prompt(
                filename=file_info['filename'],
                extension=file_info['extension'],
                size_bytes=file_info['size'],
                text_snippet=text_snippet,
                recent_paths=recent_paths[:5],  # Top 5 for context
                policy=policy,
                base_destination=self.config.base_destination
            )

            # Compute prompt hash for logging
            prompt_hash = hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:16]

            # Log agent call
            self.logger.agent_call(
                model_name=self.ollama_client.model,
                prompt_hash=prompt_hash,
                file_path=str(path)
            )

            # Call Ollama
            response_text = self._call_ollama(prompt)

            # Compute response hash
            response_hash = hashlib.sha256(response_text.encode('utf-8')).hexdigest()[:16]

            # Parse and validate JSON response
            agent_plan = self._parse_and_validate_response(response_text)

            # Log response result
            self.logger.agent_response(
                model_name=self.ollama_client.model,
                prompt_hash=prompt_hash,
                response_hash=response_hash,
                parse_success=agent_plan.get('success', False),
                validation_success=agent_plan.get('success', False)
            )

            if not agent_plan.get('success'):
                self.logger.validation_failure(
                    error=agent_plan.get('error', 'Unknown error'),
                    raw_response_preview=response_text[:200]
                )
                return agent_plan  # Return error response

            # Apply safety checks
            safe_plan = self._apply_safety_checks(
                agent_plan,
                file_path=str(path),
                policy=policy
            )

            # Log action and safety result
            blocked = safe_plan.get('action') == 'none' and safe_plan.get('block_reason')
            self.logger.agent_action(
                action=safe_plan.get('action', 'none'),
                file_path=str(path),
                destination=safe_plan.get('suggested_path'),
                blocked=blocked,
                block_reason=safe_plan.get('block_reason')
            )

            # Log to database if available
            if self.db_manager and safe_plan.get('success'):
                try:
                    # Compute prompt hash for traceability
                    prompt_for_hash = self._construct_agent_prompt(
                        filename=file_info['filename'],
                        extension=file_info['extension'],
                        size_bytes=file_info['size'],
                        text_snippet=text_snippet,
                        recent_paths=[],
                        policy=policy,
                        base_destination=self.config.base_destination
                    )
                    prompt_hash = hashlib.sha256(prompt_for_hash.encode('utf-8')).hexdigest()[:16]

                    # Store evidence, plan, and raw response in logs
                    self.db_manager.log_action(
                        filename=file_info['filename'],
                        old_path=str(path),
                        new_path=safe_plan.get('suggested_path'),
                        operation='agent_suggestion',
                        time_saved=0,  # Not executed yet
                        category=safe_plan.get('category'),
                        ai_suggested=True,
                        user_approved=False,
                        raw_response=safe_plan.get('raw_response'),
                        model_name=self.ollama_client.model,
                        prompt_hash=prompt_hash
                    )
                except Exception:
                    pass  # Logging failure shouldn't break agent

            return safe_plan

        except Exception as e:
            return self._error_response(f'Agent analysis error: {str(e)}')

    def _construct_agent_prompt(self, filename: str, extension: str, size_bytes: int,
                                text_snippet: str, recent_paths: list, policy: dict,
                                base_destination: str) -> str:
        """
        Construct the agent prompt with all context.

        Args:
            filename: File name
            extension: File extension
            size_bytes: File size in bytes
            text_snippet: Extracted text content
            recent_paths: List of recent organized file paths
            policy: Folder policy dict or None
            base_destination: Base destination directory

        Returns:
            str: Complete prompt for Ollama
        """
        # System instructions
        system_msg = (
            "You are an on-device file organization assistant. All processing is local. "
            "Follow strict safety rules: do not propose any operations that would move files "
            "under blacklisted or system/program directories. Return ONLY a single JSON object "
            "matching the schema provided. No extra text."
        )

        # Format policy for display
        policy_str = "null"
        if policy:
            policy_str = json.dumps(policy)

        # Format recent paths
        recent_str = "[]"
        if recent_paths:
            recent_str = json.dumps(recent_paths[:5])

        # Format snippet
        snippet_display = text_snippet if text_snippet else "(no text content available)"

        # Build few-shot examples section
        few_shot_section = ""
        if self.few_shot_examples:
            few_shot_section = "\n\nHere are some examples of correct JSON output:\n\n"
            for i, example in enumerate(self.few_shot_examples, 1):
                inp = example.get('input', {})
                out = example.get('output', {})
                few_shot_section += f"Example {i}:\n"
                few_shot_section += f"Input: filename={inp.get('filename')}, extension={inp.get('extension')}\n"
                few_shot_section += f"Output: {json.dumps(out, indent=2)}\n\n"

        # User message with all metadata
        user_msg = f"""You will receive:

filename: {filename}
extension: {extension}
size_bytes: {size_bytes}
snippet: {snippet_display}
recent_paths: {recent_str}
folder_policy: {policy_str}
base_destination: {base_destination}

Analyze this file and return a JSON object (only JSON) adhering to the schema. Important rules:

1. If the file looks like program/game/system file (executables, many small files, binary blobs with little text), set action='none' and set block_reason.
2. If folder_policy.allow_move == false, set action='none' and block_reason='folder policy disallows moves'.
3. For dated documents (invoices, statements), prefer date-based subfolders (YYYY or YYYY/MM).
4. If suggested_path is relative, make it concise and relative to base_destination; do not invent absolute paths outside base_destination.
5. Provide short evidence strings (1-2 line snippets) that justify the classification.
6. Return only JSON and ensure it conforms to this schema:

{{
  "category": "string (e.g., Documents, Finance, Projects)",
  "suggested_path": "string or null (relative to base_destination)",
  "rename": "string or null (improved filename if needed)",
  "confidence": "high" | "medium" | "low",
  "method": "agent",
  "reason": "string (brief explanation)",
  "evidence": ["array", "of", "strings"],
  "action": "move" | "rename" | "archive" | "delete" | "none",
  "block_reason": "string or null (if action blocked)"
}}{few_shot_section}

Provide your JSON classification:"""

        # Combine system + user (Ollama generate takes a single prompt)
        full_prompt = f"{system_msg}\n\n{user_msg}"

        return full_prompt

    def _call_ollama(self, prompt: str) -> str:
        """
        Call Ollama generate API with the prompt and proper timeout handling.

        Args:
            prompt: Full prompt text

        Returns:
            str: Response text from Ollama
        """
        import requests

        # Ensure timeout is set (CRITICAL FIX #5)
        timeout = getattr(self.ollama_client, 'timeout', 30)
        if timeout is None or timeout <= 0:
            timeout = 30  # Default 30 seconds

        try:
            response = requests.post(
                f"{self.ollama_client.base_url}/api/generate",
                json={
                    "model": self.ollama_client.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"  # Request JSON format
                },
                timeout=timeout
            )

            if response.status_code != 200:
                raise Exception(f"Ollama API returned status {response.status_code}")

            result = response.json()
            return result.get("response", "")

        except requests.exceptions.Timeout:
            # Specific timeout exception (CRITICAL FIX #5)
            raise Exception(f"Ollama request timed out after {timeout} seconds. Try increasing timeout or using a faster model.")
        except requests.exceptions.ConnectionError as e:
            # Specific connection exception (CRITICAL FIX #5)
            raise Exception(f"Cannot connect to Ollama at {self.ollama_client.base_url}. Is Ollama running?")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama request failed: {str(e)}")

    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """
        Extract JSON object from text using multiple strategies.

        Tries:
        1. Markdown code fences (```json ... ```)
        2. Regex to find first {...} block
        3. Plain text (assumes entire text is JSON)

        Args:
            text: Raw text potentially containing JSON

        Returns:
            str or None: Extracted JSON string or None if extraction fails
        """
        cleaned_text = text.strip()

        # Strategy 1: Extract from markdown code blocks
        if "```json" in cleaned_text:
            json_start = cleaned_text.find("```json") + 7
            json_end = cleaned_text.find("```", json_start)
            if json_end > json_start:
                return cleaned_text[json_start:json_end].strip()
        elif "```" in cleaned_text:
            json_start = cleaned_text.find("```") + 3
            json_end = cleaned_text.find("```", json_start)
            if json_end > json_start:
                return cleaned_text[json_start:json_end].strip()

        # Strategy 2: Regex to extract first {...} block (handles multi-line)
        import re
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        match = re.search(json_pattern, cleaned_text, re.DOTALL)
        if match:
            return match.group(0)

        # Strategy 3: Assume entire text is JSON
        if cleaned_text.startswith('{') and cleaned_text.endswith('}'):
            return cleaned_text

        return None

    def _parse_and_validate_response(self, response_text: str, raw_response_for_log: str = None) -> Dict[str, Any]:
        """
        Parse Ollama response and validate against schema with retry logic.

        Implements robust JSON extraction with fallback strategies:
        1. Try direct parsing with code fence extraction
        2. If parse fails, extract first {...} block with regex
        3. If validation fails, attempt one retry with a parsing-only prompt

        Args:
            response_text: Raw response from Ollama
            raw_response_for_log: Original raw response to log (optional)

        Returns:
            Dict: Parsed and validated agent plan (with success=True) or error
        """
        # Store raw response for debugging (will be logged to DB later)
        if raw_response_for_log is None:
            raw_response_for_log = response_text

        max_retries = 1
        attempt = 0

        while attempt <= max_retries:
            try:
                # Extract JSON using multiple strategies
                extracted_json = self._extract_json_from_text(response_text)

                if not extracted_json:
                    if attempt < max_retries:
                        # Retry with a parsing-only prompt
                        retry_prompt = f"""The following text should be a JSON object but failed to parse.
Extract and return ONLY the valid JSON object, with no additional text:

{response_text[:1000]}

Return only valid JSON matching this schema:
{json.dumps(AGENT_RESPONSE_SCHEMA, indent=2)}"""

                        try:
                            self.logger.warning(
                                "JSON extraction failed, retrying with parsing-only prompt",
                                attempt=attempt,
                                event_type='parse_retry'
                            )
                            response_text = self._call_ollama(retry_prompt)
                            attempt += 1
                            continue
                        except Exception as retry_err:
                            self.logger.model_failure(
                                model_name=self.ollama_client.model,
                                error=f"JSON extraction retry failed: {str(retry_err)}",
                                retry_count=attempt
                            )
                            return self._error_response(
                                f'JSON extraction failed and retry failed',
                                raw_response=raw_response_for_log
                            )
                    else:
                        return self._error_response(
                            'Failed to extract JSON from response',
                            raw_response=raw_response_for_log
                        )

                # Parse JSON
                try:
                    plan = json.loads(extracted_json)
                except json.JSONDecodeError as e:
                    if attempt < max_retries:
                        # One retry for parse errors
                        retry_prompt = f"""Fix this malformed JSON and return only valid JSON:

{extracted_json}

Return only valid JSON matching this schema:
{json.dumps(AGENT_RESPONSE_SCHEMA, indent=2)}"""

                        try:
                            response_text = self._call_ollama(retry_prompt)
                            attempt += 1
                            continue
                        except Exception:
                            return self._error_response(
                                f'JSON parse error: {str(e)} (retry failed)',
                                raw_response=raw_response_for_log
                            )
                    else:
                        return self._error_response(
                            f'JSON parse error: {str(e)}',
                            raw_response=raw_response_for_log
                        )

                # Validate against schema
                try:
                    validate(instance=plan, schema=AGENT_RESPONSE_SCHEMA)
                except ValidationError as e:
                    return self._error_response(
                        f'Schema validation failed: {e.message}',
                        raw_response=raw_response_for_log
                    )

                # Path safety sanitization
                suggested_path = plan.get('suggested_path')
                if suggested_path:
                    # Sanitize path: remove dangerous characters, check for absolute paths
                    sanitized = self._sanitize_path(suggested_path)
                    if sanitized != suggested_path:
                        plan['suggested_path'] = sanitized

                # Ensure method is 'agent' and mark success
                plan['method'] = 'agent'
                plan['success'] = True
                plan['raw_response'] = raw_response_for_log[:2000]  # Truncate for storage

                return plan

            except Exception as e:
                if attempt < max_retries:
                    attempt += 1
                    continue
                return self._error_response(
                    f'Response parsing error: {str(e)}',
                    raw_response=raw_response_for_log
                )

    def _check_source_blacklist(self, file_path: str) -> tuple:
        """
        Check if source file is in blacklist (CRITICAL FIX #4).

        Args:
            file_path: Path to check

        Returns:
            Tuple of (is_blacklisted, reason)
        """
        try:
            source_resolved = str(Path(file_path).resolve())
            blacklist = getattr(self.config, 'path_blacklist', []) or []

            for blacklisted in blacklist:
                try:
                    blacklisted_resolved = str(Path(blacklisted).expanduser().resolve())
                    if os.path.commonpath([source_resolved, blacklisted_resolved]) == blacklisted_resolved:
                        return True, f'source file is blacklisted: {blacklisted}'
                except (ValueError, OSError):
                    # Different drives on Windows - check prefix
                    if source_resolved.lower().startswith(blacklisted_resolved.lower()):
                        return True, f'source file is blacklisted: {blacklisted}'
        except Exception:
            pass

        return False, ''

    def _apply_safety_checks(self, plan: Dict[str, Any], file_path: str,
                            policy: dict = None) -> Dict[str, Any]:
        """
        Apply safety checks to the agent plan.

        Validates:
        1. Source file not in blacklist (CRITICAL FIX #4)
        2. Suggested paths are not in blacklist
        3. Folder policy allow_move is respected
        4. Paths don't target system/program directories

        Args:
            plan: Agent plan dict
            file_path: Original file path
            policy: Folder policy dict or None

        Returns:
            Dict: Plan with action='none' and block_reason set if unsafe
        """
        # Check if SOURCE file is in blacklist (CRITICAL FIX #4)
        is_blacklisted, reason = self._check_source_blacklist(file_path)
        if is_blacklisted:
            self.logger.safety_block(
                file_path=file_path,
                reason=reason,
                check_type='source_blacklist'
            )
            plan['action'] = 'none'
            plan['block_reason'] = reason
            return plan

        # Check folder policy allow_move
        if policy and policy.get('allow_move') is False:
            self.logger.safety_block(
                file_path=file_path,
                reason='folder policy disallows moves',
                check_type='folder_policy'
            )
            plan['action'] = 'none'
            plan['block_reason'] = 'folder policy disallows moves'
            return plan

        # Check suggested_path safety
        suggested_path = plan.get('suggested_path')
        if suggested_path:
            # Resolve path (handle relative paths)
            try:
                base_dest = Path(self.config.base_destination)

                # If relative, join with base_destination
                suggested_obj = Path(suggested_path)
                if not suggested_obj.is_absolute():
                    resolved_path = base_dest / suggested_obj
                else:
                    resolved_path = suggested_obj

                resolved_str = str(resolved_path.resolve())

                # Check against blacklist
                blacklist = getattr(self.config, 'path_blacklist', []) or []
                for blacklisted in blacklist:
                    try:
                        blacklisted_resolved = str(Path(blacklisted).expanduser().resolve())

                        # Check if resolved_path is under blacklisted path
                        try:
                            if os.path.commonpath([resolved_str, blacklisted_resolved]) == blacklisted_resolved:
                                plan['action'] = 'none'
                                plan['block_reason'] = f'destination is blacklisted: {blacklisted}'
                                return plan
                        except ValueError:
                            # Different drives on Windows - check prefix
                            if resolved_str.lower().startswith(blacklisted_resolved.lower()):
                                plan['action'] = 'none'
                                plan['block_reason'] = f'destination is blacklisted: {blacklisted}'
                                return plan
                    except Exception:
                        # Fallback: simple prefix check
                        if resolved_str.lower().startswith(str(Path(blacklisted).expanduser()).lower()):
                            plan['action'] = 'none'
                            plan['block_reason'] = f'destination is blacklisted: {blacklisted}'
                            return plan

                # Check for common system/program directories (heuristic)
                dangerous_patterns = [
                    r'[/\\](windows|program files|system32|winnt)',
                    r'[/\\](bin|sbin|usr|etc|var|opt)[/\\]',
                    r'[/\\]\.exe$',
                    r'[/\\](node_modules|\.git|\.venv|venv)[/\\]'
                ]

                for pattern in dangerous_patterns:
                    if re.search(pattern, resolved_str, re.IGNORECASE):
                        plan['action'] = 'none'
                        plan['block_reason'] = 'destination appears to be system/program directory'
                        return plan

            except Exception as e:
                # Path resolution failed - block for safety
                plan['action'] = 'none'
                plan['block_reason'] = f'path validation error: {str(e)}'
                return plan

        # All checks passed
        if not plan.get('block_reason'):
            plan['block_reason'] = None

        return plan

    def _sanitize_path(self, path: str) -> str:
        """
        Sanitize a suggested path by removing dangerous characters and patterns.

        Args:
            path: Path string to sanitize

        Returns:
            str: Sanitized path
        """
        if not path:
            return path

        # Remove null bytes and control characters
        sanitized = path.replace('\x00', '').replace('\r', '').replace('\n', '')

        # Remove potentially dangerous path traversal patterns
        sanitized = sanitized.replace('..', '').replace('~/', '').replace('~\\', '')

        # Remove leading/trailing whitespace and path separators
        sanitized = sanitized.strip().strip('/\\')

        return sanitized

    def _error_response(self, error_msg: str, raw_response: str = None) -> Dict[str, Any]:
        """
        Create a standardized error response.

        Args:
            error_msg: Error message
            raw_response: Optional raw response text for debugging

        Returns:
            Dict: Error response matching expected schema
        """
        response = {
            'success': False,
            'error': error_msg,
            'category': 'Unsorted',
            'suggested_path': None,
            'rename': None,
            'confidence': 'low',
            'method': 'agent',
            'reason': f'Agent error: {error_msg}',
            'evidence': [],
            'action': 'none',
            'block_reason': error_msg
        }

        if raw_response:
            response['raw_response'] = raw_response[:2000]  # Truncate for storage

        return response


def analyze_file(file_path: str, config, ollama_client, db_manager=None,
                policy: dict = None) -> Dict[str, Any]:
    """
    Convenience function to analyze a single file with the agent.

    Args:
        file_path (str): Path to file
        config: Configuration object
        ollama_client: OllamaClient instance
        db_manager: Optional DatabaseManager
        policy: Optional folder policy override

    Returns:
        Dict: Agent analysis result
    """
    analyzer = AgentAnalyzer(config, ollama_client, db_manager)
    return analyzer.analyze_file(file_path, policy=policy)


if __name__ == "__main__":
    # Test agent analyzer
    import sys
    from pathlib import Path

    # Add parent dirs to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.config import get_config
    from src.ai.ollama_client import OllamaClient
    from src.core.db_manager import DatabaseManager

    config = get_config()
    ollama = OllamaClient(config.ollama_base_url, config.ollama_model)
    db = DatabaseManager()

    if not ollama.is_available():
        print("Error: Ollama is not available. Please start Ollama service.")
        sys.exit(1)

    print("Testing AgentAnalyzer...")
    print(f"Using model: {ollama.model}")

    # Create a test file
    test_file = Path("test_invoice_2025.txt")
    test_file.write_text("Invoice #12345\nDate: 2025-03-15\nAmount: $250.00\nClient: Acme Corp")

    try:
        analyzer = AgentAnalyzer(config, ollama, db)
        result = analyzer.analyze_file(str(test_file))

        print("\nAgent Analysis Result:")
        print(json.dumps(result, indent=2))

    finally:
        # Clean up test file
        if test_file.exists():
            test_file.unlink()
