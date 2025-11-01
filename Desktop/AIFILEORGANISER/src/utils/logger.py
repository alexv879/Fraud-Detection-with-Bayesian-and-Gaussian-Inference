"""
Structured Logging Module

Copyright (c) 2025 Alexandru Emanuel Vasile. All rights reserved.
Proprietary Software - 200-Key Limited Release License

This module provides centralized structured logging for the AI File Organiser.
It uses Python's logging module with rotating file handlers and structured
JSON-like output for better observability and debugging.

NOTICE: This software is proprietary and confidential.
See LICENSE.txt for full terms and conditions.

Author: Alexandru Emanuel Vasile
License: Proprietary (200-key limited release)
"""

import logging
import logging.handlers
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that outputs structured log records.

    Each log record is formatted as JSON-like structured data for easy parsing.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record as structured JSON.

        Args:
            record: Log record to format

        Returns:
            str: Formatted log string
        """
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add any extra fields from the record
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)

        return json.dumps(log_data)


class AgentLogger:
    """
    Wrapper class for agent-specific logging with structured output.

    Provides convenience methods for logging agent operations, model calls,
    validation results, and errors.
    """

    def __init__(self, name: str = 'ai_file_organiser'):
        """
        Initialize the logger.

        Args:
            name: Logger name (default: 'ai_file_organiser')
        """
        self.logger = logging.getLogger(name)

        # Only configure if not already configured
        if not self.logger.handlers:
            self._configure_logger()

    def _configure_logger(self):
        """Configure the logger with rotating file handler and console output."""
        self.logger.setLevel(logging.INFO)

        # Create logs directory
        log_dir = Path(__file__).parent.parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)

        # Rotating file handler (10MB max, 5 backups)
        log_file = log_dir / "organiser.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(StructuredFormatter())

        # Console handler for warnings and errors
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(
            logging.Formatter('%(levelname)s: %(message)s')
        )

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def info(self, message: str, **kwargs):
        """Log info message with optional extra data."""
        self.logger.info(message, extra={'extra_data': kwargs})

    def warning(self, message: str, **kwargs):
        """Log warning message with optional extra data."""
        self.logger.warning(message, extra={'extra_data': kwargs})

    def error(self, message: str, **kwargs):
        """Log error message with optional extra data."""
        self.logger.error(message, extra={'extra_data': kwargs})

    def debug(self, message: str, **kwargs):
        """Log debug message with optional extra data."""
        self.logger.debug(message, extra={'extra_data': kwargs})

    def agent_call(self, model_name: str, prompt_hash: str, file_path: str):
        """Log an agent LLM call."""
        self.info(
            "Agent LLM call initiated",
            model=model_name,
            prompt_hash=prompt_hash,
            file_path=file_path,
            event_type='agent_call'
        )

    def agent_response(self, model_name: str, prompt_hash: str,
                       response_hash: str, parse_success: bool,
                       validation_success: bool):
        """Log an agent LLM response."""
        self.info(
            "Agent LLM response received",
            model=model_name,
            prompt_hash=prompt_hash,
            response_hash=response_hash,
            parse_success=parse_success,
            validation_success=validation_success,
            event_type='agent_response'
        )

    def agent_action(self, action: str, file_path: str, destination: str = None,
                     approved: bool = False, blocked: bool = False,
                     block_reason: str = None):
        """Log an agent-suggested action."""
        self.info(
            f"Agent suggested action: {action}",
            action=action,
            file_path=file_path,
            destination=destination,
            user_approved=approved,
            blocked=blocked,
            block_reason=block_reason,
            event_type='agent_action'
        )

    def model_failure(self, model_name: str, error: str, retry_count: int = 0):
        """Log a model call failure."""
        self.error(
            f"Model call failed: {error}",
            model=model_name,
            error=error,
            retry_count=retry_count,
            event_type='model_failure'
        )

    def validation_failure(self, error: str, raw_response_preview: str = None):
        """Log a validation failure."""
        self.warning(
            f"Validation failed: {error}",
            error=error,
            raw_response_preview=raw_response_preview[:200] if raw_response_preview else None,
            event_type='validation_failure'
        )

    def safety_block(self, file_path: str, reason: str, check_type: str):
        """Log a safety check blocking an action."""
        self.warning(
            f"Safety check blocked action: {reason}",
            file_path=file_path,
            reason=reason,
            check_type=check_type,
            event_type='safety_block'
        )


# Global logger instance
_logger_instance: Optional[AgentLogger] = None


def get_logger(name: str = 'ai_file_organiser') -> AgentLogger:
    """
    Get or create the global logger instance.

    Args:
        name: Logger name

    Returns:
        AgentLogger: Logger instance
    """
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = AgentLogger(name)
    return _logger_instance


if __name__ == "__main__":
    # Test logging
    logger = get_logger()
    logger.info("Test info message", test_field="test_value")
    logger.warning("Test warning")
    logger.error("Test error", error_code=500)
    logger.agent_call("llama3", "abc123", "/path/to/file.txt")
    logger.agent_response("llama3", "abc123", "def456", True, True)
    logger.agent_action("move", "/path/to/file.txt", "/new/path", approved=True)
    print("Logging test complete - check logs/ directory")
