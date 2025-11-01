"""
Agent Module

This module provides agentic AI capabilities for deep file analysis.
The agent performs multi-step reasoning using local Ollama models to
suggest classification and organization actions with safety guardrails.

Author: AI File Organiser Team
License: Proprietary (200-key limited release)
"""

from .agent_analyzer import AgentAnalyzer

__all__ = ['AgentAnalyzer']
