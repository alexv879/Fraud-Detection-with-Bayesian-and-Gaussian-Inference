# Agent Module - Deep File Analysis

## Overview

The Agent module provides **local, safe, agentic deep analysis** for file organization. It uses Ollama (local LLM inference) to perform multi-step reasoning about files and generate structured, validated classification plans.

## Key Features

- **Local Processing**: All inference runs on-device via Ollama HTTP API
- **Safety First**: Enforces path blacklists and folder policies
- **Non-Destructive**: Returns suggestions only; execution requires approval
- **Structured Output**: Returns strictly validated JSON matching a fixed schema
- **Evidence-Based**: Provides reasoning and evidence for all decisions
- **Policy-Aware**: Respects per-folder `allow_move`, `auto_mode`, and other policies

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AgentAnalyzer                            │
├─────────────────────────────────────────────────────────────┤
│  1. Extract file metadata (name, extension, size, text)    │
│  2. Gather context (folder policy, recent paths)           │
│  3. Call Ollama with structured prompt                     │
│  4. Parse and validate JSON response                       │
│  5. Apply safety checks (blacklist, policy)                │
│  6. Return safe, validated plan                            │
└─────────────────────────────────────────────────────────────┘
```

## Components

### `AgentAnalyzer`

Main class that orchestrates deep analysis:

```python
from agent.agent_analyzer import AgentAnalyzer

analyzer = AgentAnalyzer(config, ollama_client, db_manager)
result = analyzer.analyze_file(file_path, policy=None)
```

**Methods:**

- `analyze_file(file_path, policy=None, max_snippet_chars=1000)` → Dict
  - Performs deep analysis and returns validated plan
  - Returns dict with: category, suggested_path, rename, confidence, method, reason, evidence, action, block_reason

### JSON Schema

All agent responses are validated against this schema:

```json
{
  "category": "string",
  "suggested_path": "string | null",
  "rename": "string | null",
  "confidence": "high" | "medium" | "low",
  "method": "agent",
  "reason": "string",
  "evidence": ["array of strings"],
  "action": "move" | "rename" | "archive" | "delete" | "none",
  "block_reason": "string | null"
}
```

## Safety Mechanisms

### 1. Path Blacklist

Files under blacklisted paths are never processed:

```python
# config.json
{
  "path_blacklist": [
    "C:/Windows",
    "C:/Program Files",
    "~/.ssh"
  ]
}
```

If `suggested_path` resolves to a blacklisted location, `action` is set to `'none'` and `block_reason` explains why.

### 2. Folder Policies

Per-folder overrides for behavior:

```python
# config.json
{
  "folder_policies": {
    "~/Downloads": {
      "allow_move": false,
      "auto_mode": false
    },
    "~/Desktop": {
      "allow_move": true,
      "auto_mode": true
    }
  }
}
```

If `allow_move: false`, agent sets `action='none'` and provides block_reason.

### 3. System Directory Heuristics

Agent blocks moves to common system/program directories using regex patterns:
- `/windows/`, `/program files/`, `/system32/`
- `/bin/`, `/sbin/`, `/usr/`, `/etc/`
- `node_modules`, `.git`, `.venv`

### 4. JSON Validation

Every response is validated against the schema. Invalid responses are rejected and return an error plan.

## Integration

### With Classifier

The classifier calls the agent when `deep_analysis=True` or when rule-based confidence is low:

```python
from core.classifier import FileClassifier

classifier = FileClassifier(config, ollama_client)
result = classifier.classify(file_path, deep_analysis=True)
# result['method'] will be 'agent' if agent was used
```

### With Dashboard API

The dashboard exposes a `/api/files/deep-analyze` endpoint:

```javascript
fetch('/api/files/deep-analyze', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ file_path: '/path/to/file.txt' })
})
.then(response => response.json())
.then(plan => {
  console.log('Category:', plan.category);
  console.log('Evidence:', plan.evidence);
  console.log('Action:', plan.action);
});
```

## Configuration

### Required Config Keys

```json
{
  "ollama_base_url": "http://localhost:11434",
  "ollama_model": "llama3",
  "base_destination": "~/Documents",
  "path_blacklist": [],
  "folder_policies": {}
}
```

### Recommended Models

- **Llama 3 13B**: Best reasoning for on-device (if you have RAM)
- **Llama 3 8B**: Good balance of speed and quality
- **Llama 2 7B**: Lighter option for constrained machines

Install with:
```bash
ollama pull llama3
```

## Usage Examples

### Example 1: Basic Analysis

```python
from agent.agent_analyzer import AgentAnalyzer, analyze_file
from config import get_config
from ai.ollama_client import OllamaClient

config = get_config()
ollama = OllamaClient(config.ollama_base_url, config.ollama_model)

result = analyze_file("invoice_2025.pdf", config, ollama)
print(result['category'])         # "Finance"
print(result['suggested_path'])   # "Documents/Finance/Invoices/2025/"
print(result['confidence'])       # "high"
print(result['evidence'])         # ["Invoice #12345", "Date: 2025-03-15", ...]
```

### Example 2: Policy Override

```python
analyzer = AgentAnalyzer(config, ollama, db_manager=None)

# Block moves for this analysis
policy = {"allow_move": False}
result = analyzer.analyze_file("/tmp/file.txt", policy=policy)

assert result['action'] == 'none'
assert 'policy' in result['block_reason'].lower()
```

### Example 3: Handle Unavailable Ollama

```python
result = analyzer.analyze_file("file.txt")

if not result.get('success'):
    print(f"Analysis failed: {result['error']}")
    # Falls back to rule-based classification
```

## Testing

Run the test harness:

```bash
python tools/test_agent.py
```

Tests cover:
1. JSON schema validation
2. Policy enforcement (allow_move=false)
3. Path blacklist enforcement
4. Evidence and reasoning quality
5. Confidence level validity
6. Error handling (missing files)
7. Non-destructive analysis (dry-run)

## Troubleshooting

### "Ollama unavailable"

Ensure Ollama is running:
```bash
ollama serve
```

Check availability:
```bash
curl http://localhost:11434/api/tags
```

### "Schema validation failed"

The model's response didn't match the JSON schema. Try:
- Use a more capable model (llama3 vs llama2)
- Check model is instruction-tuned
- Increase `timeout` in OllamaClient if model is slow

### "Path is blacklisted"

The suggested path fell under a blacklisted directory. Check `config.json` `path_blacklist` and ensure the agent isn't suggesting system paths.

### Agent returns low confidence

If the agent consistently returns low confidence:
- Provide more context (increase `text_extract_limit` in config)
- Use a larger model
- Check the file has meaningful content to analyze

## Architecture Decisions

### Why Local (Ollama)?

- **Privacy**: Files never leave your machine
- **Cost**: No API fees
- **Speed**: Low latency for local inference
- **Offline**: Works without internet

### Why Strict JSON Schema?

- **Safety**: No arbitrary code execution
- **Predictability**: Consistent output format
- **Validation**: Catch errors before execution
- **Integration**: Easy to use in downstream code

### Why Agent vs Simple Classifier?

- **Multi-step reasoning**: Agent can think through complex cases
- **Evidence**: Provides justification for decisions
- **Context-aware**: Uses folder policies, recent patterns
- **Date extraction**: Better at parsing invoice dates, etc.

## Future Enhancements

Potential improvements (not implemented):

1. **Multi-file batching**: Analyze multiple files in one call
2. **Learning from user approvals**: Track which suggestions get accepted
3. **Cross-drive safety**: Enhanced copy-verify-delete for cross-drive moves
4. **Embeddings search**: Use recent file embeddings for context
5. **Custom prompts**: Per-folder custom classification prompts

## License

Proprietary - AI File Organiser Team
200-key limited release
