# 🤖 Private AI File Organiser

> Your local-first, privacy-respecting file organization assistant powered by AI

The Private AI File Organiser automatically sorts, renames, and archives files on your computer using on-device AI. **No cloud. No tracking. Just intelligent file management.**

## ✨ Features

- **🔐 Privacy-First**: All processing happens locally. Your files never leave your computer.
- **🧠 AI-Powered**: Uses Ollama for intelligent file classification and organization
- **📊 Time Tracking**: See exactly how much time you've saved
- **🔍 Duplicate Detection**: Find and remove duplicate files
- **🎯 Smart Classification**: Rule-based + AI hybrid approach
- **📈 Web Dashboard**: Clean, modern interface for monitoring and control
- **⏸️ Dry Run Mode**: Preview actions before they happen
- **↩️ Undo Support**: Reverse accidental actions
- **🔑 License System**: 200 limited early access keys

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+**
2. **Ollama** (for AI classification)
   - Download from: https://ollama.ai
   - Pull a model: `ollama pull llama3`

### Installation

```bash
# Clone the repository
git clone https://github.com/yourproject/ai-file-organiser.git
cd ai-file-organiser

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
python src/main.py dashboard
```

Open your browser to: http://localhost:5000

### First-Time Setup

1. **Activate License**: Enter your license key in the dashboard or via CLI:
   ```bash
   python src/main.py --activate XXXX-XXXX-XXXX-XXXX
   ```

2. **Configure Folders**: Edit `config.json` to set which folders to watch

3. **Start Organizing**: Use the dashboard or CLI commands

## 📖 Usage

### Web Dashboard (Recommended)

```bash
python src/main.py dashboard
```

The dashboard provides:
- 📥 **Inbox**: Review pending file classifications
- 📊 **History**: See all past actions
- 🔍 **Duplicates**: Find and remove duplicates
- ⚙️ **Settings**: Configure behavior
- 🔐 **License**: Manage activation

### CLI Commands

```bash
# Watch folders for new files (auto-organize)
python src/main.py watch

# Scan existing files once
python src/main.py scan

# Find duplicate files
python src/main.py duplicates

# Show statistics
python src/main.py stats

# Check license status
python src/main.py license
```

## ⚙️ Configuration

Edit `config.json` to customize:

```json
{
  "watched_folders": ["~/Desktop", "~/Downloads"],
  "auto_mode": false,  // Auto-organize without approval
  "dry_run": true,     // Simulate actions (safe testing)
  "ollama_model": "llama3",
  "base_destination": "~/Documents",  // Base path for organized files
  "path_blacklist": [                 // Never process these paths
    "C:/Windows",
    "C:/Program Files",
    "~/.ssh"
  ],
  "folder_policies": {                // Per-folder overrides
    "~/Downloads": {
      "allow_move": false,           // Block moves from this folder
      "auto_mode": false
    },
    "~/Desktop": {
      "allow_move": true,
      "auto_mode": true,
      "use_ai": true
    }
  },
  "destination_rules": {
    "pdf": "Documents/PDFs/",
    "jpg": "Pictures/",
    // Add your own rules...
  }
}
```

### Configuration Keys

- **watched_folders**: Directories to monitor for new files
- **base_destination**: Root directory for organized files (default: home directory)
- **path_blacklist**: Paths that must never be processed or moved (safety)
- **folder_policies**: Per-folder overrides for auto_mode, allow_move, use_ai
- **auto_mode**: Automatically execute actions without user approval
- **dry_run**: Simulate all actions without actually moving files (recommended for testing)
- **ollama_model**: Local LLM model to use (llama3, llama2, etc.)
- **destination_rules**: File extension → destination path mapping

## 🧠 How It Works

### Hybrid Classification System

1. **Stage 1: Rule-Based** (Fast)
   - File extension mapping
   - Filename pattern matching
   - High confidence = immediate classification

2. **Stage 2: AI-Powered** (Smart)
   - Semantic understanding via Ollama
   - Content analysis (text extraction)
   - Context-aware suggestions

3. **Stage 3: Deep Agent Analysis** (Advanced - NEW!)
   - Multi-step reasoning with local LLM
   - Evidence-based classification with justification
   - Policy-aware planning (respects blacklists & folder rules)
   - Returns structured JSON with confidence scores
   - Triggered automatically for low-confidence files or via "Deep Analyze" button

### File Processing Pipeline

```
File Detected → Classify → Suggest Action → User Approval → Execute → Log
                    ↓
            (if low confidence or requested)
                    ↓
            Deep Agent Analysis → Evidence + Safe Plan → User Review
```

### Time Saved Calculation

Each action estimates time saved based on:
- Move: 0.5 minutes
- Rename: 0.3 minutes
- Delete: 0.2 minutes
- Archive: 0.4 minutes

## 📊 Project Structure

```
AIFILEORGANISER/
├── config.json              # Configuration
├── requirements.txt         # Dependencies
├── README.md               # Documentation
│
├── src/
│   ├── main.py            # Entry point
│   ├── config.py          # Config management
│   │
│   ├── core/              # Core functionality
│   │   ├── db_manager.py        # SQLite database
│   │   ├── classifier.py        # File classification
│   │   ├── watcher.py           # Folder monitoring
│   │   ├── actions.py           # File operations
│   │   └── duplicates.py        # Duplicate detection
│   │
│   ├── ai/                # AI integration
│   │   ├── ollama_client.py     # Ollama API client
│   │   └── prompts/             # AI prompts
│   │
│   ├── agent/             # Deep analysis agent (NEW!)
│   │   ├── agent_analyzer.py    # Multi-step reasoning agent
│   │   └── README.md            # Agent documentation
│   │
│   ├── license/           # License system
│   │   ├── validator.py         # License validation
│   │   └── api_mock.py          # Mock API server
│   │
│   └── ui/                # User interface
│       └── dashboard.py         # FastAPI web UI
│
├── tools/                 # Development tools
│   └── test_agent.py      # Agent test harness
│
├── docs/                  # Documentation
│   └── agent_prompt.txt   # Agent prompt template
│
└── data/                  # Data storage (auto-created)
    ├── database/          # SQLite databases
    └── logs/              # Application logs
```

## 🔑 License System

**Limited Release**: 200 licenses available

### Features

- ✅ 30-day validity from activation
- ✅ Offline validation support
- ✅ Single activation per key
- ✅ Renewal available on website

### Getting a License

1. Visit: [Your Website URL]
2. Request early access
3. Receive license key via email
4. Activate in application

### API Endpoints (for server deployment)

```
POST /api/verify-license
{
  "key": "XXXX-XXXX-XXXX-XXXX"
}
```

## 🛠️ Development

### Running Tests

```bash
# Install dev dependencies
pip install pytest pytest-cov

# Run tests
pytest

# With coverage
pytest --cov=src
```

### Database Schema

**files_log**: Operation history
**duplicates**: Duplicate file tracking
**license**: License activation status
**stats**: Aggregated statistics

### Adding New Classification Rules

Edit `config.json`:

```json
{
  "destination_rules": {
    "newext": "Path/To/Destination/"
  }
}
```

Or modify `src/core/classifier.py` for pattern-based rules.

## 🤝 Contributing

This is a proprietary limited release. Contributions accepted from license holders only.

## 📝 Roadmap

- [x] Core file organization
- [x] AI classification
- [x] Duplicate detection
- [x] Web dashboard
- [x] License system
- [ ] Chat-with-files interface
- [ ] Machine learning preference adaptation
- [ ] Multi-user team edition
- [ ] Cloud backup integration (optional)

## ⚠️ Privacy & Security

**Your Privacy Matters**

- ✅ All processing is local
- ✅ No cloud uploads
- ✅ No tracking or telemetry
- ✅ Open source (license holders)
- ✅ Encrypted local storage

**Security Features**

- Local database encryption
- Secure license validation
- No sensitive data transmission
- Audit logs for all operations

## 🧪 Deep Analysis Agent Feature (NEW!)

The agent provides advanced, multi-step reasoning for complex file organization decisions.

### When to Use Deep Analysis

- Click "🔍 Deep Analyze" button in dashboard for any pending file
- Automatically triggered for files with low rule-based confidence
- Best for: invoices, receipts, contracts, dated documents

### What You Get

```json
{
  "category": "Finance",
  "suggested_path": "Documents/Finance/Invoices/2025/03/",
  "confidence": "high",
  "reason": "Invoice with date, amount, and client information",
  "evidence": [
    "Invoice #12345",
    "Date: 2025-03-15",
    "Amount: $250.00"
  ],
  "action": "move",
  "block_reason": null
}
```

### Safety Features

- **Non-destructive**: Agent only suggests; requires approval to execute
- **Policy-aware**: Respects `folder_policies` (e.g., `allow_move: false`)
- **Blacklist protection**: Never suggests moves to system/blacklisted paths
- **JSON validation**: Strict schema ensures no malformed output
- **Evidence-based**: Shows reasoning for transparency

### Configuration

Recommended models for agent analysis:

```bash
# Best (if you have 16GB+ RAM)
ollama pull llama3

# Good (8-16GB RAM)
ollama pull llama3:8b

# Lighter (4-8GB RAM)
ollama pull llama2:7b
```

Update `config.json`:
```json
{
  "ollama_model": "llama3",
  "classification": {
    "enable_ai": true,
    "text_extract_limit": 1000
  }
}
```

### Testing the Agent

Run the test harness to validate all safety features:

```bash
python tools/test_agent.py
```

Tests include:
- ✅ JSON schema compliance
- ✅ Policy enforcement (allow_move=false)
- ✅ Blacklist path blocking
- ✅ Evidence quality
- ✅ Confidence levels
- ✅ Error handling
- ✅ Non-destructive analysis

### Agent API

For programmatic access:

```python
from src.agent.agent_analyzer import AgentAnalyzer
from src.config import get_config
from src.ai.ollama_client import OllamaClient

config = get_config()
ollama = OllamaClient(config.ollama_base_url, config.ollama_model)
analyzer = AgentAnalyzer(config, ollama)

result = analyzer.analyze_file("invoice.pdf")
print(result['category'], result['confidence'], result['evidence'])
```

Dashboard API endpoint:
```javascript
POST /api/files/deep-analyze
{
  "file_path": "/path/to/file.pdf"
}
```

See `src/agent/README.md` for full documentation.

## 💡 Tips & Best Practices

1. **Start with Dry Run**: Test with `dry_run: true` first
2. **Review Suggestions**: Use manual approval initially
3. **Configure Blacklist**: Add system paths to `path_blacklist` for safety
4. **Set Folder Policies**: Protect critical folders with `allow_move: false`
5. **Try Deep Analyze**: Use for unclear files to get detailed reasoning
6. **Backup Important Files**: Before bulk operations
7. **Customize Rules**: Tailor destination rules to your workflow
8. **Monitor Stats**: Check time saved to stay motivated

## 🐛 Troubleshooting

### Ollama Not Available

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve

# Pull a model
ollama pull llama3
```

### License Issues

```bash
# Check license status
python src/main.py license

# Re-activate
python src/main.py --activate YOUR-LICENSE-KEY
```

### Database Corruption

```bash
# Backup database
cp data/database/organiser.db data/database/organiser.db.backup

# Reset database (warning: loses history)
rm data/database/organiser.db
python src/main.py stats  # Recreates database
```

## 📧 Support

- **Documentation**: [Your Docs URL]
- **Issues**: [GitHub Issues]
- **Email**: support@yourproject.com
- **Community**: [Discord/Forum]

## 📜 License

Proprietary - 200-Key Limited Release
© 2025 AI File Organiser Team

---

**Made with ❤️ for people who value their privacy and hate messy folders**
