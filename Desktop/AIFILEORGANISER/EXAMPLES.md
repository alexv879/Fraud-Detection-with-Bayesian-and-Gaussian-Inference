# Usage Examples

This document provides practical examples of using the Private AI File Organiser.

## 📋 Table of Contents
1. [Basic Usage](#basic-usage)
2. [Web Dashboard](#web-dashboard)
3. [CLI Commands](#cli-commands)
4. [Configuration Examples](#configuration-examples)
5. [API Examples](#api-examples)
6. [Advanced Scenarios](#advanced-scenarios)

---

## Basic Usage

### First-Time Setup

```bash
# Navigate to project directory
cd AIFILEORGANISER

# Install dependencies
pip install -r requirements.txt

# Activate license
python src/main.py --activate DEMO-AAAA-BBBB-CCCC

# Start dashboard
python src/main.py dashboard
```

### Quick Test Run

```bash
# Scan Desktop folder (dry run mode)
python src/main.py scan

# View results
python src/main.py stats
```

---

## Web Dashboard

### Starting the Dashboard

**Windows**:
```cmd
run_dashboard.bat
```

**Mac/Linux**:
```bash
chmod +x run_dashboard.sh
./run_dashboard.sh
```

**Manual**:
```bash
python src/main.py dashboard --host 0.0.0.0 --port 8080
```

### Dashboard Features

#### 1. Review Pending Files
```
1. Go to "Inbox" tab
2. See classified files waiting for approval
3. Click "✓ Approve" to execute action
4. Click "✗ Reject" to skip
```

#### 2. View Statistics
```
Dashboard automatically shows:
- Files organised: 247
- Time saved: 2.3 hours
- AI classifications: 89
- Duplicates removed: 12
```

#### 3. Find Duplicates
```
1. Go to "Duplicates" tab
2. Click "Scan for Duplicates"
3. Review duplicate groups
4. Total wasted space shown
```

#### 4. Configure Settings
```
Settings → Toggle switches:
- Auto Mode: ON/OFF
- Dry Run: ON/OFF (recommended ON for testing)
- AI Classification: ON/OFF
```

---

## CLI Commands

### Watch Mode (Background Monitoring)

```bash
# Start watching folders
python src/main.py watch

# Output:
# 👀 Starting watch mode...
# Monitoring folders:
#   - C:\Users\YourName\Desktop
#   - C:\Users\YourName\Downloads
#
# Auto mode: DISABLED
# Dry run: ENABLED
# AI classification: ENABLED
```

### Scan Existing Files

```bash
# One-time scan of existing files
python src/main.py scan

# Output:
# 🔍 Scanning existing files...
# Scanning: C:\Users\YourName\Desktop
#
# 📁 New file detected: report.pdf
#    Category: Documents
#    Suggested: Documents/PDFs/
#    Reason: Classified by file extension (.pdf)
#    Method: rule-based (high confidence)
#
# ✅ Scan complete: 47 files found
```

### Find Duplicates

```bash
# Scan for duplicates
python src/main.py duplicates

# Output:
# 🔍 Scanning for duplicate files...
# Scanning: C:\Users\YourName\Downloads
#
# ==================================================
# DUPLICATE FILES SUMMARY
# ==================================================
# Duplicate groups: 5
# Duplicate files: 12
# Wasted space: 145.23 MB (0.14 GB)
#
# TOP 10 DUPLICATE GROUPS
# --------------------------------------------------
# 1. 3 files (2048000 bytes each)
#    - C:\Users\YourName\Downloads\photo.jpg
#    - C:\Users\YourName\Desktop\photo.jpg
#    - C:\Users\YourName\Pictures\photo.jpg
```

### View Statistics

```bash
# Show overall stats
python src/main.py stats

# Output:
# ==================================================
# AI FILE ORGANISER - STATISTICS
# ==================================================
# Files organised: 247
# Time saved: 2.35 hours
# AI classifications: 89
# Duplicates removed: 12
#
# License status: ACTIVE
# Days remaining: 23
```

### License Management

```bash
# Check license status
python src/main.py license

# Activate new license
python src/main.py --activate XXXX-XXXX-XXXX-XXXX
```

---

## Configuration Examples

### Example 1: Basic Desktop/Downloads Organization

```json
{
  "watched_folders": [
    "~/Desktop",
    "~/Downloads"
  ],
  "auto_mode": false,
  "dry_run": true,
  "destination_rules": {
    "pdf": "Documents/PDFs/",
    "jpg": "Pictures/",
    "mp4": "Videos/",
    "zip": "Downloads/Archives/"
  }
}
```

### Example 2: Advanced Project Organization

```json
{
  "watched_folders": [
    "~/Desktop",
    "~/Downloads",
    "~/Documents/Inbox"
  ],
  "auto_mode": true,
  "dry_run": false,
  "destination_rules": {
    "pdf": "Documents/PDFs/",
    "docx": "Documents/Word/",
    "xlsx": "Documents/Excel/",
    "pptx": "Documents/PowerPoint/",
    "jpg": "Pictures/Photos/",
    "png": "Pictures/Screenshots/",
    "py": "Projects/Code/Python/",
    "js": "Projects/Code/JavaScript/",
    "zip": "Downloads/Archives/",
    "mp4": "Media/Videos/",
    "mp3": "Media/Music/"
  },
  "classification": {
    "enable_ai": true,
    "text_extract_limit": 1000
  },
  "time_estimates": {
    "move": 0.5,
    "rename": 0.3,
    "delete": 0.2,
    "archive": 0.4
  }
}
```

### Example 3: Developer Setup

```json
{
  "watched_folders": [
    "~/Downloads",
    "~/Desktop"
  ],
  "auto_mode": false,
  "dry_run": true,
  "destination_rules": {
    "py": "Projects/Python/",
    "js": "Projects/JavaScript/",
    "java": "Projects/Java/",
    "go": "Projects/Go/",
    "zip": "Downloads/Archives/",
    "pdf": "Documentation/",
    "md": "Documentation/Markdown/"
  },
  "ollama_model": "codellama",
  "classification": {
    "enable_ai": true,
    "text_extract_limit": 500,
    "fallback_to_rules": true
  }
}
```

---

## API Examples

### Python Integration

```python
from src.config import get_config
from src.core.classifier import FileClassifier
from src.ai.ollama_client import OllamaClient

# Initialize
config = get_config()
ollama = OllamaClient()
classifier = FileClassifier(config, ollama)

# Classify a file
result = classifier.classify("/path/to/file.pdf")

print(f"Category: {result['category']}")
print(f"Suggested path: {result['suggested_path']}")
print(f"Reason: {result['reason']}")
```

### Database Queries

```python
from src.core.db_manager import DatabaseManager

db = DatabaseManager()

# Get statistics
stats = db.get_stats('week')
print(f"Files organized this week: {stats['files_organised']}")
print(f"Time saved: {stats['time_saved_hours']} hours")

# Get recent logs
logs = db.get_recent_logs(limit=10)
for log in logs:
    print(f"{log['filename']}: {log['operation']} -> {log['new_path']}")
```

### Finding Duplicates Programmatically

```python
from src.core.duplicates import DuplicateFinder
from src.config import get_config
from src.core.db_manager import DatabaseManager

config = get_config()
db = DatabaseManager()
finder = DuplicateFinder(config, db)

# Find duplicates
duplicates = finder.find_duplicates_in_directory(
    directory="/Users/yourname/Downloads",
    recursive=True
)

# Get summary
summary = finder.get_duplicate_summary(duplicates)
print(f"Found {summary['total_duplicate_groups']} duplicate groups")
print(f"Wasted space: {summary['total_wasted_space_mb']} MB")

# Get cleanup suggestions
for group in duplicates[:5]:
    suggestion = finder.suggest_duplicates_to_keep(group)
    print(f"Keep: {suggestion['keep']}")
    print(f"Delete: {suggestion['delete']}")
```

---

## Advanced Scenarios

### Scenario 1: Organize Large Download Folder

```bash
# Step 1: Enable dry run
# Edit config.json: "dry_run": true

# Step 2: Scan and review
python src/main.py scan

# Step 3: Check pending files in dashboard
python src/main.py dashboard
# Go to Inbox tab, review suggestions

# Step 4: Enable auto mode if satisfied
# Edit config.json: "auto_mode": true, "dry_run": false

# Step 5: Run again
python src/main.py watch
```

### Scenario 2: Clean Up Duplicates

```bash
# Find duplicates
python src/main.py duplicates

# Review in dashboard
python src/main.py dashboard
# Go to Duplicates tab → Scan for Duplicates

# Manual cleanup approach:
# - Review each duplicate group
# - Decide which to keep
# - Delete duplicates manually or via dashboard
```

### Scenario 3: Custom Classification Rules

```python
# Create custom classifier extension
from src.core.classifier import FileClassifier

class CustomClassifier(FileClassifier):
    def _classify_by_patterns(self, filename, stem):
        # Add custom patterns
        if 'invoice' in stem and 'acme' in stem:
            return {
                'category': 'Finance',
                'suggested_path': 'Finance/ACME_Corp/Invoices/',
                'rename': None,
                'reason': 'ACME Corp invoice detected',
                'confidence': 'high',
                'method': 'custom-rule'
            }

        # Fallback to parent implementation
        return super()._classify_by_patterns(filename, stem)
```

### Scenario 4: Automated Workflow

```bash
#!/bin/bash
# daily_organization.sh

# Run scan
python src/main.py scan > /tmp/organiser_log.txt

# Find duplicates weekly (every Monday)
if [ $(date +%u) -eq 1 ]; then
    python src/main.py duplicates >> /tmp/organiser_log.txt
fi

# Email summary (requires mail setup)
mail -s "Daily File Organization Report" you@email.com < /tmp/organiser_log.txt
```

### Scenario 5: Integration with Cloud Backup

```python
# After organization, sync to cloud
import subprocess
from src.core.actions import ActionManager

def on_file_organized(result):
    if result['success'] and result['new_path']:
        # Sync to cloud after successful organization
        subprocess.run(['rclone', 'sync', result['new_path'], 'remote:backup/'])

# Use with watcher callback
action_manager.on_success = on_file_organized
```

---

## Testing Examples

### Test Classification Accuracy

```python
# test_classifier.py
from src.core.classifier import FileClassifier
from src.config import get_config
from src.ai.ollama_client import OllamaClient

config = get_config()
ollama = OllamaClient()
classifier = FileClassifier(config, ollama)

# Test files
test_files = [
    "/path/to/invoice_2025.pdf",
    "/path/to/photo.jpg",
    "/path/to/project_code.py",
    "/path/to/meeting_notes.docx"
]

for file in test_files:
    result = classifier.classify(file)
    print(f"\nFile: {file}")
    print(f"  Category: {result['category']}")
    print(f"  Path: {result['suggested_path']}")
    print(f"  Method: {result['method']}")
    print(f"  Confidence: {result['confidence']}")
```

### Test Dry Run vs. Real Mode

```bash
# Test with dry run
# config.json: "dry_run": true
python src/main.py scan
# Check logs - should see [DRY RUN] messages

# Test with real mode (be careful!)
# config.json: "dry_run": false
python src/main.py scan
# Files will actually be moved
```

---

## Troubleshooting Examples

### Check Ollama Connection

```python
from src.ai.ollama_client import OllamaClient

client = OllamaClient()

if client.is_available():
    print("✓ Ollama is available")
    models = client.list_models()
    print(f"Available models: {models}")
else:
    print("✗ Ollama not available")
    print("Start Ollama with: ollama serve")
```

### Verify Database

```python
from src.core.db_manager import DatabaseManager

db = DatabaseManager()

# Check if database is working
try:
    stats = db.get_stats('all')
    print("✓ Database is working")
    print(f"Total files: {stats['files_organised']}")
except Exception as e:
    print(f"✗ Database error: {e}")
```

### Reset Configuration

```bash
# Backup current config
cp config.json config.json.backup

# Reset to defaults (re-download from repo or recreate)

# Restore if needed
cp config.json.backup config.json
```

---

**For more examples and use cases, visit the documentation or join the community!**
