# 🚀 Quick Start Guide

Get up and running with the Private AI File Organiser in 5 minutes!

## Step 1: Install Prerequisites

### Install Python 3.8+
- **Windows**: Download from [python.org](https://www.python.org/downloads/)
- **Mac**: `brew install python3`
- **Linux**: `sudo apt install python3 python3-pip`

### Install Ollama (for AI features)
1. Download from [ollama.ai](https://ollama.ai)
2. Install and start Ollama
3. Pull a model:
   ```bash
   ollama pull llama3
   ```

## Step 2: Install Dependencies

```bash
cd AIFILEORGANISER
pip install -r requirements.txt
```

## Step 3: Get a License Key

You'll need one of the 200 limited early access keys:
- Request from: [Your Website]
- Or use test key: `DEMO-AAAA-BBBB-CCCC`

## Step 4: Run the Application

### Option A: Web Dashboard (Recommended)

```bash
python src/main.py dashboard
```

Then open: http://localhost:5000

### Option B: Command Line

```bash
# Watch folders and auto-organize
python src/main.py watch

# Scan existing files once
python src/main.py scan

# Find duplicates
python src/main.py duplicates

# View statistics
python src/main.py stats
```

## Step 5: Activate License

### Via Dashboard
1. Go to License tab
2. Enter your license key
3. Click Activate

### Via CLI
```bash
python src/main.py --activate XXXX-XXXX-XXXX-XXXX
```

## Step 6: Configure (Optional)

Edit `config.json` to customize:

```json
{
  "watched_folders": ["~/Desktop", "~/Downloads"],
  "auto_mode": false,
  "dry_run": true,
  "ollama_model": "llama3"
}
```

**Important Settings:**
- `auto_mode: false` - Requires manual approval (safe)
- `dry_run: true` - Simulates actions only (recommended for testing)
- `enable_ai: true` - Uses AI for classification

## Step 7: Test It Out

1. **Enable Dry Run** (in Settings tab)
2. Drop a test file in your Desktop or Downloads
3. Check the Inbox tab
4. Review the suggested classification
5. Approve to execute (or reject to skip)

## Common Issues

### "Ollama not available"
```bash
# Start Ollama
ollama serve

# Verify it's running
curl http://localhost:11434/api/tags
```

### "License validation failed"
- Check your internet connection (for online validation)
- Verify key format: XXXX-XXXX-XXXX-XXXX
- Contact support if issues persist

### "Permission denied" when moving files
- Run with appropriate permissions
- Check folder access rights
- Ensure destination folders are writable

## Next Steps

1. ✅ Review the [README.md](README.md) for full documentation
2. ✅ Join the community (link in README)
3. ✅ Share feedback and suggestions
4. ✅ Customize classification rules for your workflow

## Tips for Best Experience

1. **Start Small**: Test with one folder first
2. **Use Dry Run**: Always test with `dry_run: true` initially
3. **Review Regularly**: Check the History tab
4. **Customize Rules**: Add your own destination rules
5. **Monitor Stats**: Track your time savings!

---

**Need Help?**
- 📖 Full docs: [README.md](README.md)
- 🐛 Issues: [GitHub Issues]
- 📧 Email: support@yourproject.com

**Enjoy your newly organized files! 🎉**
